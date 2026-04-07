import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chat_models import init_chat_model
from embedding.embedder import image_embed_model
from difflib import SequenceMatcher
import streamlit as st
import json
import re

"""
Robust Time Range Extractor (Production Ready)
"""

import re
from dataclasses import dataclass
from typing import Optional


# ─── Data Model ─────────────────────────────────────────────────────────────

@dataclass
class TimeRange:
    start: Optional[float]
    end: Optional[float]
    is_semantic: bool = False
    intent: str = ""
    confidence: float = 1.0
    source: str = "rule"


# ─── Helpers ────────────────────────────────────────────────────────────────

def _to_seconds(value: float, unit: str) -> float:
    unit = (unit or "s").lower().strip()
    if unit in ("min", "mins", "minute", "minutes", "m"):
        return value * 60
    return value


def _point_to_range(t: float, window: float = 10.0):
    return (max(0.0, t - window / 2), t + window / 2)


def _normalize(start: float, end: float):
    return (min(start, end), max(start, end))


def _clamp(start, end, duration):
    if duration:
        start = max(0, min(start, duration))
        end = max(0, min(end, duration))
    return start, end


def _strip_time_tokens(query: str):
    patterns = [
        r'\b(?:between|from)\s+[\d:.]+\s*(?:min(?:ute)?s?|secs?|seconds?)?\s*(?:and|to)\s+[\d:.]+\s*(?:min(?:ute)?s?|secs?|seconds?)?\b',
        r'\b(?:first|last)\s+\d+\s*(?:min(?:ute)?s?|secs?|seconds?)\b',
        r'\b(?:after|before|around|near|about)\s+[\d:.]+\s*(?:min(?:ute)?s?|secs?|seconds?)?\b',
        r'\b\d+:\d+\b',
        r'\bat\s+[\d:.]+\s*(?:min(?:ute)?s?|secs?|seconds?)?\b',
        r'\b\d+\s*[-–]\s*\d+\s*(?:min(?:ute)?s?|secs?|seconds?)?\b',
        r'\b(?:beginning|start of|start|middle|halfway|end of|end|last part)\b',
    ]
    result = query
    for p in patterns:
        result = re.sub(p, '', result, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', result).strip(" ,;:-")


# ─── Main Function ──────────────────────────────────────────────────────────

def extract_time_range(query: str, duration: Optional[float] = None):
    q = query.strip()

    U = r'(?:min(?:ute)?s?|secs?|seconds?|s|m)'

    # ── 1. mm:ss ────────────────────────────────────────────────────────────
    mm_ss = re.search(r'\b(\d+):(\d{1,2})\b', q)
    if mm_ss:
        t = int(mm_ss.group(1)) * 60 + int(mm_ss.group(2))
        start, end = _point_to_range(t)
        start, end = _clamp(start, end, duration)
        return TimeRange(start, end, intent=_strip_time_tokens(q), source="mm:ss")

    # ── 2. explicit ranges ──────────────────────────────────────────────────
    range_pat = re.search(
        r'(?:between|from)\s+(\d+(?:\.\d+)?)\s*(' + U + r')?\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*(' + U + r')?',
        q, re.IGNORECASE
    )

    if not range_pat:
        range_pat = re.search(
            r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(' + U + r')?',
            q, re.IGNORECASE
        )
        if range_pat:
            v1, v2 = float(range_pat.group(1)), float(range_pat.group(2))
            unit = range_pat.group(3) or "s"
            start, end = _normalize(_to_seconds(v1, unit), _to_seconds(v2, unit))
            start, end = _clamp(start, end, duration)
            return TimeRange(start, end, intent=_strip_time_tokens(q), source="range")

    if range_pat and len(range_pat.groups()) == 4:
        v1 = float(range_pat.group(1))
        u1 = range_pat.group(2) or "s"
        v2 = float(range_pat.group(3))
        u2 = range_pat.group(4) or u1

        start = _to_seconds(v1, u1)
        end = _to_seconds(v2, u2)

        start, end = _normalize(start, end)
        start, end = _clamp(start, end, duration)

        return TimeRange(start, end, intent=_strip_time_tokens(q), source="range")

    # ── 3. first ────────────────────────────────────────────────────────────
    first_pat = re.search(r'\bfirst\s+(\d+(?:\.\d+)?)\s*(' + U + r')\b', q, re.IGNORECASE)
    if first_pat:
        t = _to_seconds(float(first_pat.group(1)), first_pat.group(2))
        start, end = _clamp(0, t, duration)
        return TimeRange(start, end, intent=_strip_time_tokens(q), source="first")

    # ── 4. last ─────────────────────────────────────────────────────────────
    last_pat = re.search(r'\blast\s+(\d+(?:\.\d+)?)\s*(' + U + r')\b', q, re.IGNORECASE)
    if last_pat:
        t = _to_seconds(float(last_pat.group(1)), last_pat.group(2))
        if duration:
            start, end = _clamp(duration - t, duration, duration)
            return TimeRange(start, end, intent=_strip_time_tokens(q), source="last")
        return TimeRange(None, None, is_semantic=True, intent=_strip_time_tokens(q), confidence=0.3)

    # ── 5. relative: beginning ─────────────────────────────────────────────
    if re.search(r'\b(beginning|start\s*of|at\s+the\s+start)\b', q, re.IGNORECASE):
        end = duration * 0.1 if duration else 30
        return TimeRange(0, end, intent=_strip_time_tokens(q), source="relative")

    # ── 6. middle ───────────────────────────────────────────────────────────
    if re.search(r'\b(middle|halfway)\b', q, re.IGNORECASE):
        if duration:
            mid = duration / 2
            start, end = _point_to_range(mid)
            return TimeRange(start, end, intent=_strip_time_tokens(q), source="relative")
        return TimeRange(None, None, is_semantic=True, intent=_strip_time_tokens(q))

    # ── 7. end ──────────────────────────────────────────────────────────────
    if re.search(r'\b(end|last part)\b', q, re.IGNORECASE):
        if duration:
            start = duration * 0.8
            return TimeRange(start, duration, intent=_strip_time_tokens(q), source="relative")
        return TimeRange(None, None, is_semantic=True, intent=_strip_time_tokens(q))

    # ── 8. before ───────────────────────────────────────────────────────────
    before_pat = re.search(r'\bbefore\s+(\d+(?:\.\d+)?)\s*(' + U + r')\b', q, re.IGNORECASE)
    if before_pat:
        t = _to_seconds(float(before_pat.group(1)), before_pat.group(2))
        return TimeRange(0, t, intent=_strip_time_tokens(q), source="before")

    # ── 9. after ────────────────────────────────────────────────────────────
    after_pat = re.search(r'\bafter\s+(\d+(?:\.\d+)?)\s*(' + U + r')\b', q, re.IGNORECASE)
    if after_pat:
        t = _to_seconds(float(after_pat.group(1)), after_pat.group(2))
        return TimeRange(t, t + 20, intent=_strip_time_tokens(q), source="after")

    # ── 10. approx ──────────────────────────────────────────────────────────
    approx_pat = re.search(r'\b(around|near|about)\s+(\d+(?:\.\d+)?)\s*(' + U + r')\b', q, re.IGNORECASE)
    if approx_pat:
        t = _to_seconds(float(approx_pat.group(2)), approx_pat.group(3))
        start, end = _point_to_range(t, 15)
        return TimeRange(start, end, intent=_strip_time_tokens(q), source="approx")

    # ── 11. at ──────────────────────────────────────────────────────────────
    at_pat = re.search(r'\bat\s+(\d+(?:\.\d+)?)\s*(' + U + r')?\b', q, re.IGNORECASE)
    if at_pat:
        t = _to_seconds(float(at_pat.group(1)), at_pat.group(2) or "s")

        # 🔥 FIX: avoid "top 10" type false positives
        if at_pat.group(2) is None and "second" not in q and "minute" not in q:
            return TimeRange(None, None, is_semantic=True, intent=q, confidence=0.2)

        start, end = _point_to_range(t)
        return TimeRange(start, end, intent=_strip_time_tokens(q), source="point")

    # ── 12. semantic fallback ───────────────────────────────────────────────
    return TimeRange(None, None, is_semantic=True, intent=q, confidence=0.1)

# initialize once
llm = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq"
)
 

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()
def overlap_time_filter(chunks, time_range):
    """
    Return chunks overlapping with time range
    """
    if not time_range:
        return chunks

    start, end = time_range

    return [
        c for c in chunks
        if not (c["end"] <= start or c["start"] >= end)
    ]

def retrieve_chunks(query, text_col, image_col, ocr_col,top_k=3):

    query_lower = query.lower()

    results = []

    # 🔹 TEXT SEARCH
    text_results = text_col.query(query_texts=[query], n_results=top_k)

    # 🔹 OCR SEARCH
    ocr_results = ocr_col.query(query_texts=[query], n_results=top_k)

    # 🔹 IMAGE SEARCH
    # 🔹 IMAGE SEARCH (CLIP)
    
    clipem=image_embed_model()
    query_embedding=clipem([query])
    image_results = image_col.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # 🔥 Combine all
    for doc, meta in zip(text_results["documents"][0], text_results["metadatas"][0]):
        results.append({
            "text": doc,
            "start": meta["start"],
            "end": meta["end"],
            "frames": meta.get("frames", [])
        })

    for doc, meta in zip(ocr_results["documents"][0], ocr_results["metadatas"][0]):
        results.append({
            "text": doc,
            "start": meta["start"],
            "end": meta["end"],
            "frames": [meta["frame_path"]]
        })

    for doc, meta in zip(image_results["documents"][0], image_results["metadatas"][0]):
        results.append({
            "text": f"Image showing: {query}",
            "start": meta["start"],
            "end": meta["end"],
            "frames": [meta["frame_path"]]
        })

    

    return results

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold


def deduplicate_chunks(chunks):
    unique = []

    for chunk in chunks:
        is_dup = False

        for u in unique:
            if is_similar(chunk["text"], u["text"]):
                is_dup = True
                break

        if not is_dup:
            unique.append(chunk)

    return unique
def rerank_chunks(query, chunks, top_k=5):
    """
    Re-rank chunks based on:
    1. Query similarity
    2. Semantic importance (main idea / insight)
    """

    # Encode query
    query_emb = model.encode([query])[0]

    # Encode all chunks
    texts = [c["text"] for c in chunks]
    chunk_embs = model.encode(texts)

    # 🔥 Importance embedding (GENERIC, works for all videos)
    importance_query = "main idea or key insight of the video"
    importance_emb = model.encode([importance_query])[0]

    final_scores = []

    video_duration = max(c["end"] for c in chunks) if chunks else 1

    for chunk, chunk_emb in zip(chunks, chunk_embs):

        # 1️⃣ Relevance score (query match)
        relevance_score = np.dot(chunk_emb, query_emb)

        # 2️⃣ Importance score (insight detection)
        importance_score = np.dot(chunk_emb, importance_emb)

        # 🔥 Final score (tunable weights)
        time_bonus = 0.1 if "start" in chunk else 0

        score = (0.7 * relevance_score) + (0.3 * importance_score) + time_bonus

        # 🔥 POSITION-AWARE BOOST
        

        position = chunk["start"] / video_duration if video_duration else 0

            # 🔥 BOOST MIDDLE
        if 0.3 <= position <= 0.7:
            score += 0.1

        # 🔥 BOOST END
        elif position > 0.7:
            score += 0.05

        final_scores.append((chunk, score))    

    # Sort by score
    ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)

    return [c[0] for c in ranked[:top_k]]

def merge_chunks(chunks, gap_threshold=1):
    """
    Merge chunks that are close in time
    """

    chunks = sorted(chunks, key=lambda x: x["start"])
    if not chunks:
       return []

    merged = []
    current = chunks[0]

    for nxt in chunks[1:]:
        if nxt["start"] - current["end"] <= gap_threshold:
            # merge
            current["end"] = max(current["end"], nxt["end"])
            current["text"] += " " + nxt["text"]
            current["frames"] = list(set(current["frames"] + nxt["frames"]))
        else:
            merged.append(current)
            current = nxt

    merged.append(current)

    return merged

def build_prompt(chunks, query, video_title=None, video_description=None):
    """
    Build structured prompt for LLM with title + description
    """

    # 🔹 Build context from chunks
    context = "Context:\n\n"
    for c in chunks:
        context += f"[Time: {c['start']} - {c['end']}]\n"
        context += f"{c['text']}\n\n"

    # 🔹 Optional title
    title_section = ""
    if video_title:
        title_section = f"Video Title:\n{video_title}\n\n"

    # 🔹 Optional description (truncate if too long)
    desc_section = ""
    if video_description:
        short_desc = video_description[:1000]  # prevent token overflow
        desc_section = f"Video Description:\n{short_desc}\n\n"

    # 🔹 Final prompt
    prompt = f"""
{title_section}{desc_section}{context}

Question:
{query}

Instructions:
- Focus on the main idea or key takeaway
- Use the video title and description for better understanding
- Infer missing connections if needed
- Avoid repetition
- Answer clearly and concisely
"""

    return prompt