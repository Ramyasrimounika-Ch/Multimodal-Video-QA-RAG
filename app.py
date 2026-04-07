import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
import os
os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/transformers"
import streamlit as st
from PIL import Image
import torch
import yt_dlp
from downloader.video_downloader import download_video
from audio.audio_extractor import extract_audio
from transcription.transcriber import transcribe_audio
from frames.frame_extractor import extract_frames
from alignment.aligner import align_frames_transcript
from chunking.chunker import create_chunks
from embedding.embedder import create_embeddings,image_embed_model
from retrieval.retriever import overlap_time_filter
from image_data.handle_images import handle_image_input,explain_image_query
from retrieval.retriever import (
    retrieve_chunks,
    deduplicate_chunks,
    rerank_chunks,
    merge_chunks,
    build_prompt,
    extract_time_range
)
import hashlib

def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()


def get_video_metadata(url):
    ydl_opts = {"quiet": True, "skip_download": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            return {
                "title": info.get("title", ""),
                "description": info.get("description", ""),
                "duration": info.get("duration", 0)  # 🔥 ADD THIS
            }
    except:
        return {"title": "", "description": "", "duration": 0}
    
# 🔥 LangChain + Groq
from langchain.chat_models import init_chat_model

@st.cache_resource
def load_llm():
    return init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq"
    )

llm = load_llm()

# -------------------------------
# 🔥 INTENT CLASSIFIER
# -------------------------------
def classify_query_intent(query):
    q = query.lower()

    # 🔥 RULE-BASED OVERRIDE (VERY IMPORTANT)
    time_keywords = ["beginning", "middle", "end", "at", "between", "from", "-", ":"]
    
    if any(k in q for k in time_keywords):
        return "TIME"

    # 🔥 LLM fallback
    prompt = f"""
Classify the user query into ONE category:

1. TIME → specific timestamp queries
2. SUMMARY → overall video summary / main idea
3. QA → specific question about content
4. FRAME → visual / scene / object queries

Query: "{query}"

Return ONLY one word:
TIME / SUMMARY / QA / FRAME
"""

    response = llm.invoke(prompt)
    intent = response.content.strip().upper()

    if intent not in ["TIME", "SUMMARY", "QA", "FRAME"]:
        return "QA"
    return intent
# -------------------------------
# UI
# -------------------------------
st.title("🎬 Video Search & QA System")

video_url = st.text_input("Enter Video URL")

# -------------------------------
# PROCESS VIDEO
# -------------------------------
if st.button("Process Video"):
    with st.spinner("Processing video..."):
        
        video_path = download_video(video_url)
        video_metadata=get_video_metadata(video_url)
        video_title=video_metadata["title"]
        video_description=video_metadata["description"][:1000]
        video_duration=video_metadata["duration"]
        audio_path = extract_audio(video_path)
        transcript = transcribe_audio(audio_path)
        frames = extract_frames(video_path)
        aligned = align_frames_transcript(transcript, frames)
        chunks = create_chunks(aligned)

        text_col, image_col, ocr_col = create_embeddings(chunks)

        st.session_state.text_col = text_col
        st.session_state.image_col = image_col
        st.session_state.ocr_col = ocr_col
        st.session_state.video_path = video_path
        st.session_state.video_title = video_title
        st.session_state.video_description = video_description
        st.session_state.video_duration = video_metadata["duration"]

    st.success("✅ Video processed successfully!")

# -------------------------------
# QUERY SECTION
# -------------------------------
if "text_col" in st.session_state:
    text_col = st.session_state.text_col
    image_col = st.session_state.image_col
    ocr_col = st.session_state.ocr_col

    st.subheader("🖼️ Image Query")

uploaded_image = st.file_uploader(
    "Upload Image", 
    type=["png", "jpg", "jpeg"], 
    key="image_uploader"
)

image_query = st.text_input("Ask something about the image")

if uploaded_image:

    # 🔥 Load image ONCE
    image = Image.open(uploaded_image)

    # 🔥 Compute hash
    image_hash = get_image_hash(image)

    # 🔥 Detect change
    if st.session_state.get("last_image_hash") != image_hash:
        st.session_state.last_image_hash = image_hash
        st.session_state.image_changed = True
    else:
        st.session_state.image_changed = False

    # 🔥 Button
    run_query = st.button("image_answer")

    # 🔥 Decide execution
    if run_query:

        st.session_state.image_changed = False  # reset

        # 🔥 DEBUG
        st.write("🆕 Processing new image...")

        if image_query.strip() == "":
            results, error = handle_image_input(
                image, "", image_col, text_col, ocr_col
            )

            if error:
                st.warning(error)
            else:
                st.subheader("⏱️ Matching Segments")
                for r in results:
                    st.write(f"{r['start']}s → {r['end']}s")
                    st.image(r["frames"][0], width=250)

        else:
            answer = explain_image_query(
                image, image_query, image_col, text_col, ocr_col
            )

            st.subheader("🧠 Answer")
            st.write(answer)
    st.subheader("🔎 Ask about Video")        
    query = st.text_input("Ask a question about the video")

    if st.button("Search"):
        with st.spinner("Searching..."):

            if not query:
                st.warning("Please enter a query")
                st.stop()


            # 🔥 STEP 1: Intent classification
            intent = classify_query_intent(query)
            tr = extract_time_range(query, st.session_state.video_duration)

            if not tr.is_semantic:
                time_range = (tr.start, tr.end)
                query = tr.intent   # cleaned query
            else:
                time_range = None
            print("Intent:", intent)
            print("Time Range:", time_range)
            
            # -------------------------------
            # 🔥 TEXT-BASED QUERY
            # -------------------------------
            if intent == "QA":
                results = retrieve_chunks(query, text_col, image_col, ocr_col)

                results = overlap_time_filter(results, time_range)
                results = deduplicate_chunks(results)
                results = rerank_chunks(query, results, top_k=5)
                results = merge_chunks(results)

                if not results:
                    st.warning("No relevant results found")
                    st.stop()

                prompt = build_prompt(
                    results,
                    query,
                    video_title=st.session_state.video_title,
                    video_description=st.session_state.video_description
                )

                answer = llm.invoke(prompt).content

                st.subheader("🧠 Answer")
                st.write(answer)

                st.subheader("⏱️ Supporting Segments")

                for i, r in enumerate(results):
                    st.markdown(f"### Segment {i+1}")
                    st.write(f"⏱️ {r['start']}s → {r['end']}s")

                    st.video(
                        st.session_state.video_path,
                        start_time=int(r["start"])
                    )

                    if r["frames"]:
                        st.image(r["frames"][:3], width=250)

                    st.markdown("---")
            elif intent == "SUMMARY":

                # 🔥 Get ALL chunks
                data = text_col.get()
                results = [
                    {
                        "text": doc,
                        "start": meta["start"],
                        "end": meta["end"],
                        "frames": meta.get("frames", [])
                    }
                    for doc, meta in zip(data["documents"], data["metadatas"])
                ]
            
                results = overlap_time_filter(results, time_range)

                # Deduplicate
                results = deduplicate_chunks(results)

                # Rerank (keep best)
                results = rerank_chunks(query, results, top_k=12)

                # Merge
                results = merge_chunks(results)
                if not results:
                    st.warning("No relevant segments for summary")
                    st.stop()

                prompt = build_prompt(results, query,video_title=st.session_state.video_title,video_description=st.session_state.video_description)
                answer = llm.invoke(prompt).content
                st.subheader("📘 Summary")
                st.write(answer)
            # -------------------------------
            # 🔥 FRAME-BASED QUERY
            # -------------------------------
            elif intent == "FRAME":
                
                clip_embedding_model=image_embed_model();
                query_embedding=clip_embedding_model([query])

                image_results = image_col.query(
                    query_embeddings=query_embedding,
                    n_results=10
                )
                metas = image_results["metadatas"][0]

                if time_range:
                    start, end = time_range
                    metas = [
                        m for m in metas
                        if not (m["end"] <= start or m["start"] >= end)
                    ]

                if not metas:
                    st.warning("No frames found")
                    st.stop()

                st.subheader("🖼️ Relevant Frames")

                for meta in metas:   # ✅ CORRECT
                    st.image(meta["frame_path"], width=300)
            elif intent == "TIME":
                data = text_col.get()

                results = [
                    {
                        "text": doc,
                        "start": meta["start"],
                        "end": meta["end"],
                        "frames": meta.get("frames", [])
                    }
                    for doc, meta in zip(data["documents"], data["metadatas"])
                ]

                # 🔥 STEP 2: Filter by time
                results = overlap_time_filter(results, time_range)
                results = deduplicate_chunks(results)
                results = rerank_chunks(query, results, top_k=5)
                results = merge_chunks(results)
                if not results:
                    st.warning("No relevant results found")
                    print(".")
                    st.stop()

                prompt = build_prompt(
                    results,
                    query,
                    video_title=st.session_state.video_title,
                    video_description=st.session_state.video_description
                )

                answer = llm.invoke(prompt).content

                st.subheader("🧠 Answer")
                st.write(answer)

                st.subheader("🖼️ Frames in this time range")

                for i, r in enumerate(results):
                    st.markdown(f"### Segment {i+1}")
                    st.write(f"⏱️ {r['start']}s → {r['end']}s")

                    # show only first few frames (avoid overload)
                    if r["frames"]:
                        st.image(r["frames"][:4], width=250)
            
                st.subheader("🎬 Video Playback")

                for r in results:
                    st.video(
                        st.session_state.video_path,
                        start_time=int(r["start"])
                    )