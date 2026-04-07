import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"
import os
os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/transformers"
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenCLIPEmbeddingFunction
)
import cv2
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# -------------------------------
# 🔥 CHROMA CLIENT (PERSISTENT)
# -------------------------------
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="chroma_db"
    )
)

# -------------------------------
# 🔹 TEXT + OCR EMBEDDING
# -------------------------------
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

text_collection = client.get_or_create_collection(
    name="text_data",
    embedding_function=get_embedding_function()
)

ocr_collection = client.get_or_create_collection(
    name="ocr_data",
    embedding_function=get_embedding_function()
)

# -------------------------------
# 🔹 IMAGE EMBEDDING (OPENCLIP 🔥)
# -------------------------------
@st.cache_resource
def image_embed_model():
    return OpenCLIPEmbeddingFunction(
            model_name="ViT-H-14",
            checkpoint="laion2b_s32b_b79k"
        )

image_collection = client.get_or_create_collection(
    name="image_data",
    embedding_function=image_embed_model()
)

# -------------------------------
# 🔹 OCR HELPERS
# -------------------------------
def extract_text_from_image(frame):
    try:
        img = cv2.imread(frame)
        text = pytesseract.image_to_string(img, config="--psm 6")
        return frame, text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return frame, ""

def parallel_ocr(frames, max_workers=4):
    ocr_results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(extract_text_from_image, frames)

        for frame, text in results:
            ocr_results[frame] = text

    return ocr_results

# -------------------------------
# 🔥 MAIN FUNCTION
# -------------------------------
def create_embeddings(chunks):

    seen_texts = set()
    seen_images = set()
    seen_ocr = set()

    images_batch = []
    metas_batch = []
    ids_batch = []

    # -------------------------------
    # 🔥 OCR PREPROCESS (PARALLEL)
    # -------------------------------
    all_frames = []
    for chunk in chunks:
        all_frames.extend(chunk["frames"])

    all_frames = list(set(all_frames))
    ocr_cache = parallel_ocr(all_frames)

    # -------------------------------
    # 🔥 PROCESS CHUNKS
    # -------------------------------
    for chunk in chunks:

        text = chunk["text"].strip()

        # 🔹 TEXT EMBEDDING
        if text and text not in seen_texts and len(text.split()) >= 3:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "duration": chunk["end"] - chunk["start"]
            }

            if chunk["frames"]:
                metadata["frames"] = chunk["frames"]

            text_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[f"text_{chunk['chunk_id']}"]
            )

            seen_texts.add(text)

        # 🔹 IMAGE + OCR
        for i, frame in enumerate(chunk["frames"]):

            # -------------------------------
            # 🔥 IMAGE (OpenCLIP auto embedding)
            # -------------------------------
            if frame not in seen_images:
                try:
                    img = Image.open(frame).convert("RGB")
                    img_np = np.array(img)

                    images_batch.append(img_np)

                    metas_batch.append({
                        "chunk_id": chunk["chunk_id"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "frame_path": frame
                    })

                    ids_batch.append(f"{chunk['chunk_id']}_img_{i}")

                    seen_images.add(frame)

                except Exception as e:
                    print(f"Image error: {e}")

            # -------------------------------
            # 🔥 OCR
            # -------------------------------
            ocr_text = ocr_cache.get(frame, "").strip()

            if len(ocr_text) > 10 and ocr_text not in seen_ocr:
                ocr_collection.add(
                    documents=[ocr_text],
                    metadatas=[{
                        "chunk_id": chunk["chunk_id"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "frame_path": frame
                    }],
                    ids=[f"ocr_{chunk['chunk_id']}_{i}"]
                )

                seen_ocr.add(ocr_text)

    # -------------------------------
    # 🔥 ADD IMAGE DATA (BATCH)
    # -------------------------------
    if images_batch:
        image_collection.add(
            images=images_batch,
            metadatas=metas_batch,
            ids=ids_batch
        )

    return text_collection, image_collection, ocr_collection
