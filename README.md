# 🎬 Multimodal Video Search & QA System

An end-to-end AI system that enables intelligent search and question answering over videos using text, audio, and visual understanding.

## 🚀 Features
🔍 Semantic Search over Video Content

🧠 Question Answering using LLMs

⏱️ Time-aware Retrieval (timestamp-based queries)

🖼️ Frame-based Visual Search (CLIP)

🔤 OCR-based Text Search inside Frames

📘 Automatic Video Summarization

⚡ Fast Retrieval using Vector Database (ChromaDB)

## 🧠 System Architecture
```bash 
Video URL 
   ↓
Download (yt-dlp) 
   ↓ 
Audio Extraction (FFmpeg)
   ↓ 
Transcription (Whisper) 
   ↓ 
Frame Extraction (OpenCV) 
   ↓ 
OCR (Tesseract) 
   ↓
Alignment (Text + Frames) 
   ↓ 
Chunking (with overlap) 
   ↓
Embeddings: 
   - Text → SentenceTransformers
   - Image → OpenCLIP
   - OCR → SentenceTransformers
   ↓
Vector DB (ChromaDB)
   ↓
Query Pipeline:
   - Intent Classification (LLM)
   - Time Extraction (LLM)
   - Retrieval (Text + Image + OCR)
   - Filtering + Deduplication
   - Reranking (Relevance + Importance + Position)
   - Merging
   ↓
LLM Answer Generation (Groq - LLaMA 3)
```
## 🛠️ Tech Stack
Frontend: Streamlit

LLM: Groq (LLaMA 3.3 70B)

Speech-to-Text: Whisper

Video Processing: OpenCV, FFmpeg

OCR: Tesseract

Embeddings:

SentenceTransformers (text)

OpenCLIP (image)

Vector DB: ChromaDB

Downloader: yt-dlp

## 📂 Project Structure
```bash 
├── downloader/
│   └── video_downloader.py
├── audio/
│   └── audio_extractor.py
├── transcription/
│   └── transcriber.py
├── frames/
│   └── frame_extractor.py
├── alignment/
│   └── aligner.py
├── chunking/
│   └── chunker.py
├── embedding/
│   └── embedder.py
├── retrieval/
│   └── retriever.py
├── app.py
└── README.md
```

## ⚙️ Installation
1. Clone the repository
``` python
git clone https://github.com/Ramyasrimounika-Ch/Multimodal-Video-QA-RAG.git
cd Multimodal-Video-QA-RAG
```

2. Install dependencies
``` python
pip install -r requirements.txt
```
3. Install external tools
```python
Install FFmpeg
Install Tesseract OCR
```

4. Set environment variables
```python
export GROQ_API_KEY=your_api_key
```

## ▶️ Running the App
```python
streamlit run app.py
```

## 💡 How It Works
**1. Process Video**

Enter YouTube URL

System extracts:

Audio → transcription

Frames → visual data

OCR → text from images

**2. Ask Questions**

#### Supported queries:

🧠 "What is the main idea?"

⏱️ "What happens at 2 minutes?"

🖼️ "Show the diagram explained"

📘 "Summarize the video"

🔍 Retrieval Strategy

Hybrid Retrieval:

  - Text embeddings
   
  - OCR embeddings
    
  - Image embeddings (CLIP)
    
Reranking based on:

  - Query relevance
    
  - Semantic importance
    
  - Temporal position in video
    
Time-aware filtering with buffer for better context

## ⚡ Optimizations

✅ Caching (Streamlit cache_resource)

✅ OCR parallelization

✅ Deduplication of chunks

✅ Batch image embedding

✅ Persistent vector DB

## 📌 Future Improvements

🔥 Real-time streaming support

🔥 Chat history (multi-turn conversation)

🔥 Multi-video search

🔥 Better visual grounding
