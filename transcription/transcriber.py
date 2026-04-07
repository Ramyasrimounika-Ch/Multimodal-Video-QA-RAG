import os
import json
import whisper
from utils.text_utils import clean_text


def transcribe_audio(audio_path: str, save_dir: str = "data/transcripts"):
    """
    Transcribes audio using Whisper with timestamps + cleaning + caching
    """

    os.makedirs(save_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = os.path.join(save_dir, f"{file_name}.json")

    # 🔥 CACHING
    if os.path.exists(transcript_path):
        print(f"[CACHE] Transcript already exists: {transcript_path}")
        with open(transcript_path, "r") as f:
            return json.load(f)

    print("Running Whisper transcription...")

    # Load model (CPU-friendly)
    model = whisper.load_model("base")

    result = model.transcribe(audio_path)

    segments = result["segments"]

    processed_segments = []

    for seg in segments:
        cleaned_text = clean_text(seg["text"])

        processed_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": cleaned_text
        })

    # Save transcript (caching)
    with open(transcript_path, "w") as f:
        json.dump(processed_segments, f, indent=2)

    print(f"Transcript saved at: {transcript_path}")

    return processed_segments