import os
import json


def create_chunks(aligned_data, chunk_size=10, overlap=2, save_dir="data/chunks"):
    """
    Create overlapping chunks from aligned data

    Args:
        aligned_data (list)
        chunk_size (int): seconds
        overlap (int): seconds

    Returns:
        list: chunked data
    """

    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.basename(os.path.dirname(aligned_data[0]["frames"][0]))

    chunk_path = os.path.join(save_dir, f"{video_name}_chunks.json")

    # 🔥 CACHING (disable during debugging if needed)
    if os.path.exists(chunk_path):
        print(f"[CACHE] Chunks already exist: {chunk_path}")
        with open(chunk_path, "r") as f:
            return json.load(f)

    print("Creating chunks with overlap...")

    chunks = []
    n = len(aligned_data)

    i = 0
    chunk_id = 0

    while i < n:
        current_start = aligned_data[i]["start"]
        current_end = current_start + chunk_size

        chunk_text = []
        seen_texts = set()   # 🔥 deduplicate text inside chunk
        chunk_frames = []

        j = i

        # collect segments within chunk window
        while j < n and aligned_data[j]["start"] < current_end:
            text = aligned_data[j]["text"].strip()

            # 🔥 avoid duplicate sentences
            if text and text not in seen_texts:
                chunk_text.append(text)
                seen_texts.add(text)

            chunk_frames.extend(aligned_data[j]["frames"])
            j += 1

        # 🔥 remove duplicate frames
        chunk_frames = sorted(list(set(chunk_frames)))

        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "start": round(current_start, 2),
            "end": round(current_end, 2),
            "text": " ".join(chunk_text).strip(),  # 🔥 clean final text
            "frames": chunk_frames
        })

        chunk_id += 1

        # 🔥 move index with reduced overlap
        next_start_time = current_start + (chunk_size - overlap)

        while i < n and aligned_data[i]["start"] < next_start_time:
            i += 1

    # save chunks
    with open(chunk_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Chunks saved at: {chunk_path}")

    return chunks