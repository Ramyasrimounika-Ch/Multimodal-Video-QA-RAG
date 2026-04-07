import os
import json


def align_frames_transcript(transcript_segments, frame_data, save_dir="data/aligned"):
    """
    Aligns transcript segments with corresponding frames
    """

    os.makedirs(save_dir, exist_ok=True)
    video_name = os.path.basename(os.path.dirname(frame_data[0]["frame"]))

    aligned_path = os.path.join(save_dir, f"{video_name}_aligned.json")

    # 🔥 Caching
    if os.path.exists(aligned_path):
        print(f"[CACHE] Aligned data already exists: {aligned_path}")
        with open(aligned_path, "r") as f:
            return json.load(f)

    aligned_segments = []

    for seg in transcript_segments:
        start_time = seg["start"]
        end_time = seg["end"]
        text = seg["text"]

        # Get frames within this segment
        frames_in_segment = [
            f["frame"] for f in frame_data
            if start_time <= f["timestamp"] <= end_time
        ]

        aligned_segments.append({
            "start": start_time,
            "end": end_time,
            "frames": frames_in_segment,
            "text": text
        })

    # Save aligned result
    with open(aligned_path, "w") as f:
        json.dump(aligned_segments, f, indent=2)

    print(f"Aligned segments saved at: {aligned_path}")

    return aligned_segments