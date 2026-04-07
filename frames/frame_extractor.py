import cv2
import os
import json
import re

def extract_frames(video_path: str, save_dir: str = "data/frames", interval: int = 2):
    """
    Extract frames every 'interval' seconds and store timestamps

    Args:
        video_path (str)
        save_dir (str)
        interval (int): seconds between frames

    Returns:
        list: frame metadata (path + timestamp)
    """

    os.makedirs(save_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Clean filename (remove special chars)
    video_name = re.sub(r'[^\w\s-]', '', video_name)
    video_name = video_name.replace(" ", "_")
    frame_dir = os.path.join(save_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    metadata_path = os.path.join(frame_dir, "frames.json")

    # 🔥 CACHING
    if os.path.exists(metadata_path):
        print(f"[CACHE] Frames already extracted: {frame_dir}")
        with open(metadata_path, "r") as f:
            return json.load(f)

    print("Extracting frames...")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_data = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps

            frame_filename = f"frame_{saved_count}.jpg"
            frame_path = os.path.join(frame_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            if frame_data and abs(frame_data[-1]["timestamp"] - timestamp) < 0.5:
                continue
            frame_data.append({
                "frame": frame_path.replace("\\", "/"),
                "timestamp": round(timestamp, 2)
            })

            saved_count += 1

        frame_count += 1

    cap.release()

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(frame_data, f, indent=2)

    print(f"Frames saved in: {frame_dir}")

    return frame_data