import os
import subprocess


def extract_audio(video_path: str, save_dir: str = "data/audio") -> str:
    """
    Extracts audio from a video file and saves it as .wav

    Args:
        video_path (str): Path to input video
        save_dir (str): Directory to save audio

    Returns:
        str: Path to extracted audio file
    """

    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate audio file name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(save_dir, f"{video_name}.wav")

    # 🔥 CACHING: Skip if already exists
    if os.path.exists(audio_path):
        print(f"[CACHE] Audio already exists: {audio_path}")
        return audio_path

    print("Extracting audio...")

    # ffmpeg command
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",              # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",     # sample rate (important for Whisper)
        "-ac", "1",         # mono channel
        audio_path
    ]

    # Run command
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Audio saved at: {audio_path}")

    return audio_path