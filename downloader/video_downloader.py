import os
from yt_dlp import YoutubeDL


def download_video(url: str, save_dir: str = "data/videos") -> str:
    """
    Downloads a video from a URL and saves it locally.
    
    Args:
        url (str): Video URL (YouTube or others)
        save_dir (str): Directory to save video
    
    Returns:
        str: Path to downloaded video file
    """

    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(save_dir, "%(title)s.%(ext)s"),
        "cookiefile": "D:/video_search/cookies.txt",
        # ✅ FIXED
        "js_runtimes": {
            "node": {}
        },
        "remote_components": "ejs:github",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    return video_path