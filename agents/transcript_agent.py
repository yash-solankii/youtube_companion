# agents/transcript_agent.py
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    raise ValueError(f"Invalid YouTube URL: {url}")

def get_transcript(video_url: str) -> list[str]:
    video_id = extract_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return [entry["text"] for entry in transcript]
