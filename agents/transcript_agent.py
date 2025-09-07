from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Optional, Tuple
import logging

from utils.security import security
from utils.cache import cache_transcript, get_cached_transcript
from utils.logger import get_logger, timer
from config import config

logger = get_logger("transcript_agent")

def extract_video_id(url: str) -> str:
    """Get the video ID from a YouTube URL"""
    try:
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif "youtu.be" in parsed.netloc:
            video_id = parsed.path.lstrip("/")
        else:
            raise ValueError(f"Not a YouTube URL: {url}")
        
        if not video_id or len(video_id) != 11:
            raise ValueError(f"Invalid video ID: {video_id}")
        
        return video_id
    except Exception as e:
        logger.error(f"Failed to extract video ID from {url}: {e}")
        raise ValueError(f"Invalid YouTube URL format: {url}")

def get_transcript(video_url: str) -> List[str]:
    """Get transcript from YouTube, using cache if available"""
    timer.start("get_transcript")
    
    try:
        # Check cache first
        cached = get_cached_transcript(video_url)
        if cached:
            logger.info("Got transcript from cache")
            timer.end("get_transcript", {"source": "cache"})
            return cached
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        logger.info(f"Fetching transcript for video: {video_id}")
        
        # Get transcript from YouTube
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not transcript_data:
            raise ValueError("No transcript data received")
        
        # Extract text and check length
        transcript_list = []
        total_length = 0
        
        for entry in transcript_data:
            if "text" in entry and entry["text"].strip():
                text = entry["text"].strip()
                transcript_list.append(text)
                total_length += len(text)
        
        # Check if transcript is too long
        if total_length > config.MAX_TRANSCRIPT_LENGTH:
            logger.warning(f"Transcript too long: {total_length} characters")
            raise ValueError(f"Transcript too long ({total_length} chars). Max allowed: {config.MAX_TRANSCRIPT_LENGTH}")
        
        if not transcript_list:
            raise ValueError("Transcript is empty")
        
        # Save to cache
        cache_transcript(video_url, transcript_list)
        
        logger.info(f"Transcript fetched: {len(transcript_list)} segments, {total_length} characters")
        timer.end("get_transcript", {
            "source": "youtube",
            "segments": len(transcript_list),
            "total_length": total_length
        })
        
        return transcript_list
        
    except Exception as e:
        logger.error(f"Failed to get transcript for {video_url}: {e}")
        timer.end("get_transcript", {"error": str(e)})
        raise

def get_transcript_info(video_url: str) -> Optional[dict]:
    """Get basic info about the transcript"""
    try:
        video_id = extract_video_id(video_url)
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        
        if transcript_data:
            # Calculate duration
            total_duration = 0
            if transcript_data:
                last_entry = transcript_data[-1]
                total_duration = last_entry.get("start", 0) + last_entry.get("duration", 0)
            
            return {
                "video_id": video_id,
                "segments": len(transcript_data),
                "duration": total_duration,
                "languages": list(set(entry.get("language", "unknown") for entry in transcript_data))
            }
    except Exception as e:
        logger.error(f"Failed to get transcript info for {video_url}: {e}")
        return None


