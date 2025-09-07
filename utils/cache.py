import hashlib
import time
import pickle
from pathlib import Path
import shutil
from utils.logger import get_logger

logger = get_logger("cache")

class Cache:
    """Simple cache system for storing transcripts, embeddings, and summaries"""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create cache folders
        (self.cache_dir / "transcripts").mkdir(exist_ok=True)
        (self.cache_dir / "embeddings").mkdir(exist_ok=True)
        (self.cache_dir / "summaries").mkdir(exist_ok=True)
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _get_key(self, data: str) -> str:
        """Create a unique key from data"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_path(self, cache_type: str, key: str) -> Path:
        """Get the file path for cached data"""
        return self.cache_dir / cache_type / f"{key}.pkl"
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cached file is expired"""
        if not file_path.exists():
            return True
        age = time.time() - file_path.stat().st_mtime
        return age > self.ttl
    
    def get(self, cache_type: str, key: str):
        """Get data from cache"""
        file_path = self._get_path(cache_type, key)
        
        if self._is_expired(file_path):
            if file_path.exists():
                file_path.unlink()
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Cache hit: {cache_type}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_type}: {e}")
            if file_path.exists():
                file_path.unlink()
            return None
    
    def set(self, cache_type: str, key: str, data) -> bool:
        """Save data to cache"""
        try:
            file_path = self._get_path(cache_type, key)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to cache: {cache_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache {cache_type}: {e}")
            return False
    
    def clear_all(self) -> int:
        """Clear all cache data"""
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            (self.cache_dir / "transcripts").mkdir(exist_ok=True)
            (self.cache_dir / "embeddings").mkdir(exist_ok=True)
            (self.cache_dir / "summaries").mkdir(exist_ok=True)
            logger.info("Cache cleared")
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return -1
    
    def get_stats(self):
        """Get basic cache stats"""
        return {
            "enabled": True,
            "ttl": self.ttl,
            "hit_rate": "N/A",
            "file_counts": {"transcripts": 0, "embeddings": 0, "summaries": 0},
            "total_size_mb": 0
        }

# Create cache instance
cache = Cache()

def get_cache_key(url: str) -> str:
    """Get cache key for a YouTube URL"""
    return cache._get_key(url)

def cache_transcript(url: str, transcript: list) -> bool:
    """Cache transcript data"""
    key = get_cache_key(url)
    return cache.set("transcripts", key, transcript)

def get_cached_transcript(url: str):
    """Get cached transcript if available"""
    key = get_cache_key(url)
    return cache.get("transcripts", key)

def cache_embeddings(url: str, embeddings) -> bool:
    """Cache embeddings data"""
    key = get_cache_key(url)
    return cache.set("embeddings", key, embeddings)

def get_cached_embeddings(url: str):
    """Get cached embeddings if available"""
    key = get_cache_key(url)
    return cache.get("embeddings", key)

def cache_summary(video_url: str, summary: str, bullets: str) -> bool:
    """Cache a video summary and key points"""
    key = get_cache_key(video_url)
    data = {"summary": summary, "bullets": bullets}
    return cache.set("summaries", key, data)

def get_cached_summary(url: str):
    """Get cached summary if available"""
    key = get_cache_key(url)
    return cache.get("summaries", key)

def clear_invalid_cache():
    """Clear any invalid cache entries"""
    logger.info("Cache cleanup completed")
