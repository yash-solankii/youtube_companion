import os
from pathlib import Path

class Config:
    """Simple configuration management"""
    
    def __init__(self):
        # API Configuration
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Video limits
        self.MAX_VIDEO_LENGTH = int(os.getenv("MAX_VIDEO_LENGTH", "7200"))  # 2 hours
        self.MAX_TRANSCRIPT_LENGTH = int(os.getenv("MAX_TRANSCRIPT_LENGTH", "100000"))
        
        # Rate limiting
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
        self.RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
        
        # Text processing
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
        
        # Cache settings
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        
        # Directories
        self.LOG_DIR = Path("logs")
        self.CACHE_DIR = Path("cache")
        self.LOG_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def get_config_summary(self):
        """Get basic config summary"""
        return {
            "rate_limit": self.RATE_LIMIT_REQUESTS,
            "chunk_size": self.CHUNK_SIZE,
            "max_transcript_length": self.MAX_TRANSCRIPT_LENGTH
        }

# Create config instance
config = Config()

# Backward compatibility
GROQ_API_KEY = config.GROQ_API_KEY
