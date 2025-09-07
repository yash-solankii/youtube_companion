import re
import time
from typing import Tuple
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """Handles input validation and security checks for the YouTube companion app"""
    
    def __init__(self):
        self.request_counts = {}  # Track requests per user
        self.max_requests = 15  # Allow 15 requests per minute
        
        # Common attack patterns to block
        self.blocked_patterns = [
            r"<script", r"javascript:", r"on\w+\s*=", r"data:text/html",
            r"<iframe", r"<object", r"<embed", r"vbscript:",
            r"prompt\(", r"alert\(", r"confirm\(", r"eval\(",
            r"import\s+os", r"subprocess", r"rm\s+-rf", r"del\s+/s"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_patterns]
        
        # Prompt injection attempts to detect
        self.injection_patterns = [
            r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions?",
            r"forget\s+(?:all\s+)?(?:previous\s+)?instructions?",
            r"you\s+are\s+now\s+(?:a\s+)?(?:different\s+)?(?:ai|assistant|bot)",
            r"act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:a\s+)?(?:different\s+)?(?:ai|assistant|bot)",
            r"pretend\s+(?:to\s+be|you\s+are)\s+(?:a\s+)?(?:different\s+)?(?:ai|assistant|bot)",
            r"system\s+prompt", r"system\s+message", r"system\s+instruction",
            r"ignore\s+(?:the\s+)?(?:above|previous|earlier)",
            r"above\s+is\s+(?:wrong|incorrect|false|fake)",
            r"you\s+can\s+(?:now\s+)?(?:access|use|connect\s+to)\s+(?:the\s+)?(?:internet|web)",
            r"you\s+are\s+(?:now\s+)?(?:unrestricted|unlimited|free)",
            r"jailbreak", r"break\s+free", r"escape\s+restrictions",
            r"bypass\s+(?:safety|security|content|filtering)",
            r"you\s+are\s+(?:now\s+)?(?:evil|malicious|harmful|dangerous)"
        ]
        self.compiled_injection = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
    
    def validate_youtube_url(self, url: str) -> Tuple[bool, str]:
        """Check if the YouTube URL is valid and safe"""
        if not url or not isinstance(url, str):
            return False, "Please provide a valid YouTube URL"
        
        # Check URL length
        if len(url) > 300:
            return False, "URL is too long"
        
        # Look for dangerous content
        if self._has_dangerous_content(url):
            logger.warning(f"Potentially dangerous content in URL: {url[:50]}...")
            return False, "URL contains unsafe content"
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
            
            # Only allow YouTube domains
            valid_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
            if not any(domain in parsed.netloc for domain in valid_domains):
                return False, "Please provide a valid YouTube URL"
            
            # Extract and validate video ID
            if 'youtube.com' in parsed.netloc:
                video_id = parsed.query.split('v=')[-1].split('&')[0] if 'v=' in parsed.query else None
            else:  # youtu.be format
                video_id = parsed.path.strip('/')
            
            if not video_id or len(video_id) != 11:
                return False, "Invalid YouTube video ID"
                
            return True, "URL looks good"
            
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False, "Could not validate URL"
    
    def _has_dangerous_content(self, text: str) -> bool:
        """Check if text contains dangerous patterns"""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def clean_input(self, text: str) -> str:
        """Remove potentially dangerous characters from user input"""
        if not text:
            return ""
        
        # Remove dangerous characters
        dangerous = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}', '[', ']']
        cleaned = text
        for char in dangerous:
            cleaned = cleaned.replace(char, '')
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Limit length ONLY for inputs, not outputs
        if len(cleaned) > 800:
            cleaned = cleaned[:800] + "..."
        
        return cleaned
    
    def clean_output(self, text: str) -> str:
        """Clean AI output without truncating - only remove dangerous content"""
        if not text:
            return ""
        
        # Only remove truly dangerous characters, preserve normal punctuation
        dangerous = ['<script', '<iframe', '<object', '<embed', 'javascript:', 'vbscript:']
        cleaned = text
        for danger in dangerous:
            cleaned = cleaned.replace(danger, '')
        
        # Clean up extra whitespace but preserve formatting
        cleaned = re.sub(r'\n\s+\n', '\n\n', cleaned)  # Remove lines with only spaces
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Normalize spaces and tabs
        
        return cleaned.strip()
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, str]:
        """Prevent users from making too many requests too quickly"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        if user_id not in self.request_counts:
            self.request_counts[user_id] = []
        
        # Remove old requests
        self.request_counts[user_id] = [
            ts for ts in self.request_counts[user_id] 
            if ts > minute_ago
        ]
        
        # Check if limit exceeded
        if len(self.request_counts[user_id]) >= self.max_requests:
            return False, "You're making requests too quickly. Please wait a moment."
        
        # Add current request
        self.request_counts[user_id].append(current_time)
        return True, "Rate limit OK"
    
    def validate_question(self, question: str) -> Tuple[bool, str]:
        """Check if the user question is safe and appropriate"""
        if not question or not isinstance(question, str):
            return False, "Please provide a valid question"
        
        # Check length
        if len(question) > 400:
            return False, "Question is too long"
        
        # Look for dangerous content
        if self._has_dangerous_content(question):
            logger.warning(f"Dangerous content in question: {question[:50]}...")
            return False, "Question contains unsafe content"
        
        # Check for prompt injection attempts
        if self._is_prompt_injection(question):
            logger.warning(f"Prompt injection attempt: {question[:50]}...")
            return False, "Question contains inappropriate content"
        
        return True, "Question looks good"
    
    def _is_prompt_injection(self, text: str) -> bool:
        """Detect attempts to manipulate the AI's behavior"""
        for pattern in self.compiled_injection:
            if pattern.search(text):
                return True
        return False
    
    def log_security_event(self, event_type: str, details: str, level: str = "INFO"):
        """Log security-related events for monitoring"""
        message = f"SECURITY_{level.upper()}: {event_type} - {details}"
        if level.upper() == "WARNING":
            logger.warning(message)
        elif level.upper() == "ERROR":
            logger.error(message)
        else:
            logger.info(message)

# Create a single instance to use throughout the app
security = SecurityManager()
