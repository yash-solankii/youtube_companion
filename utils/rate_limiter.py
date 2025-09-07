import time
import threading
from collections import deque
from typing import Callable, Any
import logging
from queue import Queue

logger = logging.getLogger(__name__)

class SmartRateLimiter:
    """Smart rate limiter that uses full capacity while avoiding limits"""

    def __init__(self, max_requests_per_minute: int = 30, max_tokens_per_minute: int = 6000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # Request and token tracking
        self.request_times = deque()
        self.token_usage = deque()  # [(timestamp, tokens_used), ...]

        self._lock = threading.Lock()
        self.backoff_until = 0

        # Adaptive delays based on usage
        self.min_delay = 0.5  # Minimum delay between requests
        self.max_delay = 3.0  # Maximum delay when approaching limits

        # Token estimation for Llama models
        self.token_estimates = {
            'llama': {'base': 80, 'per_char': 0.3},
            'default': {'base': 50, 'per_char': 0.25}
        }

    def estimate_tokens(self, text: str, model_name: str = 'llama') -> int:
        """Estimate token usage for a request"""
        if not text:
            return 50  # Base overhead

        estimator = self.token_estimates.get('llama' if 'llama' in model_name.lower() else 'default')
        estimated = estimator['base'] + int(len(text) * estimator['per_char'])
        return min(estimated, 4000)  # Cap at reasonable limit

    def _get_current_usage(self) -> tuple[int, int]:
        """Get current requests and tokens in last minute"""
        now = time.time()

        # Clean old data
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        while self.token_usage and now - self.token_usage[0][0] > 60:
            self.token_usage.popleft()

        current_requests = len(self.request_times)
        current_tokens = sum(tokens for _, tokens in self.token_usage)

        return current_requests, current_tokens

    def _calculate_smart_delay(self, estimated_tokens: int) -> float:
        """Calculate optimal delay to maximize usage without hitting limits"""
        current_requests, current_tokens = self._get_current_usage()

        # If we're at or near limits, use maximum delay
        if current_requests >= self.max_requests_per_minute * 0.9:
            return self.max_delay

        if current_tokens + estimated_tokens >= self.max_tokens_per_minute * 0.9:
            return self.max_delay

        # If we're well below limits, use minimum delay
        if current_requests < self.max_requests_per_minute * 0.5 and current_tokens < self.max_tokens_per_minute * 0.5:
            return self.min_delay

        # Adaptive delay based on usage percentage
        request_ratio = current_requests / self.max_requests_per_minute
        token_ratio = current_tokens / self.max_tokens_per_minute

        # Use the higher ratio for more conservative delay
        usage_ratio = max(request_ratio, token_ratio)

        # Scale delay between min and max based on usage
        delay = self.min_delay + (self.max_delay - self.min_delay) * usage_ratio

        return min(delay, self.max_delay)

    def _wait_smart_delay(self, estimated_tokens: int) -> None:
        """Smart waiting that maximizes throughput"""
        now = time.time()

        # Check backoff from previous 429
        if now < self.backoff_until:
            wait_time = self.backoff_until - now
            logger.warning(f"Backoff active, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            return

        # Check if we need to wait for rate limits
        current_requests, current_tokens = self._get_current_usage()

        if current_requests >= self.max_requests_per_minute:
            oldest_time = self.request_times[0]
            wait_time = 60 - (now - oldest_time)
            if wait_time > 0:
                logger.warning(f"At request limit, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                return

        if current_tokens + estimated_tokens >= self.max_tokens_per_minute:
            # Calculate wait time for token reset
            if self.token_usage:
                oldest_time = self.token_usage[0][0]
                wait_time = 60 - (now - oldest_time)
                if wait_time > 0:
                    logger.warning(f"Token limit approaching, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    return

        # Smart delay based on current usage
        if self.request_times:
            time_since_last = now - self.request_times[-1]
            required_delay = self._calculate_smart_delay(estimated_tokens)

            if time_since_last < required_delay:
                sleep_time = required_delay - time_since_last
                time.sleep(sleep_time)

    def execute_with_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with smart rate limiting that maximizes throughput"""
        # Extract text content for token estimation
        text_content = self._extract_content(args, kwargs)
        estimated_tokens = self.estimate_tokens(text_content)

        with self._lock:
            self._wait_smart_delay(estimated_tokens)
            now = time.time()
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate_limit" in error_msg or "too many requests" in error_msg:
                logger.warning("Rate limit hit, backing off for 15s")
                self.backoff_until = time.time() + 15
                time.sleep(15)
                # Retry with same function
                return self.execute_with_rate_limit(func, *args, **kwargs)
            else:
                raise e

    def _extract_content(self, args: tuple, kwargs: dict) -> str:
        """Extract text content from function arguments"""
        # Common parameter names that contain text
        text_params = ['text', 'content', 'message', 'prompt', 'input', 'query', 'question']

        # Check kwargs first
        for param in text_params:
            if param in kwargs and isinstance(kwargs[param], str):
                return kwargs[param][:2000]  # Limit for estimation

        # Check args
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10:
                return arg[:2000]

        # Default estimate
        return "medium_request"

    def get_stats(self) -> dict:
        """Get comprehensive usage stats"""
        now = time.time()
        current_requests, current_tokens = self._get_current_usage()

        return {
            "current_requests": current_requests,
            "current_tokens": current_tokens,
            "max_requests": self.max_requests_per_minute,
            "max_tokens": self.max_tokens_per_minute,
            "request_utilization": (current_requests / self.max_requests_per_minute) * 100,
            "token_utilization": (current_tokens / self.max_tokens_per_minute) * 100,
            "backoff_active": self.backoff_until > now,
            "backoff_remaining": max(0, int(self.backoff_until - now)) if self.backoff_until > now else 0
        }

# Global rate limiter instance - will be configured with proper values
rate_limiter = None

def get_rate_limiter():
    """Get or create the smart rate limiter"""
    global rate_limiter
    if rate_limiter is None:
        try:
            from config import config
            rate_limiter = SmartRateLimiter(
                max_requests_per_minute=config.RATE_LIMIT_REQUESTS,  # 30 RPM
                max_tokens_per_minute=6000  # Groq token limit
            )
            logger.info(f"Smart rate limiter: {config.RATE_LIMIT_REQUESTS} RPM, 6000 TPM")
        except ImportError:
            rate_limiter = SmartRateLimiter()
            logger.warning("Using default smart rate limiter")
    return rate_limiter
