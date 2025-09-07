import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

class AppLogger:
    """Professional logging system with detailed console output"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Only set up handlers once
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up console and file logging with proper formatting"""
        # Console output with detailed formatting
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        
        # File logging for errors and debugging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'app.log')
        file_handler.setLevel(logging.INFO)
        
        # Error file for critical issues
        error_handler = logging.FileHandler(log_dir / 'errors.log')
        error_handler.setLevel(logging.ERROR)
        
        # Format messages professionally
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        
        console.setFormatter(console_format)
        file_handler.setFormatter(file_format)
        error_handler.setFormatter(file_format)
        
        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional context"""
        if extra:
            context = f" | {json.dumps(extra, default=str)}"
            self.logger.info(f"{message}{context}")
        else:
            self.logger.info(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        if extra:
            context = f" | {json.dumps(extra, default=str)}"
            self.logger.warning(f"{message}{context}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        if extra:
            context = f" | {json.dumps(extra, default=str)}"
            self.logger.error(f"{message}{context}")
        else:
            self.logger.error(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        if extra:
            context = f" | {json.dumps(extra, default=str)}"
            self.logger.debug(f"{message}{context}")
        else:
            self.logger.debug(message)

class Timer:
    """Performance tracking with detailed metrics"""
    
    def __init__(self):
        self.start_times = {}
        self.performance_metrics = {}
    
    def start(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end(self, operation: str, extra: Optional[Dict[str, Any]] = None):
        """End timing and log the duration with metrics"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            
            # Store performance metrics
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = []
            self.performance_metrics[operation].append(duration)
            
            # Calculate statistics
            metrics = self.performance_metrics[operation]
            avg_duration = sum(metrics) / len(metrics)
            min_duration = min(metrics)
            max_duration = max(metrics)
            
            context = {
                "duration": f"{duration:.3f}s",
                "avg_duration": f"{avg_duration:.3f}s",
                "min_duration": f"{min_duration:.3f}s",
                "max_duration": f"{max_duration:.3f}s",
                "total_calls": len(metrics)
            }
            
            if extra:
                context.update(extra)
            
            logger = get_logger("timer")
            logger.info(f"Completed: {operation}", context)
            del self.start_times[operation]
        else:
            logger = get_logger("timer")
            logger.warning(f"No timer found for: {operation}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        summary = {}
        for operation, metrics in self.performance_metrics.items():
            if metrics:
                summary[operation] = {
                    "total_calls": len(metrics),
                    "avg_duration": sum(metrics) / len(metrics),
                    "min_duration": min(metrics),
                    "max_duration": max(metrics),
                    "total_time": sum(metrics)
                }
        return summary

# Create logger instances
app_logger = AppLogger("youtube_companion")
timer = Timer()

def get_logger(name: str) -> AppLogger:
    """Get a logger for a specific module"""
    return AppLogger(name)
