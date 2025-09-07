from langchain_groq import ChatGroq
from typing import Optional, Any
import logging
from config import config

logger = logging.getLogger(__name__)

# Model fallback hierarchy - from fastest to most capable
MODEL_HIERARCHY = [
    "llama-3.1-8b-instant",      # Fastest, lowest token limit
    "llama-3.3-70b-versatile",   # More capable, higher limits
]

def create_llm_with_fallback(model_name: str = None, **kwargs) -> ChatGroq:
    """Create LLM with automatic fallback to other models"""
    if not model_name:
        model_name = MODEL_HIERARCHY[0]  

    # Set default parameters if not provided
    defaults = {
        "temperature": 0.3,
        "max_tokens": 1000,
        "groq_api_key": config.GROQ_API_KEY,
        "max_retries": 0,
        "timeout": 30.0
    }
    defaults.update(kwargs)

    try:
        llm = ChatGroq(model=model_name, **defaults)
        logger.info(f"Using model: {model_name}")
        return llm
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}")
        return get_fallback_model(**defaults)

def get_fallback_model(**kwargs) -> ChatGroq:
    """Get the best available fallback model"""
    for model in MODEL_HIERARCHY:
        try:
            llm = ChatGroq(model=model, **kwargs)
            logger.info(f"Fallback to model: {model}")
            return llm
        except Exception as e:
            logger.warning(f"Fallback model {model} failed: {e}")
            continue

    # If all models fail, raise error
    raise Exception("All models failed to load")

def get_model_for_task(task_type: str) -> str:
    """Choose appropriate model based on task"""
    if task_type == "summary":
        return MODEL_HIERARCHY[1] if len(MODEL_HIERARCHY) > 1 else MODEL_HIERARCHY[0]  # More capable model for comprehensive summaries
    elif task_type == "qa":
        return MODEL_HIERARCHY[1] if len(MODEL_HIERARCHY) > 1 else MODEL_HIERARCHY[0]  # More capable model for Q&A
    elif task_type == "complex":
        return MODEL_HIERARCHY[1] if len(MODEL_HIERARCHY) > 1 else MODEL_HIERARCHY[0]  # Most capable for complex tasks
    else:
        return MODEL_HIERARCHY[0]  # Default
