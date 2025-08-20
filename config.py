import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Read the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY is missing.\n"
        "Please set it in your .env file (see .env.example)."
    )
  
