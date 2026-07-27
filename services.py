import os
import json
import logging
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient
from groq import Groq

# Load environment variables (ensure .env is already loaded in main)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "learnnova")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize MongoDB client and expose the database
client = AsyncIOMotorClient(MONGO_URI)
# The database object used throughout the app
db = client[DB_NAME]

# Initialize Groq client for LLM calls
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY not set – LLM calls will fail")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)


def extract_json(text: str):
    """Safely extract a JSON object/array from a string.

    The LLM sometimes returns surrounding text before/after the JSON. This helper
    attempts to locate the first opening brace/bracket and the matching closing
    counterpart, then parses the substring with ``json.loads``.
    If parsing fails, the raw string is returned so the caller can handle the
    error gracefully.
    """
    # Find the first JSON start character
    start_idx = None
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start_idx = i
            break
    if start_idx is None:
        return text  # no JSON found
    # Use a simple stack to find the matching closing character
    stack = []
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if not stack:
                break
            opening = stack.pop()
            if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                break
            if not stack:
                try:
                    return json.loads(text[start_idx : i + 1])
                except json.JSONDecodeError:
                    return text
    return text
