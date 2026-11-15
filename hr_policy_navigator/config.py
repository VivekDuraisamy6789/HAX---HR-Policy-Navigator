import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DATA_ROOT", "sourcegraph_files")).resolve() if os.getenv("DATA_ROOT") else ROOT / "sourcegraph_files"

APP_TITLE = os.getenv("APP_TITLE", "HR Policy Navigator")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen-8b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", OPENAI_MODEL)
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")


def has_model_config() -> bool:
    return bool(OPENAI_BASE_URL and OPENAI_API_KEY)
