"""
config.py — Application configuration and environment settings for Verity.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Root directory of the backend
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Relational Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'verity_dev.db'}")

# Vector Database (ChromaDB)
# Default to backend/data/vector_db (persisted on disk and covered by .gitignore)
DEFAULT_VECTOR_DB_DIR = str(BACKEND_DIR / "data" / "vector_db")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", DEFAULT_VECTOR_DB_DIR)

# ChromaDB Collection
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "verity_knowledge_base")

# Embedding Model
# Small, fast, high-quality sentence embedding model producing 384-dimensional vectors
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)

# Chunking Strategy
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))

# LLM-as-a-Judge Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # "auto", "openai", "gemini", "local"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
