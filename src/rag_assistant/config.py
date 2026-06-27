"""Configuration defaults for the local RAG assistant."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROFILE_STORE_PATH = PROCESSED_DATA_DIR / "profiles.json"


def _default_vector_store_dir() -> Path:
    return PROJECT_ROOT / "vector_store"


VECTOR_STORE_DIR = Path(os.getenv("RAG_VECTOR_STORE_DIR", _default_vector_store_dir()))

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_PROFILE = "general"

#DEFAULT_LLM_MODEL = "qwen3-coder:30b"
#FAST_LLM_MODEL = "qwen2.5-coder:7b"
#DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

#DEFAULT_LLM_MODEL = "qwen3:30b"          # falls verfügbar

DEFAULT_LLM_MODEL = "qwen3:8b"          # falls verfügbar
FAST_LLM_MODEL = "qwen3:8b"             # oder gemma3:4b/12b
CODING_LLM_MODEL = "qwen3-coder:30b"

DEFAULT_EMBEDDING_MODEL = "bge-m3"      # gute Wahl für Deutsch/Englisch
FAST_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_TOP_K = 4
DEFAULT_EMBEDDING_BATCH_SIZE = 16
