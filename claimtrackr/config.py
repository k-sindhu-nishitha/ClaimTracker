"""All settings in one place. Override with environment variables or a .env file.

Default behaviour is "just works":
  * OPENAI_API_KEY set     -> real OpenAI embeddings + chat model
  * no key                 -> offline demo mode (hashing embeddings + rule-based mock LLM)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent


def _auto(env_name: str) -> str:
    explicit = os.getenv(env_name)
    if explicit:
        return explicit.lower()
    return "openai" if os.getenv("OPENAI_API_KEY") else "offline"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _auto("LLM_PROVIDER"))  # openai | offline
    embedding_provider: str = field(default_factory=lambda: _auto("EMBEDDING_PROVIDER"))  # openai | offline
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    docs_dir: Path = ROOT / "documents"  # the knowledge base: BUPA handbook PDFs
    index_dir: Path = ROOT / "faiss_index"  # persisted FAISS index
    db_path: Path = ROOT / "storage" / "claims.db"  # claim history (SQLite)

    chunk_size: int = 1000  # same values the original project used
    chunk_overlap: int = 200
    top_k: int = 4  # chunks retrieved per query
    exclusion_threshold: float = 0.4  # cosine similarity needed to flag an exclusion


def get_settings() -> Settings:
    return Settings()
