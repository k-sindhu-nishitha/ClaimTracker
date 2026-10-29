from dataclasses import replace

import claimtrackr.processor as proc
from claimtrackr.config import Settings


def test_startup_falls_back_to_offline_when_the_api_is_unusable(tmp_path):
    settings = replace(Settings(llm_provider="openai", embedding_provider="openai"),
                       index_dir=tmp_path / "index", db_path=tmp_path / "claims.db")
    original = proc.get_embedder

    def fake_get_embedder(s):
        if s.embedding_provider == "openai":
            raise RuntimeError("429 You have no credits remaining")  # what OpenAI raised for you
        return original(s)

    proc.get_embedder = fake_get_embedder
    try:
        processor = proc.build_processor(settings)
    finally:
        proc.get_embedder = original
    assert processor.backend.name.startswith("offline")
    assert processor.embedder.name.startswith("hashing")
