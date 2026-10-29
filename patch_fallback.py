import pathlib

# 1) processor.py: build_processor() with graceful fallback
p = pathlib.Path("claimtrackr/processor.py"); s = p.read_text()
if "def build_processor" not in s:
    s = s.replace("import logging\nimport time\n", "import logging\nimport time\nfrom dataclasses import replace\n")
    s += '''

def build_processor(settings: Settings) -> ClaimProcessor:
    """Create the processor; if the real LLM/embedding service is unusable at startup (no credits = HTTP 429,
    bad key = 401, no network) fall back to offline demo mode instead of crashing."""
    try:
        return ClaimProcessor(settings)
    except Exception as exc:
        if settings.llm_provider == "offline" and settings.embedding_provider == "offline":
            raise  # nothing left to fall back to
        logger.warning("Could not start with llm=%s embeddings=%s (%s: %s). Falling back to OFFLINE demo mode. "
                       "Fix your API key/billing and restart to use the real LLM.",
                       settings.llm_provider, settings.embedding_provider, type(exc).__name__, str(exc)[:160])
        return ClaimProcessor(replace(settings, llm_provider="offline", embedding_provider="offline"))
'''
    p.write_text(s)

# 2) app.py: use it
a = pathlib.Path("app.py"); t = a.read_text()
t = t.replace("from claimtrackr.processor import ClaimProcessor", "from claimtrackr.processor import build_processor")
t = t.replace("processor = ClaimProcessor(settings)  # loads (or builds) the FAISS index once at startup",
              "processor = build_processor(settings)  # loads (or builds) the FAISS index once at startup")
a.write_text(t)

# 3) test
pathlib.Path("tests/test_fallback.py").write_text('''from dataclasses import replace

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
''')
print("patched")
