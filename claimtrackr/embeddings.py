"""Turn text into vectors. Every embedder returns L2-normalised float32 vectors, so the inner
product used by FAISS equals cosine similarity.

* OpenAIEmbedder  - real semantic embeddings (needs OPENAI_API_KEY)
* HashingEmbedder - offline fallback: hashed bag of words/bigrams. Lexical, not semantic, but
                    needs no network or model download, so the project demos anywhere.
"""
from __future__ import annotations

import re
import zlib
from typing import List

import numpy as np

from .config import Settings

_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "be", "by", "as", "at",
    "with", "that", "this", "it", "any", "all", "from", "what", "which", "give", "list",
}


class HashingEmbedder:
    """Hashed TF-IDF. fit() learns how rare each word is from the handbook, so rare, meaningful
    words ("exclusions", "Alzheimer") count more than common ones ("benefit", "claim")."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.idf = np.ones(dim, dtype="float32")

    @staticmethod
    def _grams(text: str) -> List[str]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def _counts(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for g in self._grams(text):
            vec[zlib.crc32(g.encode()) % self.dim] += 1.0
        return vec

    def fit(self, texts: List[str]) -> None:
        df = np.zeros(self.dim, dtype="float32")
        for t in texts:
            df += (self._counts(t) > 0)
        self.idf = (np.log((len(texts) + 1) / (df + 1)) + 1.0).astype("float32")

    def state(self) -> List[float]:  # saved next to the FAISS index so queries use the same weights
        return self.idf.tolist()

    def load_state(self, state: List[float]) -> None:
        self.idf = np.asarray(state, dtype="float32")

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.log1p(self._counts(t)) * self.idf  # sublinear tf * idf
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        return np.vstack(vecs).astype("float32")


class OpenAIEmbedder:
    def __init__(self, model: str):
        from openai import OpenAI  # imported lazily so offline mode needs no openai package

        self._client = OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL if you set one)
        self.model = model
        self.name = f"openai:{model}"
        self.dim = len(self.encode(["dimension probe"])[0])

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), 100):  # API accepts batches
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + 100])
            vectors.extend(d.embedding for d in resp.data)
        arr = np.asarray(vectors, dtype="float32")
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)


def get_embedder(settings: Settings):
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.embedding_model)
    if settings.embedding_provider == "offline":
        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
