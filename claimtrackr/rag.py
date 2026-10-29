"""RAG knowledge base: handbook PDFs -> chunks -> embeddings -> FAISS index (saved to disk).

Improvement over the original project: the original rebuilt the whole index (and paid for
embeddings) on EVERY request. Here it is built once, saved to faiss_index/, and reloaded.
It only rebuilds when the PDFs or the embedding model change.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import split_text
from .config import Settings
from .pdf_utils import extract_pages

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source: str  # PDF file name
    page: int  # 1-based page number, used for citations


class VectorIndex:
    """Thin wrapper around FAISS. IndexFlatIP = exact inner-product search; on normalised
    vectors this is cosine similarity. A few hundred chunks need no approximate index."""

    def __init__(self, dim: int, embedder_name: str, fingerprint: str = "", embedder_state=None):
        self.dim, self.embedder_name, self.fingerprint = dim, embedder_name, fingerprint
        self.embedder_state = embedder_state  # e.g. IDF weights of the offline embedder
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        k = min(k, len(self.chunks))
        if k == 0:
            return []
        scores, ids = self.index.search(np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = {"embedder": self.embedder_name, "dim": self.dim, "fingerprint": self.fingerprint, "embedder_state": self.embedder_state,
                "chunks": [asdict(c) for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        idx = cls(meta["dim"], meta["embedder"], meta["fingerprint"], meta.get("embedder_state"))
        idx.index = faiss.read_index(str(directory / "index.faiss"))
        idx.chunks = [Chunk(**c) for c in meta["chunks"]]
        return idx


def docs_fingerprint(docs_dir: Path) -> str:
    """Cheap change detector: file names + sizes."""
    parts = [f"{p.name}:{p.stat().st_size}" for p in sorted(docs_dir.glob("*.pdf"))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def load_chunks(settings: Settings) -> List[Chunk]:
    chunks: List[Chunk] = []
    pdfs = sorted(settings.docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF documents found in {settings.docs_dir}")
    for pdf in pdfs:
        for page_no, page_text in enumerate(extract_pages(str(pdf)), start=1):
            for piece in split_text(page_text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(piece, pdf.name, page_no))
    return chunks


def build_index(settings: Settings, embedder) -> VectorIndex:
    chunks = load_chunks(settings)
    texts = [c.text for c in chunks]
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    index = VectorIndex(embedder.dim, embedder.name, docs_fingerprint(settings.docs_dir),
                        embedder.state() if hasattr(embedder, "state") else None)
    index.add(embedder.encode(texts), chunks)
    index.save(settings.index_dir)
    logger.info("Built index: %d chunks from %s using %s", len(chunks), settings.docs_dir, embedder.name)
    return index


def load_or_build_index(settings: Settings, embedder) -> VectorIndex:
    if (settings.index_dir / "index.faiss").exists():
        index = VectorIndex.load(settings.index_dir)
        if index.embedder_name == embedder.name and index.fingerprint == docs_fingerprint(settings.docs_dir):
            if index.embedder_state is not None and hasattr(embedder, "load_state"):
                embedder.load_state(index.embedder_state)
            return index
        logger.info("Documents or embedding model changed -> rebuilding index")
    return build_index(settings, embedder)
