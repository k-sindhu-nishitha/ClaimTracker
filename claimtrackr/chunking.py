"""Recursive character text splitter (same idea as LangChain's RecursiveCharacterTextSplitter).

Try to split on paragraph breaks first, then lines, then spaces, so chunks end at natural
boundaries. Neighbouring chunks overlap so a sentence cut at a boundary still appears whole
in at least one chunk.
"""
from __future__ import annotations

from typing import List, Sequence

SEPARATORS = ("\n\n", "\n", " ")


def _merge(pieces: List[str], sep: str, size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    for piece in pieces:
        if current and len(sep.join(current + [piece])) > size:
            chunks.append(sep.join(current))
            # drop pieces from the front until what is left is <= overlap and the next piece fits
            while current and (len(sep.join(current)) > overlap or len(sep.join(current + [piece])) > size):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(sep.join(current))
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, size: int = 1000, overlap: int = 200, separators: Sequence[str] = SEPARATORS) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not separators:  # last resort: hard cut
        step = max(1, size - overlap)
        return [text[i : i + size] for i in range(0, len(text), step)]

    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return split_text(text, size, overlap, ())
    pieces = [p for p in text.split(sep) if p.strip()]

    out: List[str] = []
    small: List[str] = []
    for piece in pieces:
        if len(piece) <= size:
            small.append(piece)
        else:  # piece too long -> flush what we have, then split the piece with finer separators
            out.extend(_merge(small, sep, size, overlap))
            small = []
            out.extend(split_text(piece, size, overlap, separators[separators.index(sep) + 1 :]))
    out.extend(_merge(small, sep, size, overlap))
    return out
