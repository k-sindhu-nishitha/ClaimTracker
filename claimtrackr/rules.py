"""Deterministic checks. Anything that must be exactly right (numbers, names, a hard exclusion
list) is plain code - fast, free, testable and explainable. The LLM handles the fuzzy parts."""
from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional, Tuple

# Same list as the original project.
EXCLUSION_LIST = [
    "HIV/AIDS",
    "Parkinson's disease",
    "Alzheimer's disease",
    "pregnancy",
    "substance abuse",
    "self-inflicted injuries",
    "sexually transmitted diseases(std)",
    "pre-existing conditions",
]

# Words that appear in many diagnoses. Original bug: "Heart DISEASE" scored 0.5 against
# "Parkinson's DISEASE" and was wrongly rejected, so generic words are ignored here.
_GENERIC = {"disease", "diseases", "condition", "conditions", "injury", "injuries", "treatment", "therapy",
            "with", "and", "the", "of", "for", "patient", "claim"}


def tokens(text: str) -> list:
    text = text.lower().replace("\u2019", "'")
    return [t for t in re.findall(r"[a-z]+(?:-[a-z]+)*", text) if len(t) > 1 and t not in _GENERIC]


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca)
    return dot / (math.sqrt(sum(v * v for v in ca.values())) * math.sqrt(sum(v * v for v in cb.values())))


def find_exclusion(text: str, threshold: float = 0.4) -> Optional[Tuple[str, float]]:
    """Return (matching exclusion, score) when text is similar enough to an excluded item."""
    best = max(((item, cosine(text, item)) for item in EXCLUSION_LIST), key=lambda x: x[1])
    return best if best[1] > threshold else None


def names_match(a: str, b: str, min_ratio: float = 0.85) -> bool:
    """Fuzzy name comparison: at least half of the shorter name's words must match a word in the other."""
    ta = re.findall(r"[a-z]+", (a or "").lower())
    tb = re.findall(r"[a-z]+", (b or "").lower())
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    hits = sum(any(SequenceMatcher(None, s, l).ratio() >= min_ratio for l in long_) for s in short)
    return hits / len(short) >= 0.5


def parse_amount(value) -> Optional[float]:
    """'Rs. 3,150.00' -> 3150.0 ; the original crashed on anything but a clean int."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))  # first number, ignoring 'Rs.', '$', commas
    return float(match.group(0).replace(",", "")) if match else None
