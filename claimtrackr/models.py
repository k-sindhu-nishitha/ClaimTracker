"""Plain dataclasses that flow through the pipeline (easy to read, easy to JSON-serialise)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

APPROVED, REJECTED, NEEDS_REVIEW = "APPROVED", "REJECTED", "NEEDS_REVIEW"


@dataclass
class ClaimInput:  # what the user types into the web form
    name: str
    address: str = ""
    claim_type: str = ""
    claim_reason: str = ""
    date: str = ""
    medical_facility: str = ""
    total_claim_amount: float = 0.0
    description: str = ""


@dataclass
class BillInfo:  # what the LLM extracts from the uploaded bill
    patient_name: Optional[str] = None
    disease: Optional[str] = None
    expense: Optional[float] = None


@dataclass
class Source:  # one retrieved handbook excerpt
    source: str
    page: int
    score: float
    query: str
    text: str


@dataclass
class Assessment:  # what the "reasoning" LLM returns
    verdict: str
    information_complete: bool
    exclusion_triggered: bool
    report: str
    fraud_signals: List[str] = field(default_factory=list)


@dataclass
class Step:  # one line of the pipeline trace shown in the UI
    name: str
    status: str  # pass | fail | warn | info
    detail: str


@dataclass
class ClaimResult:
    verdict: str
    approved_amount: Optional[float]
    summary: str
    steps: List[Step]
    bill: BillInfo
    sources: List[Source]
    report: str
    backend: str
    latency_ms: float
