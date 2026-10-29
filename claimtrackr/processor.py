"""The claim workflow. Read top to bottom: this is the heart of the project.

    bill PDF text
      -> 1. extract disease / expense / patient name   (LLM)
      -> 2. amount check                               (rule)
      -> 3. name check                                 (rule)
      -> 4. exclusion screen                           (rule: cosine similarity vs exclusion list)
      -> 5. retrieve handbook excerpts                 (RAG: FAISS)
      -> 6. assess + write report                      (LLM, grounded in the excerpts)
      -> 7. final decision with guardrails

Guardrail principle: the LLM may make a decision STRICTER (reject / send to a human) but can never
overrule a failed rule, and anything unclear goes to NEEDS_REVIEW instead of a guess.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Dict, List, Optional

from .config import Settings
from .embeddings import get_embedder
from .llm import get_backend, template_report
from .models import APPROVED, NEEDS_REVIEW, REJECTED, BillInfo, ClaimInput, ClaimResult, Source, Step
from .rag import VectorIndex, load_or_build_index
from .rules import find_exclusion, names_match

logger = logging.getLogger(__name__)

# Query rewriting: the queries are phrased the way the handbook itself is written ("The Scheme shall not
# cover...", "Required claim documents: ..."). Tested against the original wording ("Give a list of all
# general exclusions"), this moved the real exclusions page from rank 4-5 to rank 1-2 with the offline embedder.
STATIC_QUERIES: Dict[str, str] = {
    "required documents": "Required claim documents: original receipt must include consultation date, patient's full name, diagnosis",
    "general exclusions": "General exclusions: the Scheme shall not cover any treatment or expenses relating to",
}


class ClaimProcessor:
    def __init__(self, settings: Settings, index: Optional[VectorIndex] = None, backend=None, embedder=None):
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.index = index or load_or_build_index(settings, self.embedder)
        self.backend = backend or get_backend(settings)

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, bill: BillInfo, claim: ClaimInput) -> List[Source]:
        """Static queries (documents required, general exclusions) + one dynamic query about THIS disease."""
        queries = dict(STATIC_QUERIES)
        if bill.disease:
            queries["this diagnosis"] = f"{bill.disease} general exclusions not covered"  # dynamic: depends on the claim
        best: Dict[tuple, Source] = {}
        vectors = self.embedder.encode(list(queries.values()))
        for (label, text), vec in zip(queries.items(), vectors):
            for chunk, score in self.index.search(vec, self.settings.top_k):
                key = (chunk.source, chunk.page, chunk.text[:60])
                if key not in best or score > best[key].score:
                    best[key] = Source(chunk.source, chunk.page, round(score, 4), label, chunk.text)
        return sorted(best.values(), key=lambda s: s.score, reverse=True)

    # ---------------------------------------------------------------- main workflow
    def process(self, claim: ClaimInput, bill_text: str) -> ClaimResult:
        start = time.perf_counter()
        steps: List[Step] = []
        bill, sources = BillInfo(), []

        def finish(verdict: str, summary: str, report: str = "") -> ClaimResult:
            amount = min(claim.total_claim_amount, bill.expense) if verdict == APPROVED and bill.expense else None
            steps.append(Step("Final decision", "pass" if verdict == APPROVED else "fail" if verdict == REJECTED else "warn", summary))
            return ClaimResult(verdict, amount, summary, steps, bill, sources,
                               report or template_report(claim, bill, [summary], sources, verdict),
                               self.backend.name, round((time.perf_counter() - start) * 1000, 1))

        # 0. Did we get any text? (scanned/image PDFs have no text layer; OCR is not enabled)
        if not bill_text.strip():
            steps.append(Step("Read bill PDF", "fail", "No text found - the PDF may be a scanned image."))
            return finish(NEEDS_REVIEW, "Could not read the bill. Please upload a text-based consultation receipt.")
        steps.append(Step("Read bill PDF", "pass", f"Extracted {len(bill_text)} characters of text."))

        # 1. Extract structured fields from the bill
        try:
            bill = self.backend.extract_bill(bill_text)
        except Exception as exc:  # network error, bad JSON ...
            logger.warning("Bill extraction failed: %s", exc)
            steps.append(Step("Extract bill details", "fail", f"Extraction failed ({type(exc).__name__})."))
            return finish(NEEDS_REVIEW, "Bill details could not be extracted automatically; sent to a human reviewer.")
        if not bill.disease or bill.expense is None:
            steps.append(Step("Extract bill details", "fail", "Diagnosis or amount missing from the bill."))
            return finish(NEEDS_REVIEW, "Please enter a valid consultation receipt (diagnosis and amount not found).")
        steps.append(Step("Extract bill details", "pass",
                          f"Patient: {bill.patient_name or '?'} | Diagnosis: {bill.disease} | Billed: {bill.expense:,.0f}"))

        failures: List[str] = []

        # 2. Amount check (original bug: claimed == billed fell into the 'invalid receipt' branch)
        if claim.total_claim_amount > bill.expense:
            msg = f"Claimed amount {claim.total_claim_amount:,.0f} is more than the billed amount {bill.expense:,.0f}."
            failures.append(msg)
            steps.append(Step("Amount check", "fail", msg))
        else:
            steps.append(Step("Amount check", "pass", f"Claimed {claim.total_claim_amount:,.0f} <= billed {bill.expense:,.0f}."))

        # 3. Name check
        if bill.patient_name and not names_match(claim.name, bill.patient_name):
            msg = f"Claimant '{claim.name}' does not match the patient name on the bill '{bill.patient_name}'."
            failures.append(msg)
            steps.append(Step("Name check", "fail", msg))
        else:
            steps.append(Step("Name check", "pass" if bill.patient_name else "warn",
                              "Names match." if bill.patient_name else "No patient name on the bill."))

        # 4. Exclusion screen on the bill diagnosis and the stated claim reason
        hit = find_exclusion(bill.disease, self.settings.exclusion_threshold) or \
            (find_exclusion(claim.claim_reason, self.settings.exclusion_threshold) if claim.claim_reason else None)
        if hit:
            msg = f"'{bill.disease}' matches general exclusion '{hit[0]}' (similarity {hit[1]:.2f})."
            failures.append(msg)
            steps.append(Step("Exclusion screen", "fail", msg))
        else:
            steps.append(Step("Exclusion screen", "pass", "Diagnosis does not match the exclusion list."))

        # 5. RAG retrieval
        try:
            sources = self.retrieve(bill, claim)
            steps.append(Step("Retrieve handbook context", "pass",
                              f"{len(sources)} excerpts, e.g. " + ", ".join(f"{s.source} p.{s.page}" for s in sources[:3])))
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            steps.append(Step("Retrieve handbook context", "fail", f"Retrieval failed ({type(exc).__name__})."))

        # 6. LLM assessment + report
        try:
            assessment = self.backend.assess(claim, bill, failures, sources)
            steps.append(Step("LLM assessment", "info",
                              f"verdict={assessment.verdict}, information_complete={assessment.information_complete}, "
                              f"exclusion_triggered={assessment.exclusion_triggered}"))
        except Exception as exc:
            logger.warning("Assessment failed: %s", exc)
            steps.append(Step("LLM assessment", "fail", f"Assessment unavailable ({type(exc).__name__})."))
            assessment = None

        # 7. Final decision with guardrails
        if failures:  # rules always win; the report explains why
            report = assessment.report if assessment else template_report(claim, bill, failures, sources, REJECTED)
            return finish(REJECTED, " ".join(failures), report)
        if assessment is None:
            return finish(NEEDS_REVIEW, "Automated assessment unavailable; sent to a human reviewer.")
        if assessment.verdict == REJECTED and assessment.exclusion_triggered:
            return finish(REJECTED, "The assessment found an exclusion in the handbook that applies to this claim.", assessment.report)
        if assessment.verdict == REJECTED:  # a rejection with no stated exclusion is not trustworthy
            return finish(NEEDS_REVIEW, "The model suggested rejection without citing an exclusion; sent to a human reviewer.", assessment.report)
        if assessment.verdict == NEEDS_REVIEW or not assessment.information_complete:
            return finish(NEEDS_REVIEW, "Required information looks incomplete; sent to a human reviewer.", assessment.report)
        return finish(APPROVED, "All checks passed.", assessment.report)


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
