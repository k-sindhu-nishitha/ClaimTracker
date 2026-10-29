"""LLM backends. The processor only talks to two methods:

    extract_bill(bill_text)                       -> BillInfo
    assess(claim, bill, rule_failures, sources)   -> Assessment

OpenAIBackend  : real LLM calls (works with any OpenAI-compatible endpoint via OPENAI_BASE_URL,
                 e.g. OpenRouter or a local Ollama server).
OfflineBackend : deterministic regex/template stand-in so the whole app runs without an API key.
                 It is NOT an AI model; it exists for demos and tests.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from . import prompts
from .config import Settings
from .models import APPROVED, NEEDS_REVIEW, REJECTED, Assessment, BillInfo, ClaimInput, Source
from .rules import parse_amount


def template_report(claim: ClaimInput, bill: BillInfo, failures: List[str], sources: List[Source], verdict: str) -> str:
    cites = ", ".join(sorted({f"{s.source} p.{s.page}" for s in sources[:3]})) or "none retrieved"
    reasons = "\n".join(f"- {f}" for f in failures) if failures else "No rule violations found."
    return (
        f"Executive Summary\nClaim for {claim.name} ({bill.disease or 'unknown condition'}) was {verdict}.\n\n"
        f"Introduction\nThis report records how the claim was checked against the membership handbook.\n\n"
        f"Claim Details\nClaimed amount {claim.total_claim_amount:,.0f}; billed amount "
        f"{(bill.expense if bill.expense is not None else 0):,.0f}; facility: {claim.medical_facility or 'n/a'}.\n\n"
        f"Claim Description\n{claim.description or claim.claim_reason or 'No description supplied.'}\n\n"
        f"Document Verification\nBill diagnosis: {bill.disease}. Handbook excerpts consulted: {cites}.\n\n"
        f"Document Summary\nAutomated checks:\n{reasons}"
    )


class OfflineBackend:
    name = "offline-demo (no LLM)"

    def extract_bill(self, text: str) -> BillInfo:
        def grab(pattern: str) -> Optional[str]:
            m = re.search(pattern, text, re.I | re.M)
            return m.group(1).strip() if m else None

        payable = grab(r"amount\s*payable\s*[-:]?\s*(?:rs\.?|inr|\$|\u20b9)?\s*([\d,]+(?:\.\d+)?)")
        total = grab(r"total\s*(?:charge|amount|bill)s?\s*[-:]?\s*(?:rs\.?|inr|\$|\u20b9)?\s*([\d,]+(?:\.\d+)?)")
        return BillInfo(
            patient_name=grab(r"\bname\s*:\s*(.+?)\s*$"),
            disease=grab(r"\b(?:diagnosis|disease|illness)\s*:\s*(.+?)\s*$"),
            expense=parse_amount(payable or total),
        )

    def assess(self, claim, bill, failures, sources) -> Assessment:
        complete = all([bill.patient_name, bill.disease, bill.expense])
        verdict = REJECTED if failures else (APPROVED if complete else NEEDS_REVIEW)
        report = "[Offline demo mode: report generated from a template, not by an LLM]\n\n"
        report += template_report(claim, bill, failures, sources, verdict)
        return Assessment(verdict, complete, False, report)


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)  # models sometimes wrap JSON in prose or code fences
    if not match:
        raise ValueError("no JSON object in LLM output")
    return json.loads(match.group(0))


class OpenAIBackend:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = settings.chat_model
        self.name = f"openai:{settings.chat_model}"

    def _chat(self, system: str, user: str) -> str:
        kwargs = dict(model=self._model, temperature=0,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            resp = self._client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except Exception:  # some OpenAI-compatible servers do not support JSON mode
            resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def extract_bill(self, text: str) -> BillInfo:
        data = _parse_json(self._chat(prompts.EXTRACT_SYSTEM, f"<bill>\n{text}\n</bill>"))
        return BillInfo(data.get("patient_name"), data.get("disease"), parse_amount(data.get("expense")))

    def assess(self, claim, bill, failures, sources) -> Assessment:
        excerpts = "\n".join(f'<excerpt file="{s.source}" page="{s.page}">\n{s.text}\n</excerpt>' for s in sources)
        user = (
            f"<claim>\n{json.dumps(claim.__dict__, indent=2)}\n</claim>\n"
            f"<bill>\n{json.dumps(bill.__dict__)}\n</bill>\n"
            f"<rule_failures>\n{json.dumps(failures)}\n</rule_failures>\n"
            f"<handbook>\n{excerpts}\n</handbook>"
        )
        data = _parse_json(self._chat(prompts.ASSESS_SYSTEM, user))
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in (APPROVED, REJECTED, NEEDS_REVIEW):
            raise ValueError(f"invalid verdict from LLM: {verdict!r}")
        return Assessment(verdict, bool(data.get("information_complete", False)),
                          bool(data.get("exclusion_triggered", False)), str(data.get("report", "")),
                          list(data.get("fraud_signals") or []))


def get_backend(settings: Settings):
    if settings.llm_provider == "openai":
        return OpenAIBackend(settings)
    if settings.llm_provider == "offline":
        return OfflineBackend()
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
