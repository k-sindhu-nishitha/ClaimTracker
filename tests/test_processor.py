import pytest

from claimtrackr.models import APPROVED, NEEDS_REVIEW, REJECTED, Assessment


def test_offline_bill_extraction(processor, bill_text):
    bill = processor.backend.extract_bill(bill_text("Bodyache"))
    assert bill.patient_name == "Surbhit"
    assert "Bodyache" in bill.disease
    assert bill.expense == 3150  # 'amount payable', not the 3500 total


def test_rag_retrieves_the_exclusions_page(processor, bill_text, claim):
    bill = processor.backend.extract_bill(bill_text("Alzheimer"))
    sources = processor.retrieve(bill, claim())
    assert any("shall not cover" in s.text for s in sources)
    assert any("Alzheimer" in s.text for s in sources)


@pytest.mark.parametrize("bill,amount,name,expected", [
    ("Bodyache", 3000, "Surbhit", APPROVED),
    ("Bodyache", 3150, "Surbhit", APPROVED),   # claimed == billed (original code mishandled this)
    ("Bodyache", 9000, "Surbhit", REJECTED),   # more than billed
    ("Bodyache", 3000, "Rahul Sharma", REJECTED),  # name mismatch
    ("HIV", 3000, "Surbhit", REJECTED),        # general exclusion
    ("Alzheimer", 3000, "Surbhit", REJECTED),  # general exclusion
])
def test_end_to_end_verdicts(processor, bill_text, claim, bill, amount, name, expected):
    result = processor.process(claim(name=name, total_claim_amount=amount), bill_text(bill))
    assert result.verdict == expected, [(s.name, s.status, s.detail) for s in result.steps]
    assert result.steps and result.report


def test_approved_amount_never_exceeds_billed(processor, bill_text, claim):
    result = processor.process(claim(total_claim_amount=3000), bill_text("Bodyache"))
    assert result.approved_amount == 3000


def test_empty_or_scanned_pdf_goes_to_manual_review(processor, claim):
    assert processor.process(claim(), "").verdict == NEEDS_REVIEW
    assert processor.process(claim(), "random text with no invoice fields").verdict == NEEDS_REVIEW


# ------------------------------------------------------------------ guardrails with a scripted LLM
class ScriptedBackend:
    name = "scripted"

    def __init__(self, assessment=None, error=None):
        self.assessment, self.error = assessment, error

    def extract_bill(self, text):
        from claimtrackr.llm import OfflineBackend
        return OfflineBackend().extract_bill(text)

    def assess(self, claim, bill, failures, sources):
        if self.error:
            raise self.error
        return self.assessment


@pytest.fixture()
def with_backend(processor):
    original = processor.backend
    yield lambda backend: setattr(processor, "backend", backend)
    processor.backend = original


def test_llm_cannot_overrule_failed_rule(processor, with_backend, bill_text, claim):
    with_backend(ScriptedBackend(Assessment(APPROVED, True, False, "approve!")))
    result = processor.process(claim(total_claim_amount=9000), bill_text("Bodyache"))
    assert result.verdict == REJECTED


def test_llm_can_catch_exclusion_the_keyword_screen_missed(processor, with_backend, bill_text, claim):
    with_backend(ScriptedBackend(Assessment(REJECTED, True, True, "dementia is excluded (item 14)")))
    result = processor.process(claim(), bill_text("Bodyache"))
    assert result.verdict == REJECTED


def test_unexplained_llm_rejection_goes_to_human(processor, with_backend, bill_text, claim):
    with_backend(ScriptedBackend(Assessment(REJECTED, True, False, "just because")))
    assert processor.process(claim(), bill_text("Bodyache")).verdict == NEEDS_REVIEW


def test_incomplete_information_goes_to_human(processor, with_backend, bill_text, claim):
    with_backend(ScriptedBackend(Assessment(APPROVED, False, False, "missing stamp")))
    assert processor.process(claim(), bill_text("Bodyache")).verdict == NEEDS_REVIEW


def test_llm_outage_fails_safe(processor, with_backend, bill_text, claim):
    with_backend(ScriptedBackend(error=TimeoutError("api down")))
    assert processor.process(claim(), bill_text("Bodyache")).verdict == NEEDS_REVIEW
    # ...but a rule-based rejection still works without the LLM
    assert processor.process(claim(total_claim_amount=9000), bill_text("Bodyache")).verdict == REJECTED
