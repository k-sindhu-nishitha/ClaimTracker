from dataclasses import replace

import pytest

from claimtrackr.config import ROOT, Settings
from claimtrackr.models import ClaimInput
from claimtrackr.pdf_utils import extract_text
from claimtrackr.processor import ClaimProcessor


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("claimtrackr")
    # explicit offline settings so tests never call a paid API
    return replace(Settings(llm_provider="offline", embedding_provider="offline"),
                   index_dir=tmp / "index", db_path=tmp / "claims.db")


@pytest.fixture(scope="session")
def processor(settings):
    return ClaimProcessor(settings)


@pytest.fixture()
def bill_text():
    return lambda name: extract_text(str(ROOT / "sample_bills" / f"{name}.pdf"))


@pytest.fixture()
def claim():
    return lambda **kw: ClaimInput(**{"name": "Surbhit", "claim_reason": "Fever", "total_claim_amount": 3000, **kw})
