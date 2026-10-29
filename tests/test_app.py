import io

import pytest

from app import create_app
from claimtrackr.config import ROOT


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def _form(bill="Bodyache", **overrides):
    data = {"name": "Surbhit", "claim_type": "General Practitoner", "claim_reason": "Fever",
            "total_claim_amount": "3000", "medical_facility": "Apollo", "date": "2024-03-18", "description": "x"}
    data.update(overrides)
    data["medical_bill"] = (io.BytesIO((ROOT / "sample_bills" / f"{bill}.pdf").read_bytes()), f"{bill}.pdf")
    return data


def test_pages_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200
    health = client.get("/health").get_json()
    assert health["status"] == "ok" and health["indexed_chunks"] > 50


def test_submit_valid_claim_shows_approved_and_is_saved(client):
    resp = client.post("/", data=_form(), content_type="multipart/form-data")
    assert resp.status_code == 200 and b"APPROVED" in resp.data
    assert b"Surbhit" in client.get("/history").data


def test_submit_excluded_claim_shows_rejected(client):
    resp = client.post("/", data=_form("HIV"), content_type="multipart/form-data")
    assert b"Rejected" in resp.data and b"HIV/AIDS" in resp.data


def test_api_returns_json_with_trace(client):
    body = client.post("/api/verify", data=_form("Alzheimer"), content_type="multipart/form-data").get_json()
    assert body["verdict"] == "REJECTED" and len(body["steps"]) >= 6 and body["id"] >= 1


def test_bad_input_is_rejected_cleanly(client):
    assert client.post("/", data={"name": "A"}, content_type="multipart/form-data").status_code == 400
    bad = {"name": "A", "total_claim_amount": "100", "medical_bill": (io.BytesIO(b"not a pdf"), "x.pdf")}
    assert client.post("/", data=bad, content_type="multipart/form-data").status_code == 400


def test_report_text_is_escaped_not_injected(client):
    resp = client.post("/", data=_form(description="<script>alert(1)</script>"), content_type="multipart/form-data")
    assert b"<script>alert(1)</script>" not in resp.data


def test_static_assets_and_sample_downloads(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    pdf = client.get("/samples/Test_Bill_ViralFever.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    assert client.get("/samples/../app.py").status_code == 404
    assert client.get("/samples/notes.txt").status_code == 404


def test_new_sample_claim_is_approved(client):
    data = {"name": "Anita Rao", "total_claim_amount": "2500", "claim_reason": "Viral fever",
            "medical_bill": (io.BytesIO((ROOT / "sample_bills" / "Test_Bill_ViralFever.pdf").read_bytes()), "b.pdf")}
    assert b"Approved" in client.post("/", data=data, content_type="multipart/form-data").data
