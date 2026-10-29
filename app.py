"""Flask web app.   Run:  python app.py   ->  http://127.0.0.1:8081"""
from __future__ import annotations

import io
import logging
from collections import Counter
from dataclasses import asdict
from datetime import date
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

from claimtrackr.config import ROOT, Settings, get_settings
from claimtrackr.models import ClaimInput
from claimtrackr.pdf_utils import extract_text
from claimtrackr.processor import build_processor
from claimtrackr.rules import parse_amount
from claimtrackr.storage import ClaimStore

logging.basicConfig(level=logging.INFO)

CLAIM_TYPES = ["General Practitioner", "Specialist", "Hospital admission", "Diagnostic tests",
               "Medication", "Physiotherapy", "Dental / Optical"]

# Shown on the home page: downloadable bills + a button that pre-fills the form for them.
SAMPLES = [
    {"file": "Test_Bill_ViralFever.pdf", "title": "Viral fever", "tag": "Should be approved", "tone": "ok",
     "fill": {"name": "Anita Rao", "claim_type": "General Practitioner", "claim_reason": "Viral fever",
              "total_claim_amount": 2500, "medical_facility": "City Care Hospital"}},
    {"file": "Bodyache.pdf", "title": "Body ache with fever", "tag": "Should be approved", "tone": "ok",
     "fill": {"name": "Surbhit", "claim_type": "General Practitioner", "claim_reason": "Fever and body ache",
              "total_claim_amount": 3000, "medical_facility": "Apollo Hospitals"}},
    {"file": "HIV.pdf", "title": "HIV treatment", "tag": "Excluded condition", "tone": "bad",
     "fill": {"name": "Surbhit", "claim_type": "Specialist", "claim_reason": "Medication",
              "total_claim_amount": 3000, "medical_facility": "Apollo Hospitals"}},
    {"file": "Alzheimer.pdf", "title": "Alzheimer\u2019s medication", "tag": "Excluded condition", "tone": "bad",
     "fill": {"name": "Surbhit", "claim_type": "Specialist", "claim_reason": "Medication",
              "total_claim_amount": 3000, "medical_facility": "Apollo Hospitals"}},
]


def _parse_request():
    """Validate the form + uploaded PDF. Returns (ClaimInput, bill_text) or raises ValueError."""
    form = request.form
    amount = parse_amount(form.get("total_claim_amount"))
    if not form.get("name", "").strip() or amount is None or amount <= 0:
        raise ValueError("Please provide the claimant name and a valid claim amount.")
    upload = request.files.get("medical_bill")
    if upload is None or not upload.filename:
        raise ValueError("Please upload the medical bill as a PDF.")
    data = upload.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a PDF.")
    try:
        bill_text = extract_text(io.BytesIO(data))
    except Exception:  # corrupt / encrypted PDF
        bill_text = ""
    claim = ClaimInput(
        name=form["name"].strip(), address=form.get("address", ""), claim_type=form.get("claim_type", ""),
        claim_reason=form.get("claim_reason", ""), date=form.get("date", ""),
        medical_facility=form.get("medical_facility", ""), total_claim_amount=amount,
        description=form.get("description", ""))
    return claim, bill_text


def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or get_settings()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload cap
    app.jinja_env.filters["money"] = lambda v: "-" if v is None else f"{float(v):,.0f}"
    store = ClaimStore(settings.db_path)
    processor = build_processor(settings)  # loads (or builds) the FAISS index once at startup

    @app.get("/")
    def index():
        return render_template("index.html", samples=SAMPLES, claim_types=CLAIM_TYPES, today=date.today().isoformat(),
                               engine=processor.backend.name, embeddings=processor.embedder.name,
                               chunks=len(processor.index.chunks))

    @app.post("/")
    def submit():
        try:
            claim, bill_text = _parse_request()
        except ValueError as exc:
            app.logger.warning("Rejected submission: %s", exc)
            return render_template("error.html", message=str(exc)), 400
        result = processor.process(claim, bill_text)
        claim_id = store.save(claim, result)
        return render_template("result.html", claim=claim, r=result, claim_id=claim_id)

    @app.post("/api/verify")
    def api_verify():
        """Same pipeline as JSON: curl -F name=... -F total_claim_amount=3000 -F medical_bill=@bill.pdf"""
        try:
            claim, bill_text = _parse_request()
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        result = processor.process(claim, bill_text)
        return jsonify(id=store.save(claim, result), **asdict(result))

    @app.get("/samples/<path:filename>")
    def sample_bill(filename: str):
        if not filename.lower().endswith(".pdf"):
            return render_template("error.html", message="Sample not found."), 404
        return send_from_directory(ROOT / "sample_bills", filename, mimetype="application/pdf")

    @app.get("/history")
    def history():
        rows = store.recent()
        return render_template("history.html", rows=rows, stats=Counter(r["verdict"] for r in rows))

    @app.get("/claims/<int:claim_id>")
    def claim_detail(claim_id: int):
        rec = store.get(claim_id)
        if rec is None:
            return render_template("error.html", message="Claim not found."), 404
        return render_template("result.html", claim=rec["claim"], r=rec["result"], claim_id=claim_id)

    @app.get("/health")
    def health():
        return jsonify(status="ok", llm=processor.backend.name, embeddings=processor.embedder.name,
                       indexed_chunks=len(processor.index.chunks))

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8081, debug=False)
