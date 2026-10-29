"""SQLite history of processed claims (the 'backend workflow' part: every decision is auditable)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import ClaimInput, ClaimResult


class ClaimStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, claimant TEXT, claim_amount REAL,
                diagnosis TEXT, verdict TEXT NOT NULL, claim_json TEXT NOT NULL, result_json TEXT NOT NULL)""")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, claim: ClaimInput, result: ClaimResult) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO claims (created_at, claimant, claim_amount, diagnosis, verdict, claim_json, result_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), claim.name, claim.total_claim_amount,
                 result.bill.disease, result.verdict, json.dumps(asdict(claim)), json.dumps(asdict(result))))
            return cur.lastrowid

    def recent(self, limit: int = 50) -> List[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, created_at, claimant, claim_amount, diagnosis, verdict "
                             "FROM claims ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, claim_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "created_at": row["created_at"],
                "claim": json.loads(row["claim_json"]), "result": json.loads(row["result_json"])}
