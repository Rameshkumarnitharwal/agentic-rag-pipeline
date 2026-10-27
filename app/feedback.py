"""User feedback loop: record every answer, accept 👍/👎 on it, export 👎 cases for the golden dataset.

Answers are stored server-side (keyed by ``trace_id``) so feedback refers to what the system actually
said, and only the caller who received an answer can rate it. Questions are PII-redacted before storage;
answers already are (responder).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.guardrails.pii import redact_pii

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    trace_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    principal TEXT NOT NULL,
    thread_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations TEXT NOT NULL,       -- JSON list
    model TEXT,
    blocked INTEGER NOT NULL,
    latency_ms REAL,
    cost_usd REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    trace_id TEXT PRIMARY KEY REFERENCES responses(trace_id),
    created_at REAL NOT NULL,
    rating INTEGER NOT NULL,       -- 1 = 👍, -1 = 👎
    comment TEXT
);
CREATE INDEX IF NOT EXISTS responses_created ON responses(created_at);
"""


@dataclass
class FeedbackSummary:
    responses: int
    up: int
    down: int


class FeedbackStore:
    def __init__(self, path: Path, pii_entities: list[str] | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._pii_entities = pii_entities

    def record_response(
        self,
        *,
        trace_id: str,
        principal: str,
        thread_id: str | None,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        model: str | None,
        blocked: bool,
        latency_ms: float | None,
        cost_usd: float | None,
    ) -> None:
        question, _ = redact_pii(question, self._pii_entities) if self._pii_entities is not None else (question, [])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trace_id, time.time(), principal, thread_id, question, answer, json.dumps(citations), model,
                 int(blocked), latency_ms, cost_usd),
            )
            self._conn.commit()

    def add_feedback(self, trace_id: str, principal: str, rating: int, comment: str | None) -> bool:
        """Store (or replace) feedback. False if the answer doesn't exist or belongs to someone else."""
        with self._lock:
            row = self._conn.execute("SELECT principal FROM responses WHERE trace_id = ?", (trace_id,)).fetchone()
            if row is None or row[0] != principal:
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?)", (trace_id, time.time(), rating, comment)
            )
            self._conn.commit()
            return True

    def summary(self) -> FeedbackSummary:
        with self._lock:
            responses = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            up, down = self._conn.execute(
                "SELECT COALESCE(SUM(rating = 1), 0), COALESCE(SUM(rating = -1), 0) FROM feedback"
            ).fetchone()
        return FeedbackSummary(responses=responses, up=up, down=down)

    def downvoted(self, since: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.trace_id, r.question, r.answer, r.citations, r.model, f.comment, f.created_at
                   FROM feedback f JOIN responses r USING (trace_id)
                   WHERE f.rating = -1 AND f.created_at >= ? ORDER BY f.created_at""",
                (since,),
            ).fetchall()
        return [
            {"trace_id": t, "question": q, "answer": a, "citations": json.loads(c), "model": m, "comment": cm,
             "rated_at": ts}
            for t, q, a, c, m, cm, ts in rows
        ]
