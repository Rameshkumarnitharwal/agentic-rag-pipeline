"""Turn 👎-rated answers into golden-dataset candidates for human review.

Writes evals/feedback_candidates.jsonl in the evals/dataset.jsonl format, with an empty ``reference``
for a reviewer to fill in from the source documents. Reviewed lines can then be appended to
evals/dataset.jsonl so the eval suite covers real failures.

Usage:
    python -m evals.feedback_export                  # all 👎 so far
    python -m evals.feedback_export --since-days 7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import get_settings
from app.feedback import FeedbackStore

OUT = Path(__file__).resolve().parent / "feedback_candidates.jsonl"


def export(store: FeedbackStore, out: Path, since: float = 0.0) -> int:
    rows = store.downvoted(since)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps({
                "id": f"fb-{row['trace_id'][:8]}",
                "category": "rag",
                "question": row["question"],
                "reference": "",  # TODO(reviewer): write the correct answer from the source documents
                "expected_sources": sorted({c["source"] for c in row["citations"] if c.get("source")}),
                "review": {"bad_answer": row["answer"], "user_comment": row["comment"], "trace_id": row["trace_id"],
                           "model": row["model"]},
            }, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 👎 answers as golden-dataset candidates.")
    parser.add_argument("--since-days", type=float, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    settings = get_settings()
    since = time.time() - args.since_days * 86400 if args.since_days else 0.0
    n = export(FeedbackStore(settings.feedback_db_path), args.out, since)
    print(f"Wrote {n} candidate(s) to {args.out}. Fill in 'reference' (and fix 'expected_sources'), drop the "
          f"'review' field, then append to evals/dataset.jsonl.")


if __name__ == "__main__":
    main()
