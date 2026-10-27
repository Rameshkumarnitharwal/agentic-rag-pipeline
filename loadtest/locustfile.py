"""Load test for the RAG API (Locust).

Simulated users ask the golden-dataset questions: 75% via /query, 25% via /query/stream, where time to
first token is reported as its own metric ("stream: first token"). Every request makes real LLM, Jina
and Qdrant calls, so a run costs money — start small.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless --users 5 --spawn-rate 1 --run-time 2m --csv loadtest/results/run

Env: RAG_API_KEY (bearer token, if the API requires one). Disable or raise RATE_LIMIT_PER_MINUTE on the
API for the test, or every user beyond the limit gets 429s.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from pathlib import Path

from locust import HttpUser, between, events, task

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"
QUESTIONS = [row["question"] for row in map(json.loads, DATASET.read_text().splitlines()) if row["category"] == "rag"]
HEADERS = {"Authorization": f"Bearer {os.environ['RAG_API_KEY']}"} if os.environ.get("RAG_API_KEY") else {}


class RagUser(HttpUser):
    wait_time = between(1, 3)  # think time between questions

    def on_start(self) -> None:
        self.thread_id = uuid.uuid4().hex  # each simulated user is one conversation

    @task(3)
    def query(self) -> None:
        with self.client.post("/query", json={"question": random.choice(QUESTIONS), "thread_id": self.thread_id},
                              headers=HEADERS, name="/query", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif response.json().get("blocked"):
                response.failure("legitimate question was blocked")

    @task(1)
    def query_stream(self) -> None:
        started = time.perf_counter()
        first_token_ms = None
        error = None
        with self.client.post("/query/stream", json={"question": random.choice(QUESTIONS), "thread_id": self.thread_id},
                              headers=HEADERS, name="/query/stream", stream=True, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            event = None
            for raw in response.iter_lines():
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    if event == "token" and first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    elif event == "error":
                        error = line[6:]
            if error:
                response.failure(f"stream error: {error}")
        if first_token_ms is not None:
            events.request.fire(request_type="SSE", name="stream: first token", response_time=first_token_ms,
                                response_length=0, exception=None, context={})
