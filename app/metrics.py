"""Prometheus metrics for the RAG pipeline and guardrails (exposed at /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

QUERY_REQUESTS = Counter(
    "rag_query_requests_total",
    "/query requests by outcome",
    ["outcome"],  # answered | no_context | blocked_input | blocked_output | error | rate_limited | unauthorized
)
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end /query latency",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
NODE_LATENCY = Histogram(
    "rag_node_latency_seconds",
    "Latency per agent node / guardrail stage",
    ["node"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15),
)
GUARDRAIL_DECISIONS = Counter(
    "guardrails_decisions_total",
    "Guardrail decisions by stage, result and deciding rail",
    ["stage", "result", "rail"],
)
RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Chunks passed to the responder per query",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
)
AGENT_ITERATIONS = Histogram("rag_agent_iterations", "Planner/retriever cycles per query", buckets=(1, 2, 3, 4, 5))
CITED_DATASET = Counter(
    "rag_cited_chunks_total",
    "Cited chunks by dataset label (true vs noisy) — noisy citations indicate retrieval leakage",
    ["dataset"],
)
LLM_MODEL_USED = Counter("rag_llm_answers_total", "Answers by the model that served them (shows fallbacks)", ["model"])
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens used by /query", ["model", "kind"])  # kind: prompt|completion
LLM_COST = Counter("rag_llm_cost_usd_total", "LLM cost of /query requests in USD (from LLM_PRICES)")
QUERY_COST = Histogram(
    "rag_query_cost_usd",
    "LLM cost per /query request in USD",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1),
)
