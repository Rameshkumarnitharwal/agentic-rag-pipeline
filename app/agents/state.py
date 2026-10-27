"""LangGraph state shared by the agent nodes.

No reducers: every field is overwritten by the node that returns it. Per-turn fields are reset
in the graph input on each new question, while ``history`` persists across turns via the checkpointer.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Turn(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class Step(TypedDict, total=False):
    """One reasoning step, surfaced to the UI for transparency."""

    node: str
    summary: str
    detail: dict[str, Any]
    latency_ms: float


class AgentState(TypedDict, total=False):
    # Conversation memory (persists across turns for a thread_id)
    history: list[Turn]

    # Per-turn input
    question: str
    trace_id: str
    filters: dict[str, Any] | None

    # Plan computed before the graph ran (in parallel with the input guardrail); used on iteration 1
    precomputed_plan: dict[str, Any] | None

    # Planner output
    standalone_question: str
    needs_retrieval: bool
    search_queries: list[str]
    tried_queries: list[str]
    iteration: int

    # Retriever output: chunks as dicts (id, text, metadata, vector_score, rerank_score)
    retrieved: list[dict[str, Any]]
    retrieval_sufficient: bool

    # Responder output
    answer: str
    citations: list[dict[str, Any]]
    model: str

    steps: list[Step]
