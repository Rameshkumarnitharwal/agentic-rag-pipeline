"""Planner node: history-aware query rewriting and retrieval planning."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.agents.prompts import PLANNER_REPLAN, PLANNER_SYSTEM
from app.agents.state import AgentState
from app.config import Settings
from app.gateway import LLMGateway, LLMGatewayError
from app.gateway.client import FAST

logger = logging.getLogger(__name__)


class Plan(BaseModel):
    standalone_question: str = Field(description="The latest user message rewritten to be self-contained")
    needs_retrieval: bool = Field(description="Whether the documentation must be searched")
    search_queries: list[str] = Field(default_factory=list, description="Search queries (empty if no retrieval)")
    reasoning: str = Field(default="", description="One sentence explaining the plan")


def format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-max_turns * 2 :]
    return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in recent) or "(no previous messages)"


class PlannerNode:
    def __init__(self, llm: LLMGateway, settings: Settings):
        self.llm = llm
        self.settings = settings

    def make_plan(self, state: AgentState) -> Plan:
        """The planner's LLM call. Also used speculatively, in parallel with the input guardrail."""
        question = state["question"]
        tried = state.get("tried_queries", [])
        system = PLANNER_SYSTEM.format(
            topics=self.settings.guardrails_allowed_topics, max_queries=self.settings.agent_max_search_queries
        )
        if tried:
            system += "\n\n" + PLANNER_REPLAN.format(tried="; ".join(tried))
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Conversation history:\n{format_history(state.get('history', []), self.settings.agent_history_turns)}"
                f"\n\nLatest user message: {question}",
            },
        ]
        try:
            plan, _ = self.llm.chat_json(
                messages, Plan, trace_id=state.get("trace_id"), metadata={"node": "planner"}, max_tokens=400, tier=FAST
            )
            return plan
        except LLMGatewayError as exc:
            # Degrade gracefully: search with the raw question rather than failing the request.
            logger.warning("Planner failed, falling back to raw question: %s", exc)
            return Plan(standalone_question=question, needs_retrieval=True, search_queries=[question], reasoning="planner unavailable")

    def __call__(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        question = state["question"]
        iteration = state.get("iteration", 0) + 1
        tried = state.get("tried_queries", [])

        precomputed = state.get("precomputed_plan") if iteration == 1 else None
        if precomputed:
            # Planned already, in parallel with the input guardrail (see app/pipeline.py).
            plan = Plan(**precomputed["plan"])
            latency_ms = precomputed["latency_ms"]
        else:
            plan = self.make_plan(state)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)

        queries = [q.strip() for q in plan.search_queries if q.strip()][: self.settings.agent_max_search_queries]
        needs_retrieval = plan.needs_retrieval
        if needs_retrieval and not queries:
            queries = [plan.standalone_question or question]
        if tried:
            # On a re-plan we must search again; drop exact repeats of earlier queries.
            needs_retrieval = True
            queries = [q for q in queries if q not in tried] or [plan.standalone_question or question]

        step = {
            "node": "planner",
            "summary": (
                f"Search for: {'; '.join(queries)}" if needs_retrieval else "No document search needed"
            )
            + (f" (re-plan, attempt {iteration})" if iteration > 1 else ""),
            "detail": {"standalone_question": plan.standalone_question, "reasoning": plan.reasoning,
                       "ran_in_parallel_with_guardrail": bool(precomputed)},
            "latency_ms": latency_ms,
        }
        return {
            "standalone_question": plan.standalone_question or question,
            "needs_retrieval": needs_retrieval,
            "search_queries": queries if needs_retrieval else [],
            "iteration": iteration,
            "steps": [*state.get("steps", []), step],
        }
