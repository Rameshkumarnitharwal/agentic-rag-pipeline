"""LangGraph agent: planner -> retriever -> (re-plan loop) -> responder.

    START -> planner --needs_retrieval--> retriever --sufficient or max iterations--> responder -> END
                 |                            |
                 +--no retrieval needed-------|----------------------------------->  responder
                 ^                            |
                 +------ nothing relevant ----+   (cyclic re-plan with different queries)

Conversation memory is kept per ``thread_id`` by the checkpointer.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.observability import span
from app.gateway import LLMGateway
from app.services.retrieval import Retriever

from .memory import build_checkpointer
from .nodes import PlannerNode, ResponderNode, RetrieverNode
from .state import AgentState


def _with_span(name: str, node: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    """Wrap a node in a Logfire span carrying the step summary it produced."""

    def run(state: AgentState) -> AgentState:
        with span(f"agent.{name}", iteration=state.get("iteration", 0)) as s:
            update = node(state)
            if s is not None and update.get("steps"):
                s.set_attribute("summary", update["steps"][-1].get("summary", ""))
            return update

    run.__name__ = name
    return run


def build_graph(
    llm: LLMGateway,
    retriever: Retriever,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
    planner: PlannerNode | None = None,
):
    def after_planner(state: AgentState) -> str:
        return "retriever" if state.get("needs_retrieval") else "responder"

    def after_retriever(state: AgentState) -> str:
        if state.get("retrieval_sufficient") or state.get("iteration", 1) >= settings.agent_max_iterations:
            return "responder"
        return "planner"

    graph = StateGraph(AgentState)
    graph.add_node("planner", _with_span("planner", planner or PlannerNode(llm, settings)))
    graph.add_node("retriever", _with_span("retriever", RetrieverNode(retriever, settings)))
    graph.add_node("responder", _with_span("responder", ResponderNode(llm, settings)))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_planner, ["retriever", "responder"])
    graph.add_conditional_edges("retriever", after_retriever, ["planner", "responder"])
    graph.add_edge("responder", END)
    return graph.compile(checkpointer=checkpointer)


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    standalone_question: str
    thread_id: str
    trace_id: str
    model: str
    iterations: int
    retrieved: list[dict[str, Any]] = field(default_factory=list)  # full contexts (used by evals)


class RAGAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMGateway | None = None,
        retriever: Retriever | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(self.settings)
        llm = llm or LLMGateway(self.settings)
        self.planner = PlannerNode(llm, self.settings)
        self.graph = build_graph(
            llm, self.retriever, self.settings, checkpointer or build_checkpointer(self.settings), self.planner
        )

    def plan(self, question: str, thread_id: str, trace_id: str | None = None) -> dict[str, Any]:
        """Run the planner on its own (read-only: conversation memory is not modified).

        Lets callers overlap planning with the input guardrail; pass the result to ``run(plan=...)``.
        """
        started = time.perf_counter()
        with span("agent.planner.speculative"):
            plan = self.planner.make_plan({"question": question, "trace_id": trace_id, "history": self.history(thread_id)})
        return {"plan": plan.model_dump(), "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def run(
        self,
        question: str,
        thread_id: str | None = None,
        trace_id: str | None = None,
        filters: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        acl_groups: list[str] | None = None,
    ) -> AgentResult:
        turn_input, config = self._prepare(question, thread_id, trace_id, filters, plan, acl_groups, stream=False)
        # Persist once at the end of the turn rather than after every node (fewer writes, same result).
        state = self.graph.invoke(turn_input, config=config, durability="exit")
        return self._result(state, turn_input, config)

    def run_stream(
        self,
        question: str,
        thread_id: str | None = None,
        trace_id: str | None = None,
        filters: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        acl_groups: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Like ``run`` but yields events as they happen:
        ``{"type": "step", "step": {...}}`` after each node, ``{"type": "token", "text": ...}`` while the
        answer is generated (already PII-redacted), and finally ``{"type": "result", "result": AgentResult}``.
        """
        turn_input, config = self._prepare(question, thread_id, trace_id, filters, plan, acl_groups, stream=True)
        for mode, data in self.graph.stream(turn_input, config=config, stream_mode=["custom", "updates"], durability="exit"):
            if mode == "custom":
                yield data
            else:
                for update in data.values():
                    if update and update.get("steps"):
                        yield {"type": "step", "step": update["steps"][-1]}
        state = self.graph.get_state(config).values
        yield {"type": "result", "result": self._result(state, turn_input, config)}

    def _prepare(self, question, thread_id, trace_id, filters, plan, acl_groups, stream: bool):
        thread_id = thread_id or uuid.uuid4().hex
        trace_id = trace_id or uuid.uuid4().hex
        # Reset every per-turn field; `history` is restored from the checkpoint for this thread.
        turn_input: AgentState = {
            "question": question,
            "trace_id": trace_id,
            "filters": filters,
            "acl_groups": acl_groups,
            "precomputed_plan": plan,
            "stream": stream,
            "iteration": 0,
            "search_queries": [],
            "tried_queries": [],
            "retrieved": [],
            "retrieval_sufficient": False,
            "answer": "",
            "citations": [],
            "model": "",
            "steps": [],
        }
        config = {
            "configurable": {"thread_id": thread_id},
            # Each planner->retriever cycle is 2 steps; leave headroom for the responder.
            "recursion_limit": 4 + 2 * self.settings.agent_max_iterations,
            "metadata": {"trace_id": trace_id, "thread_id": thread_id},  # shown on LangSmith runs
            "run_name": "rag_agent",
        }
        return turn_input, config

    def _result(self, state: AgentState, turn_input: AgentState, config: dict[str, Any]) -> AgentResult:
        return AgentResult(
            answer=state["answer"],
            citations=state.get("citations", []),
            steps=state.get("steps", []),
            standalone_question=state.get("standalone_question", turn_input["question"]),
            thread_id=config["configurable"]["thread_id"],
            trace_id=turn_input["trace_id"],
            model=state.get("model", ""),
            iterations=state.get("iteration", 0),
            retrieved=state.get("retrieved", []),
        )

    def history(self, thread_id: str) -> list[dict[str, str]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(snapshot.values.get("history", [])) if snapshot else []
