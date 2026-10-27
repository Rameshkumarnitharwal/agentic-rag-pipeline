"""Retriever node: runs each planned query, merges and dedupes the reranked chunks."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from app.agents.state import AgentState
from app.config import Settings
from app.services.retrieval import Retriever

logger = logging.getLogger(__name__)


def _score(chunk: dict) -> float:
    return chunk["rerank_score"] if chunk.get("rerank_score") is not None else chunk["vector_score"]


class RetrieverNode:
    def __init__(self, retriever: Retriever, settings: Settings):
        self.retriever = retriever
        self.settings = settings

    def __call__(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        queries = state.get("search_queries") or [state.get("standalone_question") or state["question"]]

        best: dict[str, dict] = {}
        candidates = filtered = 0
        try:
            # One batched retrieval for all planned queries, reranked against the standalone question.
            result = self.retriever.retrieve_many(
                queries,
                rerank_query=state.get("standalone_question") or state["question"],
                top_n=self.settings.agent_max_context_chunks,
                filters=state.get("filters"),
                acl_groups=state.get("acl_groups"),
            )
        except Exception as exc:
            logger.exception("Retrieval failed for queries %r", queries)
            per_query = [{"queries": queries, "error": str(exc)[:200]}]
        else:
            candidates, filtered = result.candidates, result.filtered_out
            per_query = [{
                "queries": queries, "kept": len(result.chunks), "provider": result.embedding_provider,
                "reranked": result.reranked, "timings_ms": result.timings_ms,
            }]
            for chunk in result.chunks:
                best[chunk.id] = asdict(chunk)

        # Keep chunks from earlier attempts this turn too, so a re-plan only adds evidence.
        for chunk in state.get("retrieved", []):
            best.setdefault(chunk["id"], chunk)
        chunks = sorted(best.values(), key=_score, reverse=True)[: self.settings.agent_max_context_chunks]

        sources = sorted({c["metadata"].get("source", "?") for c in chunks})
        step = {
            "node": "retriever",
            "summary": f"Kept {len(chunks)} relevant chunks from {len(sources)} documents"
            + (f" ({filtered} low-relevance chunks filtered out)" if filtered else ""),
            "detail": {"queries": per_query, "candidates": candidates, "sources": sources},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        return {
            "retrieved": chunks,
            "retrieval_sufficient": bool(chunks),
            "tried_queries": [*state.get("tried_queries", []), *queries],
            "steps": [*state.get("steps", []), step],
        }
