"""Responder node: grounded answer generation with numbered source citations."""

from __future__ import annotations

import re
import time

from app.agents.prompts import NO_CONTEXT_ANSWER, RESPONDER_NO_RETRIEVAL_SYSTEM, RESPONDER_SYSTEM
from app.agents.state import AgentState
from app.config import Settings
from app.gateway import LLMGateway

_CITATION = re.compile(r"\[(\d+)\]")


def _location(metadata: dict) -> str:
    if metadata.get("page"):
        return f"page {metadata['page']}"
    if metadata.get("slide"):
        return f"slide {metadata['slide']}"
    return ""


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        header = f"[{i}] {meta.get('source', 'unknown')}"
        if loc := _location(meta):
            header += f" ({loc})"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


class ResponderNode:
    def __init__(self, llm: LLMGateway, settings: Settings):
        self.llm = llm
        self.settings = settings

    def __call__(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        question = state["question"]
        history = state.get("history", [])
        chunks = state.get("retrieved", [])
        recent = history[-self.settings.agent_history_turns * 2 :]
        model = ""
        citations: list[dict] = []

        if state.get("needs_retrieval", True) and not chunks:
            # Searched but found nothing relevant: refuse instead of letting the LLM guess.
            answer = NO_CONTEXT_ANSWER
            summary = "No relevant documentation found; returned a fallback answer"
        else:
            if chunks:
                system = RESPONDER_SYSTEM.format(context=format_context(chunks))
            else:
                system = RESPONDER_NO_RETRIEVAL_SYSTEM.format(topics=self.settings.guardrails_allowed_topics)
            response = self.llm.chat(
                [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}],
                trace_id=state.get("trace_id"),
                metadata={"node": "responder"},
                max_tokens=self.settings.agent_answer_max_tokens,
            )
            answer, model = response.content.strip(), response.model
            citations = self._citations(answer, chunks)
            summary = f"Answered with {len(citations)} cited sources" if chunks else "Answered from conversation"

        step = {
            "node": "responder",
            "summary": summary,
            "detail": {"model": model, "context_chunks": len(chunks)},
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        new_history = [*history, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return {
            "answer": answer,
            "citations": citations,
            "model": model,
            "history": new_history[-self.settings.agent_history_turns * 2 :],
            "steps": [*state.get("steps", []), step],
        }

    @staticmethod
    def _citations(answer: str, chunks: list[dict]) -> list[dict]:
        cited = sorted({int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(chunks)})
        out = []
        for n in cited:
            chunk = chunks[n - 1]
            meta = chunk["metadata"]
            out.append(
                {
                    "ref": n,
                    "source": meta.get("source"),
                    "location": _location(meta) or None,
                    "dataset": meta.get("dataset"),
                    "score": chunk.get("rerank_score") if chunk.get("rerank_score") is not None else chunk.get("vector_score"),
                    "snippet": chunk["text"][:300],
                }
            )
        return out
