"""Responder node: grounded answer generation with numbered source citations."""

from __future__ import annotations

import re
import time

from app.agents.prompts import NO_CONTEXT_ANSWER, RESPONDER_NO_RETRIEVAL_SYSTEM, RESPONDER_SYSTEM
from app.agents.state import AgentState
from app.config import Settings
from app.gateway import LLMGateway
from app.guardrails.pii import detect_pii, redact_pii
from langgraph.config import get_stream_writer

_CITATION = re.compile(r"\[(\d+)\]")
_SEGMENT_END = re.compile(r"(?:\n|[.!?:](?=\s))")  # stream in whole lines / sentences
_MAX_UNSEGMENTED = 300  # long runs without a boundary (e.g. code) are flushed at whitespace


def _cut_point(buffer: str, entities: list[str] | None) -> int:
    """Where the buffered stream can safely be emitted up to (0 = keep buffering).

    Emits whole lines/sentences so a PII value (which never spans a sentence boundary) is always
    redacted as a whole; never cuts inside a detected PII match.
    """
    ends = [m.end() for m in _SEGMENT_END.finditer(buffer)]
    cut = ends[-1] if ends else 0
    if not cut and len(buffer) > _MAX_UNSEGMENTED:
        cut = buffer.rfind(" ", 0, len(buffer) - 40) + 1  # hold back a tail that may hold partial PII
    if cut and entities is not None:
        for m in detect_pii(buffer, entities):
            if m.start < cut < m.end:
                cut = m.start
    return max(cut, 0)


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
        pii_redacted: list[str] = []

        if state.get("needs_retrieval", True) and not chunks:
            # Searched but found nothing relevant: refuse instead of letting the LLM guess.
            answer = NO_CONTEXT_ANSWER
            summary = "No relevant documentation found; returned a fallback answer"
        else:
            if chunks:
                system = RESPONDER_SYSTEM.format(context=format_context(chunks))
            else:
                system = RESPONDER_NO_RETRIEVAL_SYSTEM.format(topics=self.settings.guardrails_allowed_topics)
            messages = [{"role": "system", "content": system}, *recent, {"role": "user", "content": question}]
            entities = self.settings.pii_entities if self.settings.pii_redaction_enabled else None
            found: list[str] = []
            if state.get("stream"):
                answer, found, model = self._stream_answer(messages, state.get("trace_id"), entities)
            else:
                response = self.llm.chat(
                    messages, trace_id=state.get("trace_id"), metadata={"node": "responder"},
                    max_tokens=self.settings.agent_answer_max_tokens,
                )
                answer, model = response.content.strip(), response.model
                if entities is not None:
                    # Before the answer is returned, logged or saved to conversation memory.
                    answer, found = redact_pii(answer, entities)
            citations = self._citations(answer, chunks)
            if entities is not None:
                for citation in citations:
                    citation["snippet"], snippet_found = redact_pii(citation["snippet"], entities)
                    found += snippet_found
                pii_redacted = sorted(set(found))
            summary = f"Answered with {len(citations)} cited sources" if chunks else "Answered from conversation"

        step = {
            "node": "responder",
            "summary": summary,
            "detail": {"model": model, "context_chunks": len(chunks), "pii_redacted": pii_redacted},
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

    def _stream_answer(
        self, messages: list[dict], trace_id: str | None, entities: list[str] | None
    ) -> tuple[str, list[str], str]:
        """Stream the answer to the graph's stream writer, redacting each segment before it is sent."""
        writer = get_stream_writer()
        emitted: list[str] = []
        found: list[str] = []
        buffer = ""

        def emit(text: str) -> None:
            if entities is not None:
                text, hits = redact_pii(text, entities)
                found.extend(hits)
            if text:
                emitted.append(text)
                writer({"type": "token", "text": text})

        served: list[str] = []
        for delta in self.llm.chat_stream(
            messages, trace_id=trace_id, metadata={"node": "responder"}, max_tokens=self.settings.agent_answer_max_tokens,
            on_complete=served.append,
        ):
            buffer += delta
            if cut := _cut_point(buffer, entities):
                emit(buffer[:cut])
                buffer = buffer[cut:]
        emit(buffer)
        # The final answer is exactly what was streamed (already redacted), so memory and UI agree.
        return "".join(emitted).strip(), found, served[0] if served else ""

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
