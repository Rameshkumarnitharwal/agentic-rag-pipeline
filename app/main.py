"""FastAPI entrypoint: guardrails gate + synchronous /query running the LangGraph agent.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app import metrics
from app.agents import RAGAgent
from app.acl import effective_groups
from app.agents.prompts import NO_CONTEXT_ANSWER
from app.config import Settings, get_settings
from app.gateway import LLMGatewayError, track_usage
from app.guardrails import GuardrailResult, GuardrailsService
from app.observability import annotate, root_trace, setup_observability
from app.feedback import FeedbackStore
from app.pipeline import PipelineOutcome, answer_question, answer_question_stream
from app.security import Principal, RateLimiter, authenticate, build_rate_limiter

logger = logging.getLogger("app")


# ---------- Schemas ----------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question")
    thread_id: str | None = Field(default=None, max_length=128, description="Conversation id for follow-ups")
    filters: dict[str, Any] | None = Field(default=None, description='Exact-match metadata filters, e.g. {"dataset": "true"}')


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64, description="trace_id from the /query response")
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


class Citation(BaseModel):
    ref: int
    source: str | None
    location: str | None = None
    dataset: str | None = None
    score: float | None = None
    snippet: str


class Step(BaseModel):
    node: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class Usage(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None = Field(description="From LLM_PRICES; null if a model has no price")


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    blocked: bool = False
    blocked_stage: str | None = None
    thread_id: str
    trace_id: str
    standalone_question: str | None = None
    model: str | None = None
    latency_ms: float
    usage: Usage | None = None


# ---------- App factory ----------


def _guard_step(result: GuardrailResult) -> Step:
    verdict = "passed" if result.allowed else "blocked"
    return Step(
        node=f"guardrails_{result.stage.value}",
        summary=f"{result.stage.value.capitalize()} guardrails {verdict}" + (f" ({result.rail})" if not result.allowed else ""),
        detail={"rail": result.rail},
        latency_ms=result.latency_ms,
    )


def _record_guard(result: GuardrailResult) -> None:
    metrics.GUARDRAIL_DECISIONS.labels(
        stage=result.stage.value, result="allowed" if result.allowed else "blocked", rail=result.rail or "none"
    ).inc()
    metrics.NODE_LATENCY.labels(node=f"guardrails_{result.stage.value}").observe(result.latency_ms / 1000)


def create_app(
    settings: Settings | None = None,
    agent: RAGAgent | None = None,
    guardrails: GuardrailsService | None = None,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.init_error = None
        try:
            app.state.agent = agent or RAGAgent(settings)
            app.state.guardrails = guardrails or GuardrailsService(settings)
            app.state.rate_limiter = rate_limiter or build_rate_limiter(settings)
            app.state.feedback = (
                FeedbackStore(settings.feedback_db_path, settings.pii_entities if settings.pii_redaction_enabled else None)
                if settings.feedback_enabled else None
            )
            await run_in_threadpool(app.state.agent.retriever.embeddings.warmup)
        except Exception as exc:
            # Keep the process up so /health and /ready can report the problem.
            logger.exception("Startup failed")
            app.state.init_error = f"{type(exc).__name__}: {exc}"
        if not settings.api_key_list:
            logger.warning("API_KEYS is not set — /query is unauthenticated")
        yield

    app = FastAPI(title="Enterprise Agentic RAG", version="1.0.0", lifespan=lifespan)
    setup_observability(settings, app)

    def caller(request: Request) -> Principal:
        try:
            principal = authenticate(request, settings)
        except HTTPException:
            metrics.QUERY_REQUESTS.labels(outcome="unauthorized").inc()
            raise
        limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
        if limiter is not None:
            allowed, retry_after = limiter.hit(principal.id)
            if not allowed:
                metrics.QUERY_REQUESTS.labels(outcome="rate_limited").inc()
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded", headers={"Retry-After": str(retry_after)}
                )
        return principal

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request, response: Response) -> dict[str, Any]:
        error = getattr(request.app.state, "init_error", None)
        if error:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "unavailable", "error": error}
        return {"status": "ready", "guardrails": settings.guardrails_enabled, "auth": bool(settings.api_key_list)}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @dataclass
    class Turn:
        """Per-request context shared by /query and /query/stream."""

        state: Any
        principal: Principal
        question: str
        thread_id: str
        internal_thread: str
        trace_id: str
        filters: dict[str, Any] | None
        acl_groups: list[str] | None
        started: float

        def respond(self, **kwargs: Any) -> QueryResponse:
            elapsed = time.perf_counter() - self.started
            metrics.QUERY_LATENCY.observe(elapsed)
            return QueryResponse(thread_id=self.thread_id, trace_id=self.trace_id, latency_ms=round(elapsed * 1000, 1), **kwargs)

    def start_turn(body: QueryRequest, request: Request, principal: Principal) -> Turn:
        state = request.app.state
        if getattr(state, "init_error", None):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not initialised; see /ready")
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"Question exceeds {settings.max_question_chars} characters"
            )
        if body.filters and "allowed_groups" in body.filters:
            # Access control comes from the API key, never from the request.
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "'allowed_groups' cannot be used as a filter")
        thread_id = body.thread_id or uuid.uuid4().hex
        return Turn(
            state=state, principal=principal, question=body.question.strip(), thread_id=thread_id,
            # Namespace conversation memory by caller so one client can't read another's thread.
            internal_thread=f"{principal.id}:{thread_id}", trace_id=uuid.uuid4().hex, filters=body.filters,
            acl_groups=effective_groups(principal.groups) if settings.acl_enabled else None,
            started=time.perf_counter(),
        )

    def gateway_error(turn: Turn, exc: Exception) -> HTTPException:
        metrics.QUERY_REQUESTS.labels(outcome="error").inc()
        if isinstance(exc, LLMGatewayError):
            logger.error("LLM gateway failure (trace %s): %s", turn.trace_id, exc)
            return HTTPException(status.HTTP_502_BAD_GATEWAY, "The language model is unavailable. Please retry.")
        logger.error("Agent failure (trace %s)", turn.trace_id, exc_info=exc)
        return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error")

    def to_response(turn: Turn, outcome: PipelineOutcome) -> QueryResponse:
        """Map a pipeline outcome to the API response (raises HTTPException for 503s), with metrics."""
        input_check = outcome.input_check
        _record_guard(input_check)
        steps = [_guard_step(input_check)]
        if not input_check.allowed and (input_check.rail or "").startswith("error:"):
            # Fail-closed because the rail itself couldn't run (LLM outage) — that's not a policy block.
            metrics.QUERY_REQUESTS.labels(outcome="error").inc()
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Safety checks are unavailable. Please retry shortly.")
        if not input_check.allowed:
            metrics.QUERY_REQUESTS.labels(outcome="blocked_input").inc()
            return turn.respond(answer=input_check.message or "", steps=steps, blocked=True, blocked_stage="input")

        result = outcome.agent_result
        steps += [Step(**s) for s in result.steps]
        for s in result.steps:
            metrics.NODE_LATENCY.labels(node=s["node"]).observe(s.get("latency_ms", 0) / 1000)
            for entity in (s.get("detail") or {}).get("pii_redacted", []):
                metrics.PII_REDACTIONS.labels(entity=entity).inc()
        metrics.AGENT_ITERATIONS.observe(result.iterations)
        metrics.RETRIEVED_CHUNKS.observe(len(result.retrieved))

        if (output_check := outcome.output_check) is not None:
            _record_guard(output_check)
            steps.append(_guard_step(output_check))
            if not output_check.allowed and (output_check.rail or "").startswith("error:"):
                metrics.QUERY_REQUESTS.labels(outcome="error").inc()
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Safety checks are unavailable. Please retry shortly.")
            if not output_check.allowed:
                metrics.QUERY_REQUESTS.labels(outcome="blocked_output").inc()
                return turn.respond(
                    answer=output_check.message or "", steps=steps, blocked=True, blocked_stage="output",
                    standalone_question=result.standalone_question, model=result.model,
                )

        metrics.QUERY_REQUESTS.labels(outcome="no_context" if result.answer == NO_CONTEXT_ANSWER else "answered").inc()
        for citation in result.citations:
            metrics.CITED_DATASET.labels(dataset=citation.get("dataset") or "unknown").inc()
        if result.model:
            metrics.LLM_MODEL_USED.labels(model=result.model).inc()
        return turn.respond(
            answer=result.answer, citations=[Citation(**c) for c in result.citations], steps=steps,
            standalone_question=result.standalone_question, model=result.model or None,
        )

    def finish(turn: Turn, response: QueryResponse, usage: Any, run: Any) -> QueryResponse:
        """Attach usage/cost, close the trace, and store the answer for feedback."""
        cost = usage.cost_usd(settings.llm_prices)
        response.usage = Usage(llm_calls=len(usage.calls), prompt_tokens=usage.prompt_tokens,
                               completion_tokens=usage.completion_tokens, cost_usd=cost)
        for call in usage.calls:
            metrics.LLM_TOKENS.labels(model=call.model, kind="prompt").inc(call.prompt_tokens)
            metrics.LLM_TOKENS.labels(model=call.model, kind="completion").inc(call.completion_tokens)
        if cost is not None:
            metrics.LLM_COST.inc(cost)
            metrics.QUERY_COST.observe(cost)
        annotate(blocked=response.blocked, blocked_stage=response.blocked_stage, model=response.model,
                 citations=len(response.citations), cost_usd=cost, llm_calls=len(usage.calls))
        run.end(outputs={"answer": response.answer, "blocked": response.blocked,
                         "citations": [c.model_dump() for c in response.citations]})
        if (store := getattr(turn.state, "feedback", None)) is not None:
            store.record_response(
                trace_id=turn.trace_id, principal=turn.principal.id, thread_id=turn.thread_id, question=turn.question,
                answer=response.answer, citations=[c.model_dump() for c in response.citations],
                model=response.model, blocked=response.blocked, latency_ms=response.latency_ms, cost_usd=cost,
            )
        return response

    def trace_for(turn: Turn):
        return root_trace(
            "rag_query",
            inputs={"question": turn.question, "filters": turn.filters},
            metadata={"trace_id": turn.trace_id, "thread_id": turn.thread_id, "principal": turn.principal.id,
                      "acl_groups": ",".join(turn.principal.groups)},
        )

    def pipeline_kwargs(turn: Turn) -> dict[str, Any]:
        return dict(
            agent=turn.state.agent, guardrails=turn.state.guardrails, thread_id=turn.internal_thread,
            trace_id=turn.trace_id, filters=turn.filters, acl_groups=turn.acl_groups,
            parallel_planning=settings.agent_parallel_planning,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request, principal: Principal = Depends(caller)) -> QueryResponse:
        turn = start_turn(body, request, principal)
        # One root trace per request: guardrails, agent nodes, LLM and retrieval calls nest under it.
        with trace_for(turn) as run, track_usage() as usage:
            try:
                outcome = await answer_question(turn.question, **pipeline_kwargs(turn))
            except Exception as exc:
                raise gateway_error(turn, exc) from exc
            return finish(turn, to_response(turn, outcome), usage, run)

    @app.post("/query/stream")
    async def query_stream(body: QueryRequest, request: Request, principal: Principal = Depends(caller)) -> StreamingResponse:
        """Server-sent events: ``step`` (each stage), ``token`` (answer text, PII-redacted), then
        ``retract`` if the output guardrail rejects the streamed answer, and finally ``done`` with the
        same JSON as /query (or ``error``)."""
        turn = start_turn(body, request, principal)

        def sse(event: str, data: Any) -> str:
            return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"

        async def events() -> AsyncIterator[str]:
            with trace_for(turn) as run, track_usage() as usage:
                yield sse("start", {"trace_id": turn.trace_id, "thread_id": turn.thread_id})
                try:
                    async for event in answer_question_stream(turn.question, **pipeline_kwargs(turn)):
                        if event["type"] == "token":
                            yield sse("token", {"text": event["text"]})
                        elif event["type"] == "step":
                            yield sse("step", event["step"])
                        elif event["type"] == "outcome":
                            response = finish(turn, to_response(turn, event["outcome"]), usage, run)
                            if response.blocked_stage == "output":
                                yield sse("retract", {"message": response.answer})
                            yield sse("done", response.model_dump())
                except HTTPException as exc:
                    yield sse("error", {"status": exc.status_code, "detail": exc.detail})
                except Exception as exc:
                    err = gateway_error(turn, exc)
                    yield sse("error", {"status": err.status_code, "detail": err.detail})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
    async def feedback(body: FeedbackRequest, request: Request, principal: Principal = Depends(caller)) -> Response:
        store: FeedbackStore | None = getattr(request.app.state, "feedback", None)
        if store is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback is disabled")
        rating = 1 if body.rating == "up" else -1
        if not store.add_feedback(body.trace_id, principal.id, rating, body.comment):
            # Same response whether the trace doesn't exist or belongs to another caller.
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown trace_id")
        metrics.FEEDBACK.labels(rating=body.rating).inc()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)  # very verbose at INFO
logging.getLogger("httpx").setLevel(logging.WARNING)
app = create_app()
