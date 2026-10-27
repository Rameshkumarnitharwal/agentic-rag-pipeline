"""FastAPI entrypoint: guardrails gate + synchronous /query running the LangGraph agent.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app import metrics
from app.agents import RAGAgent
from app.agents.prompts import NO_CONTEXT_ANSWER
from app.config import Settings, get_settings
from app.gateway import LLMGatewayError, track_usage
from app.guardrails import GuardrailResult, GuardrailsService
from app.observability import annotate, root_trace, setup_observability
from app.pipeline import answer_question
from app.security import RateLimiter, authenticate, build_rate_limiter

logger = logging.getLogger("app")


# ---------- Schemas ----------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question")
    thread_id: str | None = Field(default=None, max_length=128, description="Conversation id for follow-ups")
    filters: dict[str, Any] | None = Field(default=None, description='Exact-match metadata filters, e.g. {"dataset": "true"}')


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

    def caller(request: Request) -> str:
        try:
            principal = authenticate(request, settings)
        except HTTPException:
            metrics.QUERY_REQUESTS.labels(outcome="unauthorized").inc()
            raise
        limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
        if limiter is not None:
            allowed, retry_after = limiter.hit(principal)
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

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request, principal: str = Depends(caller)) -> QueryResponse:
        state = request.app.state
        if getattr(state, "init_error", None):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not initialised; see /ready")
        if len(body.question) > settings.max_question_chars:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"Question exceeds {settings.max_question_chars} characters"
            )

        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        thread_id = body.thread_id or uuid.uuid4().hex
        # Namespace conversation memory by caller so one client can't read another's thread.
        internal_thread = f"{principal}:{thread_id}"
        question = body.question.strip()

        def respond(**kwargs: Any) -> QueryResponse:
            elapsed = time.perf_counter() - started
            metrics.QUERY_LATENCY.observe(elapsed)
            return QueryResponse(thread_id=thread_id, trace_id=trace_id, latency_ms=round(elapsed * 1000, 1), **kwargs)

        async def pipeline() -> QueryResponse:
            # Guardrails → agent → guardrails, with the input rail and planning overlapped (app/pipeline.py).
            try:
                outcome = await answer_question(
                    question, agent=state.agent, guardrails=state.guardrails, thread_id=internal_thread,
                    trace_id=trace_id, filters=body.filters, parallel_planning=settings.agent_parallel_planning,
                )
            except LLMGatewayError as exc:
                metrics.QUERY_REQUESTS.labels(outcome="error").inc()
                logger.error("LLM gateway failure (trace %s): %s", trace_id, exc)
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The language model is unavailable. Please retry.") from exc
            except Exception as exc:
                metrics.QUERY_REQUESTS.labels(outcome="error").inc()
                logger.exception("Agent failure (trace %s)", trace_id)
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error") from exc

            input_check = outcome.input_check
            _record_guard(input_check)
            steps = [_guard_step(input_check)]
            if not input_check.allowed and (input_check.rail or "").startswith("error:"):
                # Fail-closed because the rail itself couldn't run (LLM outage) — that's not a policy block.
                metrics.QUERY_REQUESTS.labels(outcome="error").inc()
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Safety checks are unavailable. Please retry shortly.")
            if not input_check.allowed:
                metrics.QUERY_REQUESTS.labels(outcome="blocked_input").inc()
                return respond(answer=input_check.message or "", steps=steps, blocked=True, blocked_stage="input")

            result = outcome.agent_result
            steps += [Step(**s) for s in result.steps]
            for s in result.steps:
                metrics.NODE_LATENCY.labels(node=s["node"]).observe(s.get("latency_ms", 0) / 1000)
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
                    return respond(
                        answer=output_check.message or "", steps=steps, blocked=True, blocked_stage="output",
                        standalone_question=result.standalone_question, model=result.model,
                    )

            metrics.QUERY_REQUESTS.labels(outcome="no_context" if result.answer == NO_CONTEXT_ANSWER else "answered").inc()
            for citation in result.citations:
                metrics.CITED_DATASET.labels(dataset=citation.get("dataset") or "unknown").inc()
            if result.model:
                metrics.LLM_MODEL_USED.labels(model=result.model).inc()

            return respond(
                answer=result.answer,
                citations=[Citation(**c) for c in result.citations],
                steps=steps,
                standalone_question=result.standalone_question,
                model=result.model or None,
            )

        # One root trace per request: guardrails, agent nodes, LLM and retrieval calls nest under it.
        with root_trace(
            "rag_query",
            inputs={"question": question, "filters": body.filters},
            metadata={"trace_id": trace_id, "thread_id": thread_id, "principal": principal},
        ) as run, track_usage() as usage:
            response = await pipeline()
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
        return response

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)  # very verbose at INFO
logging.getLogger("httpx").setLevel(logging.WARNING)
app = create_app()
