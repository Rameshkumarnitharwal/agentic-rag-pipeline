"""Tracing with Pydantic Logfire (OpenTelemetry) and LangSmith.

Both backends are optional and independent:
- Logfire gets a span tree per request: HTTP -> rag_query -> guardrails / agent nodes -> llm / retrieval
  sub-steps, plus auto-instrumented outbound HTTP (Jina, Qdrant, Portkey).
- LangSmith gets the same pipeline as runs: LangGraph traces nodes natively; ``traced`` adds llm,
  retriever and guardrail runs that nest under them.

Code calls ``traced`` / ``span`` / ``annotate`` unconditionally; they are no-ops when tracing is off.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from opentelemetry import trace as otel_trace

from app.config import Settings

logger = logging.getLogger(__name__)

_logfire_enabled = False


def setup_observability(settings: Settings, app: Any | None = None) -> dict[str, bool]:
    """Configure tracing backends from settings. Call once, before the FastAPI app starts serving."""
    global _logfire_enabled

    # LangSmith reads its config from env vars (also used by LangGraph's built-in tracing).
    langsmith_on = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    os.environ["LANGSMITH_TRACING"] = "true" if langsmith_on else "false"
    if langsmith_on:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    logfire_on = bool(settings.logfire_token or settings.logfire_console)
    if logfire_on and not _logfire_enabled:
        import logfire

        logfire.configure(
            token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
            send_to_logfire="if-token-present",
            service_name=settings.logfire_service_name,
            environment=settings.logfire_environment,
            console=None if settings.logfire_console else False,
            # Default scrubbing redacts values whose keys look like secrets (api_key, password, token, ...).
        )
        logfire.instrument_httpx()  # Jina, Qdrant and Portkey calls appear as child spans
        _logfire_enabled = True
    if logfire_on and app is not None:
        import logfire

        logfire.instrument_fastapi(app, excluded_urls="/metrics,/health,/ready")

    logger.info("Tracing: logfire=%s langsmith=%s", logfire_on, langsmith_on)
    return {"logfire": logfire_on, "langsmith": langsmith_on}


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a Logfire span (no-op when Logfire is off). Attribute values must be JSON-serialisable."""
    if not _logfire_enabled:
        yield None
        return
    import logfire

    with logfire.span(name, _span_name=name, **attributes) as s:
        yield s


def annotate(**attributes: Any) -> None:
    """Attach attributes to the current Logfire span and metadata to the current LangSmith run."""
    clean = {k: v for k, v in attributes.items() if v is not None}
    current = otel_trace.get_current_span()
    if current.is_recording():
        for key, value in clean.items():
            current.set_attribute(key, value if isinstance(value, (str, bool, int, float)) else str(value))
    run = get_current_run_tree()
    if run is not None:
        run.metadata.update(clean)


def traced(
    name: str,
    run_type: str = "chain",
    process_inputs: Callable[[dict], dict] | None = None,
    process_outputs: Callable[[Any], dict] | None = None,
):
    """Decorator: a LangSmith run of ``run_type`` inside a Logfire span, for sync or async functions.

    ``self`` is always dropped from the recorded inputs.
    """

    def _inputs(inputs: dict) -> dict:
        inputs = {k: v for k, v in inputs.items() if k != "self"}
        return process_inputs(inputs) if process_inputs else inputs

    def decorator(fn: Callable) -> Callable:
        ls_fn = traceable(name=name, run_type=run_type, process_inputs=_inputs, process_outputs=process_outputs)(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(name):
                    return await ls_fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return ls_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def root_trace(name: str, inputs: dict[str, Any], metadata: dict[str, Any]) -> Iterator[Any]:
    """Top-level span + LangSmith run for one request; everything called inside nests under it.

    Yields the LangSmith run tree (call ``.end(outputs=...)`` on it) — or a stub when LangSmith is off.
    """
    from langsmith import trace as ls_trace

    with span(name, **metadata):
        with ls_trace(name, run_type="chain", inputs=inputs, metadata=metadata, tags=["api"]) as run:
            yield run
