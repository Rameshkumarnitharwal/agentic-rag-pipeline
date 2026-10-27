"""Portkey LLM gateway: every LLM call goes through here.

Routing (primary model + fallbacks, retries) is expressed as a Portkey config built from settings,
so changing models or adding a provider only needs ``LLM_MODELS`` changed in ``.env``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

from portkey_ai import Portkey
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.observability import annotate, span, traced

from .usage import record_usage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMGatewayError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    model: str  # model that actually served the request (after any fallback)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    trace_id: str = ""


def build_portkey_config(settings: Settings) -> dict[str, Any]:
    models = settings.llm_model_list
    if not models:
        raise ValueError("LLM_MODELS is empty")
    retry = {"attempts": settings.llm_retry_attempts}
    targets = [{"override_params": {"model": model}, "retry": retry} for model in models]
    if len(targets) == 1:
        return {"strategy": {"mode": "single"}, "targets": targets}

    strategy: dict[str, Any] = {"mode": "fallback"}
    if settings.llm_fallback_status_code_list:
        strategy["on_status_codes"] = settings.llm_fallback_status_code_list
    return {"strategy": strategy, "targets": targets}


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences and surrounding prose."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _JSON_BLOCK.findall(text)]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start, end = text.find(open_char), text.rfind(close_char)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in model output")


def _llm_run_inputs(inputs: dict) -> dict:
    return {"messages": inputs.get("messages"), "metadata": inputs.get("metadata")}


def _llm_run_outputs(response: LLMResponse | None) -> dict:
    if response is None:  # the call raised; LangSmith records the error itself
        return {}
    # OpenAI-style output + usage_metadata so LangSmith renders the message and counts tokens.
    return {
        "choices": [{"message": {"role": "assistant", "content": response.content}}],
        "usage_metadata": {
            "input_tokens": response.usage.get("prompt_tokens", 0),
            "output_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
        },
    }


FAST = "fast"  # tier for short, latency-sensitive calls (guardrail checks, planning)


class LLMGateway:
    """Two tiers: the default tier routes through the Portkey config (LLM_MODELS / PORTKEY_CONFIG_ID,
    with fallback); the ``fast`` tier calls LLM_FAST_MODEL directly and, if that fails, retries once
    on the default tier, so it keeps a fallback without needing a second saved config.
    """

    def __init__(self, settings: Settings | None = None, client: Portkey | None = None,
                 fast_client: Portkey | None = None):
        self.settings = settings or get_settings()
        if client is None:
            if self.settings.portkey_api_key is None:
                raise ValueError("PORTKEY_API_KEY is not set")
            client = self._portkey(
                # A saved config (pc-...) if configured, else an inline one built from LLM_MODELS.
                config=self.settings.portkey_config_id or build_portkey_config(self.settings)
            )
            if fast_client is None and self.settings.llm_fast_model:
                fast_client = self._portkey(config=None)  # a config would override the model
        self._client = client
        self._fast_client = fast_client
        # Model ids are set per-target in the config; this is only a placeholder for the SDK.
        self._default_model = self.settings.llm_model_list[0]

    def _portkey(self, config: Any) -> Portkey:
        kwargs: dict[str, Any] = {
            "api_key": self.settings.portkey_api_key.get_secret_value(),
            "request_timeout": self.settings.llm_timeout_seconds * 1000,  # Portkey expects milliseconds
        }
        if config is not None:
            kwargs["config"] = config
        if self.settings.portkey_base_url:
            kwargs["base_url"] = self.settings.portkey_base_url
        return Portkey(**kwargs)

    @traced("llm.chat", run_type="llm", process_inputs=_llm_run_inputs, process_outputs=_llm_run_outputs)
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, str] | None = None,
        tier: str | None = None,
        **params: Any,
    ) -> LLMResponse:
        """Send a chat completion through Portkey.

        ``trace_id`` groups related calls (e.g. one /query request) in Portkey logs;
        ``metadata`` values must be strings (Portkey requirement), e.g. ``{"node": "planner"}``.
        ``tier="fast"`` uses LLM_FAST_MODEL (falls back to the default tier on failure).
        """
        trace_id = trace_id or uuid.uuid4().hex
        request = {
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            **params,
        }
        started = time.perf_counter()
        completion = None
        if tier == FAST and self._fast_client is not None:
            fast = self._fast_client.with_options(trace_id=trace_id, metadata={**(metadata or {}), "tier": FAST})
            try:
                completion = fast.chat.completions.create(model=self.settings.llm_fast_model, **request)
            except Exception as exc:
                logger.warning("Fast-tier model %s failed, using default tier: %s", self.settings.llm_fast_model, exc)
        if completion is None:
            client = self._client.with_options(trace_id=trace_id, metadata=metadata or {})
            try:
                completion = client.chat.completions.create(model=self._default_model, **request)
            except Exception as exc:
                raise LLMGatewayError(f"LLM call failed on all targets: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        if not completion.choices:
            raise LLMGatewayError("LLM returned no choices")
        content = completion.choices[0].message.content or ""
        usage = {}
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens or 0,
                "completion_tokens": completion.usage.completion_tokens or 0,
                "total_tokens": completion.usage.total_tokens or 0,
            }
        logger.debug("LLM %s answered in %.0f ms (%s)", completion.model, latency_ms, usage)
        record_usage(completion.model or "", (metadata or {}).get("node"),
                     usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        annotate(
            ls_provider="openai", ls_model_name=completion.model, node=(metadata or {}).get("node"),
            latency_ms=latency_ms, total_tokens=usage.get("total_tokens"), portkey_trace_id=trace_id,
        )
        return LLMResponse(content=content, model=completion.model or "", usage=usage, latency_ms=latency_ms, trace_id=trace_id)

    def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_id: str | None = None,
        metadata: dict[str, str] | None = None,
        on_complete: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        """Stream a chat completion (default tier, so Portkey's fallback config applies) as text deltas.

        Usage is recorded when the stream finishes. A failure before the first token raises
        ``LLMGatewayError``; after that, the partial answer stands (it can't be un-sent).
        """
        trace_id = trace_id or uuid.uuid4().hex
        client = self._client.with_options(trace_id=trace_id, metadata=metadata or {})
        started = time.perf_counter()
        model, usage, parts = "", None, []
        with span("llm.chat_stream", node=(metadata or {}).get("node")):
            try:
                stream = client.chat.completions.create(
                    model=self._default_model,
                    messages=messages,
                    temperature=self.settings.llm_temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            except Exception as exc:
                raise LLMGatewayError(f"LLM call failed on all targets: {exc}") from exc
            for chunk in stream:
                model = getattr(chunk, "model", None) or model
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    parts.append(delta)
                    yield delta
            prompt_tokens = (usage.prompt_tokens or 0) if usage else 0
            completion_tokens = (usage.completion_tokens or 0) if usage else 0
            if usage is None:
                logger.debug("Streamed response had no usage block; tokens not counted")
            record_usage(model, (metadata or {}).get("node"), prompt_tokens, completion_tokens)
            if on_complete is not None:
                on_complete(model)  # lets callers learn which model actually answered (after fallback)
            annotate(ls_model_name=model, node=(metadata or {}).get("node"), total_tokens=prompt_tokens + completion_tokens,
                     latency_ms=round((time.perf_counter() - started) * 1000, 1), streamed=True)

    @traced("llm.chat_json", process_inputs=lambda i: {"messages": i.get("messages"), "schema": i["schema"].__name__},
            process_outputs=lambda r: {"parsed": r[0].model_dump(), "model": r[1].model} if r else {})
    def chat_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        max_attempts: int = 2,
        **kwargs: Any,
    ) -> tuple[T, LLMResponse]:
        """Chat and parse the reply into ``schema``.

        Uses prompt instructions rather than provider-specific JSON modes so it behaves the same
        on every fallback target. On invalid output, the error is fed back for one more attempt.
        """
        instruction = (
            "Respond with ONLY a JSON object (no prose, no code fences) matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        attempt_messages = [*messages, {"role": "system", "content": instruction}]
        last_error: Exception | None = None
        for _ in range(max_attempts):
            response = self.chat(attempt_messages, **kwargs)
            try:
                return schema.model_validate(extract_json(response.content)), response
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning("Invalid structured output from %s: %s", response.model, exc)
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": f"That was not valid. Error: {exc}. Reply with the corrected JSON only."},
                ]
        raise LLMGatewayError(f"Could not get valid {schema.__name__} JSON from the LLM") from last_error
