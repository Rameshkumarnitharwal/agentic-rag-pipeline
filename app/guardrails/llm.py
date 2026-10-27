"""Adapter that lets NeMo Guardrails call the LLM through the Portkey gateway."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from nemoguardrails import ChatMessage, LLMResponse, LLMResponseChunk

from app.gateway import LLMGateway
from app.gateway.client import FAST


def _to_messages(prompt: str | list[ChatMessage]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages = []
    for message in prompt:
        content = message.content
        if isinstance(content, list):  # multimodal parts -> keep the text only
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        messages.append({"role": getattr(message.role, "value", message.role), "content": content or ""})
    return messages


class PortkeyLLMModel:
    """Implements NeMo's ``LLMModel`` protocol on top of :class:`LLMGateway`.

    Rail checks are short Yes/No classifications, so calls use temperature 0 and a small token cap.
    """

    def __init__(self, gateway: LLMGateway, max_tokens: int = 16):
        self._gateway = gateway
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._gateway.settings.llm_model_list[0]

    @property
    def provider_name(self) -> str | None:
        return "portkey"

    @property
    def provider_url(self) -> str | None:
        return self._gateway.settings.portkey_base_url

    async def generate_async(
        self, prompt: str | list[ChatMessage], *, stop: list[str] | None = None, **kwargs: Any
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "temperature": 0.0, "max_tokens": self._max_tokens, "metadata": {"node": "guardrails"}, "tier": FAST,
        }
        if stop:
            params["stop"] = stop
        # LLMGateway is synchronous; run it in a worker thread so the event loop isn't blocked.
        response = await asyncio.to_thread(self._gateway.chat, _to_messages(prompt), **params)
        return LLMResponse(content=response.content, model=response.model, finish_reason="stop")

    async def stream_async(
        self, prompt: str | list[ChatMessage], *, stop: list[str] | None = None, **kwargs: Any
    ) -> AsyncIterator[LLMResponseChunk]:
        response = await self.generate_async(prompt, stop=stop, **kwargs)
        yield LLMResponseChunk(delta_content=response.content)
        yield LLMResponseChunk(finish_reason="stop")
