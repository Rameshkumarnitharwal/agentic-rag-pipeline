"""Per-request LLM token and cost accounting.

Wrap the work for one question in ``track_usage()``; every ``LLMGateway.chat`` call inside it (across
threads started with copied contexts, e.g. LangGraph nodes and ``asyncio.to_thread``) is recorded.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

_current: contextvars.ContextVar["UsageTracker | None"] = contextvars.ContextVar("llm_usage", default=None)


@dataclass
class LLMCall:
    model: str
    node: str | None
    prompt_tokens: int
    completion_tokens: int


@dataclass
class UsageTracker:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def cost_usd(self, prices: dict[str, tuple[float, float]]) -> float | None:
        """Total cost from per-1M-token (input, output) prices; None if any model has no price."""
        total = 0.0
        for call in self.calls:
            price = price_for(call.model, prices)
            if price is None:
                return None
            total += call.prompt_tokens * price[0] / 1e6 + call.completion_tokens * price[1] / 1e6
        return round(total, 6)


def price_for(model: str, prices: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Longest-prefix match, so "gpt-4o-mini-2024-07-18" uses "gpt-4o-mini", not "gpt-4o"."""
    name = model.split("/")[-1].lower()
    matches = [key for key in prices if name.startswith(key.lower())]
    return prices[max(matches, key=len)] if matches else None


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    tracker = UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)


def record_usage(model: str, node: str | None, prompt_tokens: int, completion_tokens: int) -> None:
    tracker = _current.get()
    if tracker is not None:
        tracker.calls.append(LLMCall(model, node, prompt_tokens, completion_tokens))
