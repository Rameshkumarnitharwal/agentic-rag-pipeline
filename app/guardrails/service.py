"""Guardrails gate: input checks before retrieval, output checks before returning an answer.

Input order (cheapest first): length limit -> regex injection/secret heuristics -> NeMo self-check rail.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from app.config import Settings, get_settings
from app.gateway import LLMGateway
from app.observability import annotate, traced

from .heuristics import detect_injection, detect_secret
from .llm import PortkeyLLMModel

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"

BLOCKED_INPUT_MESSAGE = (
    "I can't help with that request. I can only answer questions about the supported documentation topics."
)
BLOCKED_OUTPUT_MESSAGE = "I'm sorry, I can't provide that response."


class GuardStage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class GuardrailResult:
    allowed: bool
    stage: GuardStage
    rail: str | None = None  # which check decided (e.g. "heuristic:injection", "self check input")
    reason: str | None = None  # internal detail for logs/metrics — never shown to the user
    message: str | None = None  # user-facing message when blocked
    latency_ms: float = 0.0


def load_rails_config(settings: Settings) -> RailsConfig:
    yaml_content = (CONFIG_DIR / "config.yml").read_text()
    yaml_content = yaml_content.replace("__ALLOWED_TOPICS__", settings.guardrails_allowed_topics)
    return RailsConfig.from_content(yaml_content=yaml_content)


def _guard_outputs(result: GuardrailResult | None) -> dict:
    if result is None:
        return {}
    return {"allowed": result.allowed, "rail": result.rail, "reason": result.reason}


class GuardrailsService:
    def __init__(self, settings: Settings | None = None, gateway: LLMGateway | None = None):
        self.settings = settings or get_settings()
        self.enabled = self.settings.guardrails_enabled
        self._rails: LLMRails | None = None
        if self.enabled:
            gateway = gateway or LLMGateway(self.settings)
            self._rails = LLMRails(load_rails_config(self.settings), llm=PortkeyLLMModel(gateway))

    @traced("guardrails.input", process_outputs=_guard_outputs)
    async def check_input(self, user_input: str) -> GuardrailResult:
        started = time.perf_counter()
        result = await self._check_input(user_input)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        annotate(allowed=result.allowed, rail=result.rail)
        if not result.allowed:
            result.message = BLOCKED_INPUT_MESSAGE
            logger.warning("Input blocked by %s (%s)", result.rail, result.reason)
        return result

    def precheck_input(self, user_input: str) -> GuardrailResult | None:
        """The free, deterministic input checks only (no LLM). Returns a block result, or None to continue."""
        stage = GuardStage.INPUT
        if not self.enabled:
            return None
        if not user_input.strip():
            return GuardrailResult(False, stage, rail="heuristic:empty", reason="empty input")
        if len(user_input) > self.settings.guardrails_max_input_chars:
            return GuardrailResult(False, stage, rail="heuristic:length", reason=f"{len(user_input)} chars")
        if snippet := detect_injection(user_input):
            return GuardrailResult(False, stage, rail="heuristic:injection", reason=snippet)
        if detect_secret(user_input):
            return GuardrailResult(False, stage, rail="heuristic:secret", reason="secret-like token in input")
        return None

    async def _check_input(self, user_input: str) -> GuardrailResult:
        stage = GuardStage.INPUT
        if not self.enabled:
            return GuardrailResult(True, stage, rail="disabled")
        if blocked := self.precheck_input(user_input):
            return blocked
        return await self._run_rail([{"role": "user", "content": user_input}], RailType.INPUT, stage)

    @traced("guardrails.output", process_outputs=_guard_outputs)
    async def check_output(self, user_input: str, answer: str) -> GuardrailResult:
        started = time.perf_counter()
        stage = GuardStage.OUTPUT
        if not (self.enabled and self.settings.guardrails_output_enabled):
            result = GuardrailResult(True, stage, rail="disabled")
        elif detect_secret(answer):
            result = GuardrailResult(False, stage, rail="heuristic:secret", reason="secret-like token in answer")
        else:
            messages = [{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}]
            result = await self._run_rail(messages, RailType.OUTPUT, stage)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        annotate(allowed=result.allowed, rail=result.rail)
        if not result.allowed:
            result.message = BLOCKED_OUTPUT_MESSAGE
            logger.warning("Output blocked by %s (%s)", result.rail, result.reason)
        return result

    async def _run_rail(self, messages: list[dict], rail_type: RailType, stage: GuardStage) -> GuardrailResult:
        assert self._rails is not None
        try:
            rails_result = await self._rails.check_async(messages, rail_types=[rail_type])
        except Exception as exc:
            # NeMo already logs the full traceback; keep ours to one line.
            logger.error("%s rail errored (fail_open=%s): %s", rail_type.value, self.settings.guardrails_fail_open, exc)
            allowed = self.settings.guardrails_fail_open
            return GuardrailResult(allowed, stage, rail=f"error:{rail_type.value}", reason=str(exc)[:200])

        if rails_result.status == RailStatus.BLOCKED:
            return GuardrailResult(False, stage, rail=rails_result.rail, reason="nemo rail blocked")
        return GuardrailResult(True, stage, rail=rails_result.rail)
