"""Request orchestration shared by the API and the eval runner, so evals measure the production path.

    free pattern checks ──blocked──▶ return
          │
          ├── NeMo input rail (LLM) ─┐   run concurrently; if the rail blocks,
          └── planner (LLM) ─────────┘   the plan is discarded (no side effects)
          │
    agent (retrieve → respond, using the precomputed plan) ──▶ output guardrails
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.agents import AgentResult, RAGAgent
from app.agents.prompts import NO_CONTEXT_ANSWER
from app.guardrails import GuardrailResult, GuardrailsService

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutcome:
    input_check: GuardrailResult
    agent_result: AgentResult | None = None
    output_check: GuardrailResult | None = None

    @property
    def blocked_stage(self) -> str | None:
        if not self.input_check.allowed:
            return "input"
        if self.output_check is not None and not self.output_check.allowed:
            return "output"
        return None


async def answer_question(
    question: str,
    *,
    agent: RAGAgent,
    guardrails: GuardrailsService,
    thread_id: str,
    trace_id: str,
    filters: dict[str, Any] | None = None,
    parallel_planning: bool = True,
) -> PipelineOutcome:
    """Guardrails → agent → guardrails, overlapping the input rail with planning.

    Raises whatever the agent raises (e.g. ``LLMGatewayError``); guardrail errors come back as
    results with an ``error:`` rail, as from ``GuardrailsService``.
    """
    if guardrails.precheck_input(question) is not None:
        # Cheap rejections (injection patterns, secrets, length) never reach an LLM; check_input
        # re-runs the same free checks and returns at once with the standard message and logging.
        return PipelineOutcome(input_check=await guardrails.check_input(question))

    plan_task = None
    if parallel_planning:
        # Speculative: the plan is read-only (no retrieval, no memory writes) and dropped if blocked.
        plan_task = asyncio.create_task(asyncio.to_thread(agent.plan, question, thread_id, trace_id))

    input_check = await guardrails.check_input(question)
    if not input_check.allowed:
        if plan_task is not None:
            plan_task.add_done_callback(_swallow)  # let it finish in the background; result unused
        return PipelineOutcome(input_check=input_check)

    plan = None
    if plan_task is not None:
        try:
            plan = await plan_task
        except Exception:
            logger.warning("Speculative planning failed; the agent will plan itself", exc_info=True)

    agent_result = await asyncio.to_thread(
        agent.run, question, thread_id=thread_id, trace_id=trace_id, filters=filters, plan=plan
    )
    outcome = PipelineOutcome(input_check=input_check, agent_result=agent_result)
    if agent_result.answer != NO_CONTEXT_ANSWER:  # the fixed no-context reply is safe by construction
        outcome.output_check = await guardrails.check_output(question, agent_result.answer)
    return outcome


def _swallow(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # mark retrieved so asyncio doesn't warn
