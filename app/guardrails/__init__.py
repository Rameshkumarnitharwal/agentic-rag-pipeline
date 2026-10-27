"""NeMo Guardrails input/output filtering."""

import os

# NeMo Guardrails sends usage telemetry to NVIDIA by default; opt out unless explicitly enabled.
os.environ.setdefault("NEMO_GUARDRAILS_NO_USAGE_STATS", "1")

from .service import GuardrailResult, GuardrailsService, GuardStage

__all__ = ["GuardStage", "GuardrailResult", "GuardrailsService"]
