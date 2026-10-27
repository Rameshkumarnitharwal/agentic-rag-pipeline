"""Evaluation core: run the dataset through the pipeline, score with RAGAS, add project-specific metrics.

Pipeline is run in-process (input guardrails -> LangGraph agent -> output guardrails) so the full
retrieved contexts are available for RAGAS. Each sample gets its own conversation thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.agents import RAGAgent
from app.agents.prompts import NO_CONTEXT_ANSWER
from app.config import Settings, get_settings
from app.gateway import track_usage
from app.guardrails import GuardrailsService
from app.pipeline import answer_question

from .retrieval_metrics import DEFAULT_KS, retrieval_metrics

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALS_DIR / "dataset.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "noise_sensitivity",
]
# Lower is better for noise_sensitivity (share of claims driven by irrelevant context).
LOWER_IS_BETTER = {"noise_sensitivity"}


@dataclass
class SampleResult:
    id: str
    category: str
    question: str
    reference: str | None
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    context_sources: list[str] = field(default_factory=list)
    context_datasets: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_behavior: str | None = None
    blocked: bool = False
    blocked_stage: str | None = None
    blocked_rail: str | None = None
    no_context: bool = False
    standalone_question: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    retrieval: dict[str, float] = field(default_factory=dict)  # Recall/Precision/NDCG/Hit@K, MRR
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None  # pipeline LLM cost for this question (judge calls excluded)
    steps: list[dict[str, Any]] = field(default_factory=list)  # per-node summaries + latency
    score_errors: dict[str, str] = field(default_factory=dict)  # metric -> error, when scoring failed
    passed: bool | None = None  # behaviour check for noise_probe / guardrail samples


@dataclass
class EvalReport:
    run_id: str
    started_at: str
    duration_s: float
    config: dict[str, Any]
    summary: dict[str, Any]
    samples: list[SampleResult]

    def to_json(self) -> dict[str, Any]:
        return {**{k: v for k, v in asdict(self).items() if k != "samples"}, "samples": [asdict(s) for s in self.samples]}


def load_dataset(path: Path = DEFAULT_DATASET, limit: int | None = None, categories: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    return rows[:limit] if limit else rows


# ---------- Pipeline execution ----------


class PipelineRunner:
    def __init__(self, settings: Settings, agent: RAGAgent | None = None, guardrails: GuardrailsService | None = None):
        self.settings = settings
        self.agent = agent or RAGAgent(settings)
        self.guardrails = guardrails or GuardrailsService(settings)

    async def run_sample(self, row: dict) -> SampleResult:
        result = SampleResult(
            id=row["id"],
            category=row["category"],
            question=row["question"],
            reference=row.get("reference"),
            expected_sources=row.get("expected_sources", []),
            expected_behavior=row.get("expected_behavior"),
        )
        thread_id = f"eval-{uuid.uuid4().hex}"
        try:
            # Earlier turns for multi-turn samples build conversation memory; not timed, costed or scored.
            for prior in row.get("history", []):
                await asyncio.to_thread(self.agent.run, prior, thread_id=thread_id)
        except Exception as exc:
            logger.exception("Sample %s history failed", row["id"])
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            return result

        started = time.perf_counter()
        with track_usage() as usage:
            try:
                await self._run_turn(row, result, thread_id)
            except Exception as exc:
                logger.exception("Sample %s failed", row["id"])
                result.error = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        result.llm_calls = len(usage.calls)
        result.prompt_tokens, result.completion_tokens = usage.prompt_tokens, usage.completion_tokens
        result.cost_usd = usage.cost_usd(self.settings.llm_prices)
        if result.category == "rag" and result.expected_sources and not result.error and not result.blocked:
            result.retrieval = retrieval_metrics(result.context_sources, result.expected_sources)
        return result

    async def _run_turn(self, row: dict, result: SampleResult, thread_id: str) -> None:
        # Same orchestration as the /query endpoint, so latency and cost reflect production.
        outcome = await answer_question(
            row["question"], agent=self.agent, guardrails=self.guardrails, thread_id=thread_id,
            trace_id=uuid.uuid4().hex, parallel_planning=self.settings.agent_parallel_planning,
        )
        if not outcome.input_check.allowed:
            result.blocked, result.blocked_stage, result.blocked_rail = True, "input", outcome.input_check.rail
            result.answer = outcome.input_check.message or ""
            return

        agent_result = outcome.agent_result
        result.answer = agent_result.answer
        result.standalone_question = agent_result.standalone_question
        result.model = agent_result.model
        result.citations = agent_result.citations
        # Retrieved chunks, best first — the ranked list used for Recall/Precision/NDCG@K.
        result.contexts = [c["text"] for c in agent_result.retrieved]
        result.context_sources = [c["metadata"].get("source", "") for c in agent_result.retrieved]
        result.context_datasets = [c["metadata"].get("dataset", "") for c in agent_result.retrieved]
        result.no_context = agent_result.answer == NO_CONTEXT_ANSWER
        result.steps = agent_result.steps
        if outcome.output_check is not None and not outcome.output_check.allowed:
            result.blocked, result.blocked_stage, result.blocked_rail = True, "output", outcome.output_check.rail


# ---------- RAGAS scoring ----------


Scorer = Callable[[str, SampleResult], Any]  # async (metric_name, sample) -> float | None


def build_ragas_scorer(settings: Settings) -> Scorer:
    """Create RAGAS metrics whose judge LLM and embeddings go through the Portkey gateway."""
    from openai import AsyncOpenAI
    from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import collections as rm

    if settings.portkey_api_key is None:
        raise ValueError("PORTKEY_API_KEY is required to run RAGAS metrics")
    client = AsyncOpenAI(
        api_key="portkey",  # unused: auth is the x-portkey-api-key header; the OpenAI key lives in Portkey
        base_url=settings.portkey_base_url or PORTKEY_GATEWAY_URL,
        default_headers=createHeaders(api_key=settings.portkey_api_key.get_secret_value()),
    )
    llm = llm_factory(settings.eval_judge_model, client=client, max_tokens=settings.eval_judge_max_tokens)
    embeddings = embedding_factory("openai", model=settings.eval_embedding_model, client=client)
    metrics = {
        "faithfulness": rm.Faithfulness(llm=llm),
        "answer_relevancy": rm.AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": rm.ContextPrecision(llm=llm),
        "context_recall": rm.ContextRecall(llm=llm),
        "factual_correctness": rm.FactualCorrectness(llm=llm),
        "noise_sensitivity": rm.NoiseSensitivity(llm=llm),
    }

    async def score(name: str, s: SampleResult) -> float | None:
        metric = metrics[name]
        q, a, ref, ctx = s.question, s.answer, s.reference or "", s.contexts
        if name == "faithfulness":
            r = await metric.ascore(user_input=q, response=a, retrieved_contexts=ctx)
        elif name == "answer_relevancy":
            r = await metric.ascore(user_input=q, response=a)
        elif name == "context_precision":
            r = await metric.ascore(user_input=q, reference=ref, retrieved_contexts=ctx)
        elif name == "context_recall":
            r = await metric.ascore(user_input=q, retrieved_contexts=ctx, reference=ref)
        elif name == "factual_correctness":
            r = await metric.ascore(response=a, reference=ref)
        else:  # noise_sensitivity
            r = await metric.ascore(user_input=q, response=a, reference=ref, retrieved_contexts=ctx)
        return None if r.value is None else float(r.value)

    return score


async def score_samples(samples: list[SampleResult], scorer: Scorer, concurrency: int, metrics: list[str]) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sample: SampleResult, name: str) -> None:
        async with semaphore:
            try:
                sample.scores[name] = await scorer(name, sample)
            except Exception as exc:
                # Some failures (timeouts, truncated structured output) have an empty message.
                logger.warning("RAGAS %s failed for %s: %s: %s", name, sample.id, type(exc).__name__, str(exc)[:300] or repr(exc))
                sample.scores[name] = None
                sample.score_errors[name] = f"{type(exc).__name__}: {str(exc)[:300]}"

    jobs = [
        one(s, name)
        for s in samples
        if s.category == "rag" and not s.error and not s.blocked and not s.no_context and s.contexts
        for name in metrics
    ]
    await asyncio.gather(*jobs)


# ---------- Summary ----------


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return round(values[k], 1)


def judge_behavior(s: SampleResult) -> bool | None:
    if s.category == "guardrail":
        return s.blocked
    if s.category == "noise_probe":
        # Pass if the system declined (blocked / "not in the docs") or at least cited nothing from noisy data.
        cited_noisy = any(c.get("dataset") == "noisy" for c in s.citations)
        return s.blocked or s.no_context or not cited_noisy
    return None


def summarize(samples: list[SampleResult], metrics: list[str]) -> dict[str, Any]:
    rag = [s for s in samples if s.category == "rag"]
    answered = [s for s in rag if not s.blocked and not s.error and not s.no_context]
    all_ctx_datasets = [d for s in rag for d in s.context_datasets]
    with_expected = [s for s in rag if s.expected_sources and not s.error and not s.blocked]

    summary: dict[str, Any] = {
        "samples": len(samples),
        "errors": sum(1 for s in samples if s.error),
        "metric_failures": sum(len(s.score_errors) for s in samples),
        "ragas": {name: _mean([s.scores.get(name) for s in answered]) for name in metrics},
        "rag": {
            "questions": len(rag),
            "answered": len(answered),
            "false_blocks": sum(1 for s in rag if s.blocked),  # legit questions wrongly blocked
            "no_context_answers": sum(1 for s in rag if s.no_context),
            # Share of retrieved chunks that came from the noisy dataset — the core True-vs-Noisy signal.
            "noisy_context_rate": round(all_ctx_datasets.count("noisy") / len(all_ctx_datasets), 4) if all_ctx_datasets else None,
            "noisy_citation_rate": _rate(
                sum(1 for s in rag for c in s.citations if c.get("dataset") == "noisy"),
                sum(len(s.citations) for s in rag),
            ),
            # Did retrieval surface at least one chunk from the document that holds the answer?
            "source_hit_rate": _rate(
                sum(1 for s in with_expected if set(s.expected_sources) & set(s.context_sources)), len(with_expected)
            ),
        },
    }
    for category in ("guardrail", "noise_probe"):
        group = [s for s in samples if s.category == category]
        summary[category] = {
            "total": len(group),
            "passed": sum(1 for s in group if s.passed),
            "pass_rate": _rate(sum(1 for s in group if s.passed), len(group)),
        }
    latencies = [s.latency_ms for s in samples if not s.error]
    summary["latency_ms"] = {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "max": max(latencies, default=None)}

    # Rank-based retrieval metrics over RAG questions (a question with nothing retrieved scores 0).
    scored = [s for s in rag if s.retrieval]
    keys = [f"{m}@{k}" for k in DEFAULT_KS for m in ("recall", "precision", "ndcg", "hit")] + ["mrr"]
    summary["retrieval"] = {k: _mean([s.retrieval[k] for s in scored]) for k in keys} if scored else {}
    summary["retrieval"]["questions"] = len(scored)

    # Pipeline LLM cost per question (guardrails + planner + responder; RAGAS judge excluded).
    costs = [s.cost_usd for s in samples if s.cost_usd is not None and not s.error]
    answered_costs = [s.cost_usd for s in answered if s.cost_usd is not None]
    summary["cost_usd"] = {
        "per_query_mean": round(statistics.fmean(costs), 6) if costs else None,
        "per_query_p95": _percentile_raw(costs, 95),
        "per_answered_query_mean": round(statistics.fmean(answered_costs), 6) if answered_costs else None,
        "total": round(sum(costs), 6),
        "unpriced_samples": sum(1 for s in samples if s.cost_usd is None and s.llm_calls),
        "tokens": {"prompt": sum(s.prompt_tokens for s in samples), "completion": sum(s.completion_tokens for s in samples)},
    }
    return summary


def _percentile_raw(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))], 6)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# ---------- Entry point ----------


async def run_evaluation(
    settings: Settings | None = None,
    dataset_path: Path = DEFAULT_DATASET,
    limit: int | None = None,
    categories: list[str] | None = None,
    metrics: list[str] | None = None,
    skip_ragas: bool = False,
    runner: PipelineRunner | None = None,
    scorer: Scorer | None = None,
    progress: Callable[[int, int, SampleResult], None] | None = None,
) -> EvalReport:
    settings = settings or get_settings()
    metrics = metrics or RAGAS_METRICS
    rows = load_dataset(dataset_path, limit, categories)
    runner = runner or PipelineRunner(settings)
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    samples: list[SampleResult] = []
    for i, row in enumerate(rows, start=1):
        sample = await runner.run_sample(row)
        sample.passed = judge_behavior(sample)
        samples.append(sample)
        if progress:
            progress(i, len(rows), sample)

    if not skip_ragas:
        await score_samples(samples, scorer or build_ragas_scorer(settings), settings.eval_concurrency, metrics)

    return EvalReport(
        run_id=started_at.strftime("%Y%m%d-%H%M%S"),
        started_at=started_at.isoformat(),
        duration_s=round(time.perf_counter() - t0, 1),
        config={
            "dataset": str(dataset_path.name),
            "llm_models": settings.llm_models,
            "judge_model": None if skip_ragas else settings.eval_judge_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "rerank_min_score": settings.rerank_min_score,
            "guardrails_enabled": settings.guardrails_enabled,
            "metrics": [] if skip_ragas else metrics,
        },
        summary=summarize(samples, [] if skip_ragas else metrics),
        samples=samples,
    )


def save_report(report: EvalReport, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"eval-{report.run_id}.json"
    path.write_text(json.dumps(report.to_json(), indent=2, ensure_ascii=False))
    return path


def list_reports(results_dir: Path = RESULTS_DIR) -> list[Path]:
    return sorted(results_dir.glob("eval-*.json"), reverse=True) if results_dir.exists() else []
