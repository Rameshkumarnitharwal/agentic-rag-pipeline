"""Headless evaluation: run the dataset, score with RAGAS, print a summary, save JSON.

Usage:
    python -m evals.run_evals                         # full run (needs PORTKEY_API_KEY, JINA_API_KEY, Qdrant)
    python -m evals.run_evals --limit 5 --skip-ragas  # quick smoke test, pipeline only
    python -m evals.run_evals --fail-under faithfulness=0.8 --fail-under recall@5=0.9 \
        --fail-under cost_per_query=0.02 --fail-under latency_p95_ms=20000                    # CI gate
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from evals.runner import DEFAULT_DATASET, LOWER_IS_BETTER, RAGAS_METRICS, EvalReport, run_evaluation, save_report


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_summary(report: EvalReport) -> None:
    s = report.summary
    print(f"\nEvaluation {report.run_id}  ({s['samples']} samples, {report.duration_s}s, errors: {s['errors']}, "
          f"failed metric scores: {s.get('metric_failures', 0)})")
    if s["ragas"]:
        print("\nRAGAS (mean over answered RAG questions)")
        for name, value in s["ragas"].items():
            note = "  (lower is better)" if name in LOWER_IS_BETTER else ""
            print(f"  {name:<22} {_fmt(value)}{note}")
    r = s["rag"]
    print("\nRetrieval / True-vs-Noisy")
    print(f"  answered               {r['answered']}/{r['questions']}  (false blocks: {r['false_blocks']}, no-context: {r['no_context_answers']})")
    print(f"  source_hit_rate        {_fmt(r['source_hit_rate'])}")
    print(f"  noisy_context_rate     {_fmt(r['noisy_context_rate'])}  (lower is better)")
    print(f"  noisy_citation_rate    {_fmt(r['noisy_citation_rate'])}  (lower is better)")
    ret = s.get("retrieval") or {}
    if ret.get("questions"):
        print(f"\nRetrieval ranking (vs expected sources, {ret['questions']} questions)")
        print(f"  {'':12} {'@1':>7} {'@3':>7} {'@5':>7}")
        for metric in ("recall", "precision", "ndcg", "hit"):
            print(f"  {metric:12} " + " ".join(f"{_fmt(ret.get(f'{metric}@{k}')):>7}" for k in (1, 3, 5)))
        print(f"  {'mrr':12} {_fmt(ret.get('mrr')):>7}")
    print("\nBehaviour")
    for cat in ("guardrail", "noise_probe"):
        g = s[cat]
        print(f"  {cat + '_pass_rate':<22} {_fmt(g['pass_rate'])}  ({g['passed']}/{g['total']})")
    lat = s["latency_ms"]
    print(f"\nLatency  p50 {lat['p50']:.0f} ms  p95 {lat['p95']:.0f} ms  max {lat['max']:.0f} ms" if lat["p50"] is not None else "\nLatency  —")
    cost = s.get("cost_usd") or {}
    if cost.get("per_query_mean") is not None:
        tok = cost["tokens"]
        print(f"LLM cost per query  mean ${cost['per_query_mean']:.4f}  p95 ${cost['per_query_p95']:.4f}  "
              f"(answered questions ${cost['per_answered_query_mean'] or 0:.4f})  total ${cost['total']:.4f}  "
              f"tokens {tok['prompt']:,} in / {tok['completion']:,} out  — RAGAS judge not included")
        if cost["unpriced_samples"]:
            print(f"  ({cost['unpriced_samples']} samples used a model missing from LLM_PRICES and were not costed)")

    failed = [x for x in report.samples if x.passed is False or x.error or (x.category == "rag" and x.blocked)]
    if failed:
        print("\nNeeds attention:")
        for x in failed:
            why = x.error or (f"blocked by {x.blocked_rail}" if x.blocked else "answered instead of refusing/blocking")
            print(f"  [{x.id}] {why}")


def flat_metrics(report: EvalReport) -> dict[str, float | None]:
    s = report.summary
    flat = dict(s["ragas"])
    flat.update({k: v for k, v in s["rag"].items() if isinstance(v, float)})
    flat.update({k: v for k, v in (s.get("retrieval") or {}).items() if k != "questions"})
    cost = s.get("cost_usd") or {}
    flat["cost_per_query"] = cost.get("per_query_mean")
    flat["cost_per_query_p95"] = cost.get("per_query_p95")
    flat["latency_p95_ms"] = s["latency_ms"]["p95"]
    flat["guardrail_pass_rate"] = s["guardrail"]["pass_rate"]
    flat["noise_probe_pass_rate"] = s["noise_probe"]["pass_rate"]
    return flat


def check_thresholds(report: EvalReport, thresholds: list[str]) -> list[str]:
    """Return failure messages for ``name=value`` thresholds (max for lower-is-better metrics)."""
    flat = flat_metrics(report)
    failures = []
    for item in thresholds:
        name, _, raw = item.partition("=")
        limit = float(raw)
        value = flat.get(name)
        lower_better = name in LOWER_IS_BETTER or name.startswith(("noisy_", "cost_", "latency_"))
        if value is None:
            failures.append(f"{name}: no value")
        elif (value > limit) if lower_better else (value < limit):
            failures.append(f"{name}: {value:.3f} {'>' if lower_better else '<'} {limit}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", choices=["rag", "noise_probe", "guardrail"], help="Repeatable")
    parser.add_argument("--metric", action="append", choices=RAGAS_METRICS, help="Repeatable; default: all six")
    parser.add_argument("--skip-ragas", action="store_true", help="Only run the pipeline and behaviour checks")
    parser.add_argument("--fail-under", action="append", default=[], metavar="METRIC=VALUE",
                        help="Exit 1 if a metric misses its threshold (a maximum for noise/noisy/cost/latency metrics)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    def progress(i: int, total: int, sample) -> None:
        state = "ERROR" if sample.error else "blocked" if sample.blocked else "no-context" if sample.no_context else "answered"
        print(f"[{i}/{total}] {sample.id:<10} {state:<10} {sample.latency_ms:>8.0f} ms", flush=True)

    report = asyncio.run(
        run_evaluation(
            dataset_path=args.dataset, limit=args.limit, categories=args.category,
            metrics=args.metric, skip_ragas=args.skip_ragas, progress=progress,
        )
    )
    print_summary(report)
    if not args.no_save:
        print(f"\nSaved {save_report(report)}")

    failures = check_thresholds(report, args.fail_under)
    if failures:
        print("\nThreshold failures:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
