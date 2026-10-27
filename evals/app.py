"""Streamlit evaluation demo — 3 tabs: Run evaluation · Results explorer · True vs Noisy probe.

Run:  streamlit run evals/app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `streamlit run evals/app.py`

from app.config import get_settings  # noqa: E402
from evals.runner import (  # noqa: E402
    DEFAULT_DATASET,
    LOWER_IS_BETTER,
    RAGAS_METRICS,
    list_reports,
    load_dataset,
    run_evaluation,
    save_report,
)

st.set_page_config(page_title="RAG Evaluation", page_icon=":material/analytics:", layout="wide")

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "factual_correctness": "Factual correctness",
    "noise_sensitivity": "Noise sensitivity ↓",
}


def fmt(value, pct: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.0%}" if pct else f"{value:.3f}"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def render_summary(summary: dict, previous: dict | None = None) -> None:
    def delta(section: str, key: str):
        if not previous:
            return None
        old, new = previous.get(section, {}).get(key), summary.get(section, {}).get(key)
        return None if old is None or new is None else round(new - old, 3)

    if summary.get("ragas"):
        st.subheader("RAGAS metrics")
        cols = st.columns(len(summary["ragas"]))
        for col, (name, value) in zip(cols, summary["ragas"].items()):
            col.metric(METRIC_LABELS.get(name, name), fmt(value), delta=delta("ragas", name),
                       delta_color="inverse" if name in LOWER_IS_BETTER else "normal", border=True)

    st.subheader("True vs Noisy & behaviour")
    rag, guard, noise = summary["rag"], summary["guardrail"], summary["noise_probe"]
    cols = st.columns(5)
    cols[0].metric("Source hit rate", fmt(rag["source_hit_rate"], pct=True), delta=delta("rag", "source_hit_rate"), border=True,
                   help="RAG questions where the document holding the answer was retrieved")
    cols[1].metric("Noisy context rate ↓", fmt(rag["noisy_context_rate"], pct=True), delta=delta("rag", "noisy_context_rate"),
                   delta_color="inverse", border=True, help="Share of retrieved chunks that came from noisy data")
    cols[2].metric("Noisy citation rate ↓", fmt(rag["noisy_citation_rate"], pct=True), delta=delta("rag", "noisy_citation_rate"),
                   delta_color="inverse", border=True)
    cols[3].metric("Guardrail pass rate", fmt(guard["pass_rate"], pct=True), border=True,
                   help=f"{guard['passed']}/{guard['total']} attacks / off-topic requests blocked")
    cols[4].metric("Noise-probe pass rate", fmt(noise["pass_rate"], pct=True), border=True,
                   help=f"{noise['passed']}/{noise['total']} noisy-only questions refused or not answered from noisy data")
    ret = summary.get("retrieval") or {}
    if ret.get("questions"):
        st.subheader("Retrieval ranking")
        table = pd.DataFrame(
            {f"@{k}": [ret.get(f"{m}@{k}") for m in ("recall", "precision", "ndcg", "hit")] for k in (1, 3, 5)},
            index=["Recall", "Precision", "NDCG", "Hit"],
        )
        left, right = st.columns([3, 1])
        left.dataframe(table, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns})
        right.metric("MRR", fmt(ret.get("mrr")), delta=delta("retrieval", "mrr"), border=True,
                     help="1 / rank of the first chunk from an expected source, averaged")
        st.caption(f"Over {ret['questions']} RAG questions; a chunk is relevant when it comes from the question's expected source document.")

    cost = summary.get("cost_usd") or {}
    lat = summary["latency_ms"]
    if cost.get("per_query_mean") is not None:
        st.subheader("Cost & latency")
        cols = st.columns(4)
        cols[0].metric("Cost / query (mean)", f"${cost['per_query_mean']:.4f}", border=True,
                       help="Pipeline LLM calls (guardrails, planner, responder); RAGAS judge excluded")
        cols[1].metric("Cost / query (p95)", f"${cost['per_query_p95']:.4f}", border=True)
        cols[2].metric("Latency p95", f"{lat['p95'] / 1000:.1f} s" if lat["p95"] else "—", border=True)
        cols[3].metric("Run total (pipeline)", f"${cost['total']:.4f}", border=True)
    st.caption(
        f"Answered {rag['answered']}/{rag['questions']} RAG questions · false blocks {rag['false_blocks']} · "
        f"no-context {rag['no_context_answers']} · errors {summary['errors']} · "
        f"latency p50 {lat['p50']} ms, p95 {lat['p95']} ms"
    )


def samples_frame(samples: list[dict], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for s in samples:
        status = "error" if s["error"] else "blocked" if s["blocked"] else "no context" if s["no_context"] else "answered"
        row = {
            "id": s["id"],
            "category": s["category"],
            "status": status,
            "pass": {True: "✅", False: "❌", None: ""}[s["passed"]],
            "question": s["question"],
            "noisy ctx": sum(1 for d in s["context_datasets"] if d == "noisy"),
            "latency ms": s["latency_ms"],
            "recall@5": (s.get("retrieval") or {}).get("recall@5"),
            "ndcg@5": (s.get("retrieval") or {}).get("ndcg@5"),
            "cost $": s.get("cost_usd"),
        }
        row.update({METRIC_LABELS[m]: s["scores"].get(m) for m in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def render_sample(s: dict) -> None:
    st.markdown(f"**Question:** {s['question']}")
    if s.get("standalone_question") and s["standalone_question"] != s["question"]:
        st.caption(f"Interpreted as: {s['standalone_question']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Answer**")
        if s["blocked"]:
            st.warning(f"Blocked at {s['blocked_stage']} by `{s['blocked_rail']}`")
        st.markdown(s["answer"] or "—")
    with right:
        st.markdown("**Reference**")
        st.markdown(s.get("reference") or f"_Expected behaviour: {s.get('expected_behavior')}_")
    if s["scores"]:
        st.json(s["scores"], expanded=False)
    for i, (text, source, dataset) in enumerate(zip(s["contexts"], s["context_sources"], s["context_datasets"]), start=1):
        icon = ":material/warning:" if dataset == "noisy" else ":material/description:"
        with st.expander(f"[{i}] {source} ({dataset})", icon=icon):
            st.text(text)


# ---------- Tabs ----------

st.title("RAG Evaluation Suite")
tab_run, tab_results, tab_probe = st.tabs([":material/play_arrow: Run evaluation", ":material/table: Results explorer",
                                            ":material/compare: True vs Noisy probe"])

with tab_run:
    rows = load_dataset(DEFAULT_DATASET)
    counts = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    st.caption(f"Dataset `{DEFAULT_DATASET.name}`: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    c1, c2, c3 = st.columns([2, 2, 1])
    categories = c1.multiselect("Categories", ["rag", "noise_probe", "guardrail"], default=["rag", "noise_probe", "guardrail"])
    metrics = c2.multiselect("RAGAS metrics", RAGAS_METRICS, default=RAGAS_METRICS, format_func=METRIC_LABELS.get)
    limit = c3.number_input("Limit", min_value=0, value=0, help="0 = all samples")
    skip_ragas = st.toggle("Skip RAGAS scoring (pipeline and behaviour checks only)", value=False)

    if st.button("Run evaluation", type="primary", icon=":material/play_arrow:", disabled=not categories):
        bar = st.progress(0.0, text="Running pipeline…")

        def progress(i: int, total: int, sample) -> None:
            bar.progress(i / total, text=f"[{i}/{total}] {sample.id}")

        try:
            with st.spinner("Evaluating…"):
                report = asyncio.run(run_evaluation(
                    categories=categories, metrics=metrics or None, limit=limit or None,
                    skip_ragas=skip_ragas or not metrics, progress=progress,
                ))
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
        else:
            path = save_report(report)
            bar.progress(1.0, text=f"Done in {report.duration_s}s — saved {path.name}")
            render_summary(report.summary)

with tab_results:
    reports = list_reports()
    if not reports:
        st.info("No saved runs yet. Run an evaluation first, or use `python -m evals.run_evals`.")
    else:
        choice = st.selectbox("Run", reports, format_func=lambda p: p.stem.removeprefix("eval-"))
        data = load_report(choice)
        index = reports.index(choice)
        previous = load_report(reports[index + 1])["summary"] if index + 1 < len(reports) else None
        st.caption(f"Config: {data['config']}" + (" · deltas vs previous run" if previous else ""))
        render_summary(data["summary"], previous)

        st.subheader("Samples")
        metrics_used = data["config"].get("metrics", [])
        frame = samples_frame(data["samples"], metrics_used)
        cat_filter = st.segmented_control("Show", ["all", "rag", "noise_probe", "guardrail"], default="all")
        if cat_filter and cat_filter != "all":
            frame = frame[frame["category"] == cat_filter]
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={METRIC_LABELS[m]: st.column_config.NumberColumn(format="%.2f") for m in metrics_used})

        sample_id = st.selectbox("Inspect sample", frame["id"].tolist())
        if sample_id:
            render_sample(next(s for s in data["samples"] if s["id"] == sample_id))

with tab_probe:
    st.markdown("Compare how the reranker scores **true** vs **noisy** chunks for the same question. "
                "Chunks below the relevance threshold are what the pipeline filters out.")
    settings = get_settings()

    @st.cache_resource(show_spinner="Loading retriever…")
    def get_retriever():
        from app.services.retrieval import Retriever
        return Retriever(settings)

    question = st.text_input("Question", value="How do I schedule a Kubernetes CronJob every 5 minutes?")
    top_n = st.slider("Chunks per dataset", 1, 10, 5)
    if st.button("Compare", icon=":material/compare_arrows:") and question.strip():
        try:
            retriever = get_retriever()
            cols = st.columns(2)
            for col, dataset in zip(cols, ["true", "noisy"]):
                result = retriever.retrieve(question, top_n=top_n, min_score=0.0, filters={"dataset": dataset})
                with col:
                    st.markdown(f"#### {dataset.capitalize()} data")
                    st.caption(f"Embeddings: {result.embedding_provider} · reranked: {result.reranked} · "
                               f"threshold {settings.rerank_min_score}")
                    rows = [{
                        "source": c.metadata.get("source", "").split("/")[-1],
                        "rerank": c.rerank_score,
                        "vector": round(c.vector_score, 3),
                        "kept": "✅" if c.rerank_score is None or c.rerank_score >= settings.rerank_min_score else "filtered",
                        "text": c.text[:160].replace("\n", " "),
                    } for c in result.chunks]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                                 column_config={"rerank": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
