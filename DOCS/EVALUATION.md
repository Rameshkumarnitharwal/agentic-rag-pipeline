# Evaluation

The eval suite runs a fixed dataset through the real pipeline (guardrails → agent → guardrails),
scores the answers with RAGAS, and adds checks specific to the True-vs-Noisy problem.

## Running

```bash
python -m evals.run_evals                              # full run, all 6 RAGAS metrics
python -m evals.run_evals --skip-ragas --limit 5       # quick smoke test, no judge LLM
python -m evals.run_evals --category guardrail         # one category (repeatable)
python -m evals.run_evals --metric faithfulness        # subset of metrics (repeatable)
streamlit run evals/app.py                             # demo UI with 3 tabs
```

Results are saved to `evals/results/eval-<timestamp>.json` (skip with `--no-save`).

Requirements: everything the API needs (Portkey, Jina, an indexed Qdrant collection). The judge
model (`EVAL_JUDGE_MODEL`, default `@openai/gpt-4o-mini`) and embeddings (`EVAL_EMBEDDING_MODEL`)
go through Portkey to your OpenAI provider. A full run is ~23 pipeline runs plus ~90 judge calls.

## Dataset

`evals/dataset.jsonl`, one JSON object per line:

| Category | Count | Fields | Pass condition |
|---|---|---|---|
| `rag` | 15 | `question`, `reference` (ground-truth answer written from the true docs), `expected_sources`, optional `history` (earlier turns, for follow-ups) | Scored by RAGAS |
| `noise_probe` | 3 | `question` on a topic that exists only in noisy files | Blocked, "not in the docs", or at least nothing cited from noisy data |
| `guardrail` | 5 | Jailbreak, injection, harmful and off-topic requests | Blocked |

To add a case, append a line, e.g.:

```json
{"id": "cron-06", "category": "rag", "question": "...", "reference": "...", "expected_sources": ["true_data/cronjobs.docx"]}
```

Write references from the source documents, not from model output. Otherwise the evals measure
agreement with the model instead of correctness.

## Metrics

**RAGAS** (0–1, averaged over answered `rag` questions):

| Metric | Measures | Low score usually means |
|---|---|---|
| Faithfulness | Claims in the answer supported by the retrieved context | Hallucination; responder using outside knowledge |
| Answer relevancy | Answer addresses the question | Off-target or padded answers |
| Context precision | Relevant chunks ranked above irrelevant ones | Reranking or threshold issue; noise near the top |
| Context recall | Reference facts present in the retrieved context | Retrieval missing the right chunks |
| Factual correctness | Answer claims match the reference (F1) | Wrong or incomplete answers |
| Noise sensitivity ↓ | Incorrect claims driven by irrelevant context (**lower is better**) | Noisy chunks influencing the answer |

**Retrieval ranking** (no LLM judge; computed from the ranked chunks each question retrieved):

| Metric | Meaning |
|---|---|
| Recall@K | Share of the question's expected source documents found in the top-K chunks |
| Precision@K | Share of the top-K chunks that come from an expected source (denominator is the number retrieved when fewer than K) |
| NDCG@K | Rank-aware: rewards relevant chunks appearing earlier (binary relevance) |
| Hit@K | 1 if any of the top-K chunks is relevant |
| MRR | 1 / rank of the first relevant chunk |

K = 1, 3, 5. Relevance labels are **document-level** (`expected_sources` in the dataset): a chunk is
relevant when it comes from an expected document. The ranked list is the context actually passed to
the responder, after reranking and the relevance threshold. A question with nothing retrieved scores 0.

**Cost and latency:** every LLM call made while answering a question (guardrails, planner, responder)
is recorded with its token counts and priced with `LLM_PRICES`. The summary reports mean and p95
cost per query, mean cost per answered question, run total and token totals. The RAGAS judge's own
calls are not included. Latency is measured per question; for multi-turn cases only the scored turn
is timed and costed.

**Project-specific:**

| Metric | Meaning | Target |
|---|---|---|
| `source_hit_rate` | RAG questions where a chunk from the expected document was retrieved | High |
| `noisy_context_rate` ↓ | Share of retrieved chunks (for RAG questions) from noisy data | ~0 |
| `noisy_citation_rate` ↓ | Share of citations pointing at noisy data | 0 |
| `guardrail_pass_rate` | Attacks and off-topic requests blocked | 1.0 |
| `noise_probe_pass_rate` | Noisy-only questions refused or not answered from noisy data | 1.0 |
| `false_blocks` | Legitimate questions blocked by guardrails | 0 |
| Latency p50 / p95 | Per sample, end to end | Your SLO |

## CI gate

`--fail-under METRIC=VALUE` (repeatable) exits with status 1 when a metric misses its threshold.
For `noise_sensitivity`, `noisy_*`, `cost_*` and `latency_*` metrics the value is a maximum; for
everything else a minimum. Retrieval metrics use their summary names (`recall@5`, `ndcg@3`, `mrr`, ...).

```bash
python -m evals.run_evals \
  --fail-under faithfulness=0.8 \
  --fail-under context_recall=0.7 \
  --fail-under recall@5=0.9 \
  --fail-under cost_per_query=0.02 \
  --fail-under latency_p95_ms=20000 \
  --fail-under noisy_citation_rate=0.0 \
  --fail-under guardrail_pass_rate=1.0
```

Set thresholds from a baseline run on a known-good configuration rather than picking numbers up
front.

## Demo app tabs

1. **Run evaluation**: choose categories, metrics and a limit; progress bar; summary tiles.
2. **Results explorer**: pick a saved run; metric tiles show the change against the previous
   run; per-sample table; a detail view with the answer next to the reference and each retrieved
   chunk labelled true or noisy.
3. **True vs Noisy probe**: type a question and see the reranker scores of the best true and
   noisy chunks side by side, including which would fall below `RERANK_MIN_SCORE`. Useful for
   choosing the threshold.

## Known gap to check first

In local testing without the Jina reranker, `noise-03` (phrased like the synthetic noise text)
passed the input guardrail and was answered from noisy files. With Jina configured, this case
shows whether `RERANK_MIN_SCORE` is high enough. If it still fails, raise the threshold or
tighten `GUARDRAILS_ALLOWED_TOPICS`.
