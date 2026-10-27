# Production features

What turns the RAG pipeline into something a team can deploy for a real use case (e.g. an HR policy
assistant, IT helpdesk or customer-support bot). Each section: what it does, how to configure it, and
what was measured. Numbers come from runs on this repo's test corpus; re-measure in your environment.

## 1. Access control (document-level ACL)

Documents get `allowed_groups` at ingestion; callers get groups from their API key; retrieval only
returns chunks the caller may read.

```yaml
# DATA/acl.yaml (see DATA/acl.example.yaml) — first matching rule wins; globs relative to DATA/
default: [public]
rules:
  - match: "true_data/hr/**"
    groups: [hr, managers]
```

```bash
API_KEYS="alice-key:hr|managers,bob-key,admin-key:*"
```

- Everyone can read `public`; a key without groups gets `ACL_DEFAULT_USER_GROUPS`; `*` = unrestricted.
- The filter is applied inside Qdrant on every search, including hybrid search.
- Requests cannot pass `allowed_groups` as a filter (422).
- **ACL changes apply without re-embedding:** re-run the pipeline + indexer and the indexer
  updates the `allowed_groups` payload in place.
- Points indexed before ACL existed (no groups) are treated as `public`, matching the default policy.

Verified: a `public` key never retrieved a restricted document; a group key saw its documents plus public
ones; `*` saw everything; changing the ACL file re-tagged documents with zero re-embedding.

## 2. Persistent conversation memory

`MEMORY_BACKEND=sqlite` (default) stores LangGraph checkpoints in `state/memory.sqlite`;
`postgres` (with `MEMORY_POSTGRES_URL`, `pip install langgraph-checkpoint-postgres`) for multiple
replicas; `memory` for tests. Each turn is persisted once at the end (`durability="exit"`).

Verified: history survives an agent restart (the follow-up's planner saw the earlier turn); 8 concurrent
conversations on one SQLite file, 0 errors.

## 3. Hybrid search (BM25 + dense)

BM25 sparse vectors are computed locally with fastembed (`Qdrant/bm25`, no API cost) and stored
alongside the dense vectors. Each query runs dense and BM25 retrieval in one Qdrant request and fuses them
with reciprocal rank fusion, then the Jina reranker scores the fused candidates.

Qdrant cannot add a vector to an existing collection, so collections created before hybrid search are
migrated once — dense vectors and payloads are copied, BM25 is computed from the stored text, nothing is
re-embedded:

```bash
python -m app.services.retrieval.migrate --to enterprise_rag_v2
# then QDRANT_COLLECTION=enterprise_rag_v2; delete the old collection when satisfied
```

Until migrated, retrieval and indexing fall back to dense-only with a warning.

Measured on exact-term queries: for `concurrencyPolicy Forbid startingDeadlineSeconds` the second-best
chunk's rerank score rose 0.22 → 0.40; for `JOB_COMPLETION_INDEX` 0.43 → 0.68. Natural-language queries
were unchanged.

## 4. OCR for scanned PDFs

PDF pages with no text layer (< 20 characters extracted) are rendered (pypdfium2) and OCR'd locally with
RapidOCR (ONNX, no system install). `OCR_MAX_PAGES_PER_DOCUMENT` caps time on huge scans.

Measured: ~0.8 s per page warm. A fully scanned 313-page book went from 0 to 365 chunks; image-only pages
were also recovered in 8 other PDFs (slides, figures); the corpus grew from 4,755 to 5,257 chunks.

Note: the ingestion pipeline skips files whose content is unchanged, so after a loader upgrade (such as
enabling OCR) re-run it with `--force`.

## 5. PII redaction

Answers and citation snippets are redacted in the responder — before they are returned, streamed,
logged, stored for feedback or saved to conversation memory. Detected: email, phone (India and
international), Aadhaar (Verhoeff checksum), PAN, payment cards (brand length + Luhn) and US SSN;
`ip_address` is opt-in. Documentation domains (`example.com` etc.) are not redacted. Questions are
redacted before they are stored for feedback.

Verified: 0 false positives across all 157 true-data chunks (job ids, timestamps, ports, IPs); streamed
answers never leak PII even with 1-character deltas (output is emitted per sentence/line and redacted
before it is sent); metric `rag_pii_redactions_total{entity}`.

## 6. Feedback loop

Every answer is stored server-side by `trace_id` (`state/feedback.sqlite`). `POST /feedback`
(`{"trace_id", "rating": "up"|"down", "comment"}`) is accepted only from the caller who received that
answer. The chat UI shows 👍/👎 on each answer.

```bash
python -m evals.feedback_export --since-days 7    # 👎 answers → evals/feedback_candidates.jsonl
```

A reviewer fills in each candidate's reference answer and appends it to `evals/dataset.jsonl`, so the
golden set grows from real failures. Metric: `rag_feedback_total{rating}`.

## 7. Streaming and the planner fast path

`POST /query/stream` returns server-sent events: `start`, `step` (each stage), `token` (answer text,
already PII-redacted), then `retract` if the output guardrail rejects the streamed answer, and `done` with
the same JSON as `/query` (or `error`). The UI streams by default ("Stream answers" toggle).

The output guardrail needs the full answer, so it runs after streaming; a rejected answer is retracted
and replaced with the safe message. `/query` keeps the blocking behaviour.

`AGENT_PLANNER_FAST_PATH=true` (the default) skips the planner LLM call on a conversation's first turn
(nothing to rewrite) and searches with the question itself; follow-ups and long questions still use the
planner. A/B eval on the full golden set, all other settings equal:

| | Planner every turn | Fast path |
|---|---|---|
| p50 / p95 latency | 8.65 s / 17.1 s | **5.85 s / 12.0 s** (≈ −30%) |
| Faithfulness / answer relevancy | 0.919 / 0.900 | 0.932 / 0.878 |
| Context precision / recall | 0.879 / 0.987 | 0.907 / 0.987 |
| Precision@5 / NDCG@5 | 0.837 / 0.978 | 0.880 / 0.975 |
| Noisy leakage, guardrails, noise probes | 0%, 5/5, 3/3 | 0%, 5/5, 3/3 |

Quality differences are within judge noise (±0.03, both directions), so the fast path is on by default.
The maximum latencies in both runs (37 s, 48 s) were single slow provider responses with no error; a short
timeout with fallback on fast-tier calls is the next tail-latency fix.

Measured (warm, single user): first token at 5.3–7.9 s, the rest of the answer within ~0.5 s, the output
check ~1–1.4 s after. Most of the wait is planning and retrieval before generation starts, so streaming
helps less than the fast path; see EVALUATION.md for the fast-path quality comparison.

## 8. Auto-ingestion (connectors + sync)

```bash
cp sources.example.yaml sources.yaml        # local folder, S3, Confluence
python -m app.ingestion.sync                # once
python -m app.ingestion.sync --every 3600   # hourly; or: docker compose --profile ingest up -d
```

Each source is mirrored into `DATA/<dataset>/<name>/` (only new/changed documents are downloaded, tracked
by version/ETag), documents deleted at the source are removed, then the pipeline and indexer run
incrementally and **prune** deleted documents from `processed_data/` and Qdrant. Nothing is re-parsed or
re-indexed when no source changed. Secrets are referenced by environment-variable name in
`sources.yaml`, never stored in it.

Verified: local folder (edit/delete), S3 (with moto: prefix filtering, ETag change, deletion) and
Confluence (mocked REST API with pagination and page versions); end to end, a document deleted at the
source disappeared from DATA/, processed_data/ and the index. The S3 and Confluence connectors have not
been run against real accounts — test them with your credentials.

## 9. Load testing

```bash
pip install -r requirements-dev.txt
RATE_LIMIT_PER_MINUTE=0 uvicorn app.main:app --port 8000     # or raise the limit for the test
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
    --users 5 --spawn-rate 1 --run-time 2m --csv loadtest/results/run
```

Users ask golden-dataset questions (75% `/query`, 25% `/query/stream`, with time to first token as its
own metric). Every request makes real LLM/Jina/Qdrant calls, so a run costs money.

Measured — 5 concurrent users, 2 minutes, one API worker on a MacBook Air, home network: 42 requests,
**0 failures, 0 legitimate questions blocked**; `/query` p50 10.0 s / p95 17 s (single-user: 8.1 s /
11.5 s); streaming first token p50 7.1 s. Latency degrades ~25% at p50 and ~50% at p95 at this
concurrency. Re-run on server hardware in the deployment region before quoting capacity.

## Still out of scope

- SSO / OAuth identity (API keys carry groups today); per-user rather than per-key ACLs.
- Cross-encoder reranking hosted locally (the reranker is the Jina API).
- Answer caching and request queuing/back-pressure under heavy load.
- Connectors beyond local/S3/Confluence (SharePoint, Google Drive, Notion).
