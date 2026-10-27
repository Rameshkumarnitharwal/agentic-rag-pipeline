# Operations runbook

## Routine tasks

| Task | Command |
|---|---|
| Ingest new or changed files | `python -m app.ingestion.pipeline` |
| Re-parse everything | `python -m app.ingestion.pipeline --force` |
| Index new or changed documents | `python -m app.services.retrieval.indexer` |
| Re-embed everything (e.g. after changing embedding model) | `python -m app.services.retrieval.indexer --force` |
| Smoke-test the pipeline (no judge LLM) | `python -m evals.run_evals --skip-ragas --limit 5` |
| Full evaluation | `python -m evals.run_evals` |

With Docker, prefix commands with `docker compose run --rm api`.

**After an embedding-provider outage:** if Jina (or the local model) failed while the indexer ran,
those chunks were stored with only the other vector, and normal searches can't see them. Just re-run
`python -m app.services.retrieval.indexer` once the provider is back. A document counts as indexed
only when every chunk has a vector from every enabled embedder, so incomplete documents are detected
and re-embedded (`repaired` in the summary).

**Updating a document:** replace the file in `DATA/`, then re-run the pipeline and indexer. The
indexer deletes the old version's chunks (matched by `source`) before uploading the new ones.

**Removing a document:** delete the file and its `processed_data/.../<name>.json`. The indexer
does not remove orphans, so delete its points in Qdrant by `source`, e.g. with qdrant-client:
`client.delete("enterprise_rag", points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value="true_data/old.docx"))])))`.

**Changing the embedding model or dimension:** create a new collection (`QDRANT_COLLECTION`),
re-index with `--force`, then switch the API over. Existing collections keep their vector size.

**Automatic sync:** `python -m app.ingestion.sync` mirrors the sources in `sources.yaml` and
re-indexes only what changed, pruning documents deleted at the source (see PRODUCTION.md §8). Run it
on a schedule with `--every 3600` or the compose `ingest` profile.

**Hybrid-search migration** (collections created before BM25 support):
`python -m app.services.retrieval.migrate --to <new_collection>`, then point `QDRANT_COLLECTION` at it.

**Feedback review:** `python -m evals.feedback_export --since-days 7` writes 👎 answers to
`evals/feedback_candidates.jsonl`; add a reference answer to each and append them to the dataset.

## Tuning

| Symptom (from evals or users) | Setting to adjust |
|---|---|
| Noisy chunks in context or citations | Raise `RERANK_MIN_SCORE` (e.g. 0.2 → 0.3). Confirm the reranker is running (`reranked: true` in retrieval traces). |
| Correct answers missing ("couldn't find…" for answerable questions) | Lower `RERANK_MIN_SCORE`, or raise `RETRIEVAL_TOP_K` / `RERANK_TOP_N` |
| Legitimate questions blocked | Widen `GUARDRAILS_ALLOWED_TOPICS` |
| Off-topic questions getting through | Make `GUARDRAILS_ALLOWED_TOPICS` more specific, or edit the prompts in `app/guardrails/config/config.yml` |
| Follow-ups misunderstood | Raise `AGENT_HISTORY_TURNS` |
| High latency or cost | Lower `AGENT_MAX_ITERATIONS` to 1, or `AGENT_MAX_CONTEXT_CHUNKS` |

Re-run the evals after each change and compare runs in the Results tab of the eval demo.

## Metrics

Exposed at `GET /metrics`:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `rag_query_requests_total` | counter | `outcome`: answered, no_context, blocked_input, blocked_output, error, rate_limited, unauthorized | Request outcomes |
| `rag_query_latency_seconds` | histogram | — | End-to-end `/query` latency |
| `rag_node_latency_seconds` | histogram | `node`: guardrails_input, planner, retriever, responder, guardrails_output | Per-stage latency |
| `guardrails_decisions_total` | counter | `stage`, `result`, `rail` | Which check allowed or blocked |
| `rag_retrieved_chunks` | histogram | — | Context chunks per answer |
| `rag_agent_iterations` | histogram | — | Planner/retriever cycles per query |
| `rag_cited_chunks_total` | counter | `dataset`: true, noisy | **Noisy citations indicate retrieval leakage** |
| `rag_llm_answers_total` | counter | `model` | Which model answered (shows Portkey fallbacks) |
| `rag_llm_tokens_total` | counter | `model`, `kind`: prompt, completion | LLM tokens used by `/query` |
| `rag_llm_cost_usd_total` | counter | — | LLM cost in USD (priced with `LLM_PRICES`) |
| `rag_query_cost_usd` | histogram | — | LLM cost per `/query` request |
| `rag_pii_redactions_total` | counter | `entity` | Answers with PII redacted |
| `rag_feedback_total` | counter | `rating`: up, down | User feedback |

Suggested alerts:

- `rate(rag_query_requests_total{outcome="error"}[5m]) / rate(rag_query_requests_total[5m]) > 0.05`
  for more than 5% errors.
- `increase(rag_cited_chunks_total{dataset="noisy"}[1h]) > 0` for noisy data being cited.
- `increase(guardrails_decisions_total{rail=~"error:.*"}[5m]) > 0` for guardrails unable to run
  (requests are being rejected with 503).
- p95 of `rag_query_latency_seconds` above your SLO.
- p95 of `rag_query_cost_usd` above your budget per question.
- `rag_llm_answers_total` growing for a model other than the first in `LLM_MODELS`, meaning the
  primary model is failing.

## Tracing

| Backend | Enable | What you see |
|---|---|---|
| Logfire | `LOGFIRE_TOKEN=...` (or `LOGFIRE_CONSOLE=true` to print locally) | `POST /query` → `rag_query` → guardrails, `agent.planner/retriever/responder` → `llm.chat`, `retrieval.embed/search/rerank`, plus HTTP calls to Jina, Qdrant and Portkey |
| LangSmith | `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` | `rag_query` → guardrail runs, the LangGraph `rag_agent` run with node runs, `llm` runs with messages and token counts, `retriever` runs listing each chunk with dataset and scores |

To follow one request: take `trace_id` from the API response (or the logs) and search for it in
Portkey logs, Logfire (`trace_id` attribute on `rag_query`) and LangSmith (run metadata).

Traces contain user questions, retrieved document text and answers. Logfire redacts values whose
names look like secrets, but document content is sent as-is, so confirm that's acceptable for
your data before enabling tracing.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/ready` 503 `PORTKEY_API_KEY is not set` | Key missing or blank in `.env` | Set it |
| `/ready` 503 `Configure QDRANT_URL ... or QDRANT_PATH` | No Qdrant configured | Set `QDRANT_URL` + `QDRANT_API_KEY` |
| `/ready` 503 `Storage folder ... is already accessed` | Two processes using the same `QDRANT_PATH` | Embedded Qdrant is single-process; use Qdrant Cloud or one process |
| Every `/query` returns 401 in Docker | A comment ended up as the value of `API_KEYS` in `.env` | Put comments on their own line |
| `/query` 503 "Safety checks are unavailable" | Guardrail LLM calls failing (Portkey/OpenAI outage, bad key, wrong provider slug) | Check Portkey logs; check `LLM_MODELS` slug matches your Model Catalog |
| `/query` 502 | All `LLM_MODELS` targets failed during the agent run | Same as above |
| Portkey 400 `inline_config_blocked` | Workspace only accepts saved configs | Create a saved config and set `PORTKEY_CONFIG_ID` (see SETUP.md) |
| Portkey 400 `Following keys are not valid: <slug>` | No Model Catalog provider with that slug | Add the OpenAI provider in Model Catalog, or fix the slug in `LLM_MODELS` / `EVAL_*_MODEL` |
| Portkey 400/401 and no fallback | Fallback triggers only on `LLM_FALLBACK_ON_STATUS_CODES` (429, 5xx) by design | Fix the key or model name; or set the variable empty to fall back on any error |
| Answers always "couldn't find…" | Collection empty, reranker threshold too high, or filters excluding everything | Run the indexer; check `candidates` in the retriever step; lower `RERANK_MIN_SCORE` |
| Retrieval logs `provider=local` | Jina embedding call failing; local fallback in use | Check `JINA_API_KEY` and logs for the Jina HTTP status |
| Retrieval logs `reranked=False` | Jina reranker failing; results in vector order with **no noise threshold** | Fix Jina; expect more noisy chunks until then |
| First request very slow or hanging locally | Local embedding model loading or downloading | It loads at startup; the first download needs network. Docker images ship it pre-downloaded. |
| `ModuleNotFoundError: langchain_community.chat_models.vertexai` | `langchain-community` upgraded past 0.4.1 | Reinstall from `requirements.txt` (RAGAS 0.4.3 needs ≤ 0.4.1) |
| A PDF yields no chunks | Scanned PDF with no text layer (no OCR by design) | Use a text-based PDF, or add OCR as a separate pre-processing step |

## Load testing

`loadtest/locustfile.py` (see PRODUCTION.md §9). Measured with 5 concurrent users on a laptop: 0 failures;
`/query` p50 10.0 s / p95 17 s. Re-run from the deployment region before quoting capacity.

## Scaling and limits

- **API workers and replicas:** conversation memory is SQLite (`state/`) by default, shared by the
  workers of one container. For several replicas set `MEMORY_BACKEND=postgres` so every replica sees
  the same conversations, and `REDIS_URL` for shared rate limits.
- **Rate limits:** set `REDIS_URL` so all containers share one limit. Without it, each process
  enforces its own.
- **Cost per question:** about 4 LLM calls (input guardrail and planner on `LLM_FAST_MODEL`,
  responder on the main model, output guardrail on `LLM_FAST_MODEL`), plus 2 for each re-plan; one
  Jina embedding call and one rerank call per retrieval regardless of query count. Questions blocked by
  the pattern checks cost no LLM calls; a question blocked by the NeMo input rail costs one discarded
  planner call. Each `/query` response reports its actual tokens and cost in `usage`.
- **Latency budget:** measured per stage with Logfire spans (see Tracing). The critical path is
  max(input rail, planner) → retrieval (embed, batched search, rerank) → responder → output rail.
  Every LLM call pays the network round trip to Portkey and OpenAI, so deploy close to them and
  re-measure from the deployment region before quoting production latency.

## Security notes

- Set `API_KEYS` in any shared environment. Only a hash of the key appears in logs and memory ids.
- `/metrics`, `/health` and `/ready` are unauthenticated; don't expose them publicly.
- NeMo Guardrails usage telemetry is disabled (`NEMO_GUARDRAILS_NO_USAGE_STATS=1`).
- The responder is instructed to ignore instructions inside retrieved documents, and the output
  rail blocks answers that follow them. Treat ingested documents as untrusted input all the same.
