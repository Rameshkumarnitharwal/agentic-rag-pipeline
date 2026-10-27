# Configuration reference

All settings come from environment variables or `.env` in the project root (`app/config.py`).
Names are case-insensitive. A blank value (`KEY=`) means unset. Keep comments on their own lines.

Also read by the Streamlit chat UI: `RAG_API_URL` (default `http://localhost:8000`) and `RAG_API_KEY`.

## Paths

| Variable | Default | Notes |
|---|---|---|
| `DATA_DIR` | `./DATA` |  |
| `PROCESSED_DATA_DIR` | `./processed_data` |  |

## Jina AI (embeddings + reranker)

| Variable | Default | Notes |
|---|---|---|
| `JINA_API_KEY` | unset | Jina embeddings + reranker. Unset: local embeddings only and **no reranking** (no noise threshold). |
| `JINA_BASE_URL` | `https://api.jina.ai/v1` |  |
| `JINA_EMBEDDING_MODEL` | `jina-embeddings-v3` |  |
| `JINA_RERANKER_MODEL` | `jina-reranker-v2-base-multilingual` |  |
| `JINA_TIMEOUT_SECONDS` | `30.0` |  |
| `JINA_MAX_RETRIES` | `3` |  |
| `EMBEDDING_DIM` | `1024` | Jina vector size. Changing it needs a new collection. |
| `EMBEDDING_BATCH_SIZE` | `64` |  |

## Local embedding fallback

| Variable | Default | Notes |
|---|---|---|
| `LOCAL_EMBEDDINGS_ENABLED` | `True` | Fallback embedder; also embeds every chunk at index time. Downloads ~670 MB on first use. |
| `LOCAL_EMBEDDING_MODEL` | `mixedbread-ai/mxbai-embed-large-v1` |  |

## Qdrant

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_URL` | unset | Qdrant Cloud URL. Takes precedence over `QDRANT_PATH`. |
| `QDRANT_API_KEY` | unset |  |
| `QDRANT_PATH` | unset | Embedded Qdrant directory (or `:memory:`). Single process only. |
| `QDRANT_COLLECTION` | `enterprise_rag` | Collection name. Use a new one when changing embedding models. |
| `QDRANT_TIMEOUT_SECONDS` | `60` |  |

## Portkey LLM gateway

| Variable | Default | Notes |
|---|---|---|
| `PORTKEY_API_KEY` | unset | Required. All LLM calls (agent, guardrails, eval judge) go through Portkey. |
| `PORTKEY_BASE_URL` | unset | Only for a self-hosted gateway |
| `PORTKEY_CONFIG_ID` | unset | Saved Portkey config id (`pc-...`). Required if the workspace blocks inline configs; then it controls routing/fallback instead of `LLM_MODELS`. |
| `LLM_MODELS` | `@openai/gpt-4o,@openai/gpt-4o-mini` | Fallback order, `@<portkey-provider-slug>/<model>`. |
| `LLM_FAST_MODEL` | `@openai/gpt-4o-mini` | Model for guardrail checks and planning, called directly (no saved config); falls back to `LLM_MODELS` on failure. Empty = use `LLM_MODELS` for everything. |
| `LLM_FALLBACK_ON_STATUS_CODES` | `429,500,502,503,504` | Empty = fall back on any non-2xx |
| `LLM_RETRY_ATTEMPTS` | `2` | Retries per model before falling back. |
| `LLM_TIMEOUT_SECONDS` | `60` |  |
| `LLM_TEMPERATURE` | `0.0` |  |
| `LLM_PRICES` | `{"gpt-4o": [2.5, 10], "gpt-4o-mini": [0.15, 0.6]}` | USD per 1M tokens `[input, output]`, matched by longest model-name prefix. JSON. Used for cost per query; check OpenAI's current prices. |
| `LLM_MAX_TOKENS` | `1024` |  |

## API security

| Variable | Default | Notes |
|---|---|---|
| `API_KEYS` | unset | Comma-separated bearer tokens; unset = auth disabled |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per API key (or client IP when auth is disabled); 0 = off |
| `REDIS_URL` | unset | E.g. redis://localhost:6379/0; unset = in-memory limiter |
| `MAX_QUESTION_CHARS` | `4000` | Longer questions are rejected with 422. |

## Observability

| Variable | Default | Notes |
|---|---|---|
| `LOGFIRE_TOKEN` | unset | Enables Logfire tracing. |
| `LOGFIRE_SERVICE_NAME` | `enterprise-rag` |  |
| `LOGFIRE_ENVIRONMENT` | `development` |  |
| `LOGFIRE_CONSOLE` | `False` | Print spans to the terminal (no token needed). |
| `LANGSMITH_TRACING` | `False` | Enables LangSmith tracing (with `LANGSMITH_API_KEY`). |
| `LANGSMITH_API_KEY` | unset |  |
| `LANGSMITH_PROJECT` | `enterprise-rag` |  |
| `LANGSMITH_ENDPOINT` | unset | E.g. https://eu.api.smith.langchain.com |

## Guardrails

| Variable | Default | Notes |
|---|---|---|
| `GUARDRAILS_ENABLED` | `True` |  |
| `GUARDRAILS_OUTPUT_ENABLED` | `True` |  |
| `GUARDRAILS_FAIL_OPEN` | `False` | If a rail can't run: false = reject with 503, true = allow. |
| `GUARDRAILS_MAX_INPUT_CHARS` | `4000` | Longer inputs are blocked by the input guardrail. |
| `GUARDRAILS_ALLOWED_TOPICS` | `Kubernetes workloads … and Databricks job automation …` | Topics the input/output rails allow. Update when adding documents on new subjects. |

## Agent (LangGraph)

| Variable | Default | Notes |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | `2` | Planner -> retriever loops before answering with what we have |
| `AGENT_PARALLEL_PLANNING` | `True` | Run the planner concurrently with the input guardrail; the plan is discarded if the input is blocked. |
| `AGENT_MAX_SEARCH_QUERIES` | `3` | Search queries the planner may write per attempt. |
| `AGENT_MAX_CONTEXT_CHUNKS` | `6` | Chunks passed to the responder |
| `AGENT_HISTORY_TURNS` | `6` | Past user/assistant turns kept in conversation memory |
| `AGENT_ANSWER_MAX_TOKENS` | `1024` | Max tokens for the responder's answer. |

## Evaluation (RAGAS)

| Variable | Default | Notes |
|---|---|---|
| `EVAL_JUDGE_MODEL` | `@openai/gpt-4o-mini` | RAGAS judge LLM. |
| `EVAL_EMBEDDING_MODEL` | `@openai/text-embedding-3-small` | Embeddings for the answer-relevancy metric. |
| `EVAL_CONCURRENCY` | `4` | Samples scored in parallel |
| `EVAL_JUDGE_MAX_TOKENS` | `4096` | Judge output limit. RAGAS's default (1024) truncates statement extraction for long answers. |

## Retrieval

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_TOP_K` | `20` | Candidates fetched from Qdrant |
| `RERANK_TOP_N` | `5` | Chunks returned after reranking |
| `RERANK_MAX_CANDIDATES` | `40` | Merged multi-query candidates sent to the single rerank call (best vector scores). |
| `RERANK_MIN_SCORE` | `0.2` | Drop reranked chunks below this relevance (noise filter) |
