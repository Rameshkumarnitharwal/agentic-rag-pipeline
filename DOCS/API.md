# API reference

Base URL: `http://localhost:8000`. Interactive OpenAPI docs are served at `/docs`.

## Authentication

When `API_KEYS` is set (comma-separated), `/query` requires `Authorization: Bearer <key>`.
`/health`, `/ready` and `/metrics` are always open, so restrict them at your network edge if
needed.

Rate limiting applies per API key (or per client IP when auth is off):
`RATE_LIMIT_PER_MINUTE` requests per minute.

## `POST /query`

Runs input guardrails → agent → output guardrails synchronously and returns the final answer.

**Request**

```json
{
  "question": "How do I schedule a Kubernetes CronJob every 5 minutes?",
  "thread_id": "optional-conversation-id",
  "filters": {"dataset": "true"}
}
```

| Field | Required | Notes |
|---|---|---|
| `question` | yes | 1 to `MAX_QUESTION_CHARS` (4000) characters |
| `thread_id` | no | Reuse it for follow-up questions. A new id is generated if omitted; it is returned either way. Max 128 chars. |
| `filters` | no | Exact-match metadata filters. A list value matches any element, e.g. `{"file_type": ["pdf", "docx"]}`. Indexed fields: `dataset`, `source`, `doc_id`, `file_type`, `content_type`. |

**Response 200**

```json
{
  "answer": "Set `schedule: \"*/5 * * * *\"` on the CronJob [1].",
  "citations": [
    {"ref": 1, "source": "true_data/cronjobs.docx", "location": null, "dataset": "true",
     "score": 0.91, "snippet": "The schedule field uses the standard cron format..."}
  ],
  "steps": [
    {"node": "guardrails_input", "summary": "Input guardrails passed", "detail": {"rail": null}, "latency_ms": 412.0},
    {"node": "planner", "summary": "Search for: CronJob schedule every 5 minutes", "detail": {"standalone_question": "...", "reasoning": "..."}, "latency_ms": 890.1},
    {"node": "retriever", "summary": "Kept 5 relevant chunks from 2 documents (3 low-relevance chunks filtered out)", "detail": {"queries": [], "candidates": 20, "sources": []}, "latency_ms": 640.3},
    {"node": "responder", "summary": "Answered with 1 cited sources", "detail": {"model": "gpt-4o-2024-08-06", "context_chunks": 5}, "latency_ms": 1820.7},
    {"node": "guardrails_output", "summary": "Output guardrails passed", "detail": {"rail": null}, "latency_ms": 380.2}
  ],
  "blocked": false,
  "blocked_stage": null,
  "thread_id": "3f0c...",
  "trace_id": "9b1e...",
  "standalone_question": "How do I schedule a Kubernetes CronJob every 5 minutes?",
  "model": "gpt-4o-2024-08-06",
  "latency_ms": 4150.6,
  "usage": {"llm_calls": 4, "prompt_tokens": 4410, "completion_tokens": 395, "cost_usd": 0.0150}
}
```

(Latencies above are illustrative.)

- `detail.rail` names the check that decided only when a request is blocked (e.g.
  `heuristic:injection`, `self check input`).
- `blocked: true` with `blocked_stage: "input" | "output"` means a guardrail refused the request.
  `answer` then holds a safe message, and the reason is in `steps` and the logs, never in the
  response text.
- `model` is the model that actually answered. If it isn't the first in `LLM_MODELS`, Portkey
  fell back.
- `trace_id` identifies the request in Portkey, Logfire and LangSmith.
- `usage` totals the LLM calls made for this request; `cost_usd` uses `LLM_PRICES` and is `null` if a
  model has no price configured.

**Errors**

| Status | When | Client action |
|---|---|---|
| 401 | Missing or wrong bearer token | Fix the key |
| 422 | Invalid body or question too long | Fix the request |
| 429 | Rate limit exceeded. `Retry-After` header gives seconds. | Wait and retry |
| 502 | All LLM targets failed during the agent run | Retry later |
| 503 | Service not initialised (see `/ready`), or guardrail checks couldn't run (LLM outage, fail-closed) | Retry later |
| 500 | Unexpected error, logged with the `trace_id` | Report the `trace_id` |

## `POST /query/stream`

Same request body as `/query`. Returns server-sent events (`text/event-stream`):

| Event | Data |
|---|---|
| `start` | `{"trace_id", "thread_id"}` |
| `step` | one pipeline stage as it completes (`node`, `summary`, `detail`, `latency_ms`) |
| `token` | `{"text"}` — the next piece of the answer (a sentence or line, already PII-redacted) |
| `retract` | `{"message"}` — the output guardrail rejected the streamed answer; replace it with this |
| `done` | the same JSON as `/query`'s response |
| `error` | `{"status", "detail"}` — e.g. 502 when all LLM targets fail (headers were already sent) |

```bash
curl -N -X POST localhost:8000/query/stream -H 'Content-Type: application/json' \
  -d '{"question": "How do I schedule a CronJob every 5 minutes?"}'
```

Errors before streaming starts (401, 422, 429, 503 not initialised) are normal HTTP errors.

## `POST /feedback`

```json
{"trace_id": "9b1e...", "rating": "down", "comment": "missed the cron syntax"}
```

`204` on success. `404` if the trace is unknown **or belongs to another caller** (same response, so
answer ids can't be probed). `rating` is `up` or `down`. Ratings are stored with the answer and exported
for the golden dataset (`python -m evals.feedback_export`).

## Access control

API keys can carry groups: `API_KEYS="alice:hr|managers,bob"`. Retrieval only returns chunks whose
`allowed_groups` (from `DATA/acl.yaml`) include one of the caller's groups or `public`. Sending
`allowed_groups` in `filters` is rejected with 422. See [PRODUCTION.md](PRODUCTION.md#1-access-control-document-level-acl).

## `GET /health`

Liveness: `{"status": "ok"}` whenever the process is up.

## `GET /ready`

Readiness: 200 `{"status": "ready", "guardrails": true, "auth": true}`, or 503
`{"status": "unavailable", "error": "..."}` if startup failed (missing keys, Qdrant unreachable,
etc.). Use it for load-balancer and Kubernetes readiness probes.

## `GET /metrics`

Prometheus exposition format. Metric list in [OPERATIONS.md](OPERATIONS.md#metrics).

## Example: multi-turn conversation

```bash
H='Content-Type: application/json'
A="Authorization: Bearer $KEY"
curl -s -X POST localhost:8000/query -H "$H" -H "$A" \
  -d '{"question": "What is a Kubernetes CronJob?", "thread_id": "demo-1"}'
curl -s -X POST localhost:8000/query -H "$H" -H "$A" \
  -d '{"question": "How would I make it run every day at 2 AM?", "thread_id": "demo-1"}'
```

The second call's `standalone_question` shows how the planner resolved "it" from history.
