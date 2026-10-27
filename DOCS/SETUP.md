# Setup

## 1. Accounts and keys

| Service | What to do | Goes in `.env` |
|---|---|---|
| **Portkey** | Create an account. In **Model Catalog**, add an OpenAI provider and paste your **OpenAI API key** there. Note the provider's slug (e.g. `openai`). | `PORTKEY_API_KEY`, and `LLM_MODELS` using your slug |
| **Jina AI** | Create an API key (embeddings + reranker). | `JINA_API_KEY` |
| **Qdrant Cloud** | Create a cluster and an API key. | `QDRANT_URL`, `QDRANT_API_KEY` |
| Logfire *(optional)* | Create a project and a write token. | `LOGFIRE_TOKEN` |
| LangSmith *(optional)* | Create an API key. | `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` |

The OpenAI key is stored in Portkey, not in `.env`. If your Portkey provider slug isn't `openai`,
change `LLM_MODELS` and `EVAL_*_MODEL` to match (e.g. `@openai-prod/gpt-4o`).

**If your workspace blocks inline configs** (error `inline_config_blocked`), create a saved config
in Portkey → **Configs** with the fallback rules, and put its id in `PORTKEY_CONFIG_ID`:

```json
{
  "strategy": {"mode": "fallback", "on_status_codes": [429, 500, 502, 503, 504]},
  "targets": [
    {"override_params": {"model": "@openai/gpt-4o"}, "retry": {"attempts": 2}},
    {"override_params": {"model": "@openai/gpt-4o-mini"}, "retry": {"attempts": 2}}
  ]
}
```

Use your own provider slug in place of `openai`. With a saved config, routing is controlled in
Portkey and `LLM_MODELS` is only used as a placeholder model name.

## 2. Configure

```bash
cp .env.example .env
# edit .env: at minimum PORTKEY_API_KEY, JINA_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

Leave a value blank to leave that feature unset. Keep comments on their own lines: Docker Compose
treats `KEY=  # comment` as the value `# comment`. Every setting is listed in
[CONFIGURATION.md](CONFIGURATION.md).

For local experiments without Qdrant Cloud, set `QDRANT_PATH=./qdrant_local` instead of
`QDRANT_URL` (embedded Qdrant, single process only).

## 3a. Run locally

Requires Python 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Parse and chunk DATA/ -> processed_data/   (~30 s; re-runs skip unchanged files)
.venv/bin/python -m app.ingestion.pipeline

# Embed and upload to Qdrant   (re-runs skip unchanged documents)
.venv/bin/python -m app.services.retrieval.indexer

# API on :8000
.venv/bin/uvicorn app.main:app --port 8000

# Chat UI on :8501 (second terminal)
.venv/bin/streamlit run ui/app.py
```

Check `http://localhost:8000/ready`. It returns 503 with the reason if a key or Qdrant is
misconfigured.

The first start downloads the local fallback embedding model (~670 MB) unless
`LOCAL_EMBEDDINGS_ENABLED=false`.

## 3b. Run with Docker

Requires Docker Desktop (or another Docker engine) running.

```bash
docker compose build
docker compose run --rm api python -m app.ingestion.pipeline
docker compose run --rm api python -m app.services.retrieval.indexer
docker compose up -d                  # API :8000, UI :8501, Redis
docker compose --profile evals up -d  # optional: eval demo :8502
```

- The image includes CPU-only PyTorch and the local embedding model, and runs fully offline from
  Hugging Face (`HF_HUB_OFFLINE=1`). The first build takes ~10 minutes; the image is ~2.5–3 GB.
  Build with `--build-arg PRELOAD_LOCAL_EMBEDDINGS=false` (and set
  `LOCAL_EMBEDDINGS_ENABLED=false`) to drop ~670 MB.
- `DATA/` is mounted read-only and `processed_data/` read-write; neither is baked into the image.
- Use Qdrant Cloud with Docker. `QDRANT_PATH` data inside a container is lost on restart.
- On Linux, the container runs as uid 10001. Make `processed_data/` writable for that user if
  ingestion fails with a permission error.

## 4. Verify

```bash
curl -s localhost:8000/ready
curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $YOUR_API_KEY" \
  -d '{"question": "How do I schedule a Kubernetes CronJob every 5 minutes?"}'
```

Omit the `Authorization` header if `API_KEYS` is unset. Then run the evals
(`python -m evals.run_evals`) to get a baseline. See [EVALUATION.md](EVALUATION.md).

## Optional: access control, sources, memory

- **Access control:** copy `DATA/acl.example.yaml` to `DATA/acl.yaml`, map paths to groups, and give
  API keys groups (`API_KEYS="alice:hr|managers"`). See [PRODUCTION.md](PRODUCTION.md).
- **Auto-ingestion:** copy `sources.example.yaml` to `sources.yaml` (local folder, S3, Confluence) and
  run `python -m app.ingestion.sync` (add `--every 3600` to keep syncing).
- **Memory:** conversations persist in `state/memory.sqlite` by default; set `MEMORY_BACKEND=postgres`
  when running several API replicas.

## Adding your own documents

Put files in `DATA/true_data/` or `DATA/noisy_data/` (sub-folders are fine; the label comes from a
folder named exactly `true_data` or `noisy_data` in the file's path, and anything else gets
`dataset: unknown`). Supported: `.pdf .html .htm .txt .md .docx .pptx`.
`README.txt`/`README.md` and hidden or `~$` temp files are skipped. Then re-run the pipeline and
indexer. Only new or changed files are processed.

If the new documents cover new subjects, update `GUARDRAILS_ALLOWED_TOPICS`, or the input
guardrail will block questions about them.
