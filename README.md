# Enterprise Agentic RAG (Scalable Pipeline)

A production-grade RAG system built with LangGraph, the Portkey LLM gateway (OpenAI), and Jina AI
embeddings and reranker. It separates technical "True Data" from unrelated "Noisy Data" using
semantic reranking, history-aware planning, and NeMo Guardrails for input and output safety.

## Features

- **Agentic reasoning:** LangGraph planner → retriever → responder, with re-planning when nothing
  relevant is found and per-conversation memory.
- **Guardrails:** pattern checks plus NeMo Guardrails block off-topic, jailbreak and injection
  inputs before any retrieval, and check answers before they are returned.
- **LLM gateway:** all LLM calls go through Portkey, with model fallback (e.g. `gpt-4o` → `gpt-4o-mini`).
- **Search:** Qdrant Cloud vector search plus the Jina reranker with a relevance threshold.
- **Embeddings:** `jina-embeddings-v3` (1024-dim), with local `mxbai-embed-large-v1` as a fallback.
- **Local parsing:** PDF, HTML, TXT, DOCX and PPTX parsed on-device, with no external OCR service.
- **Observability:** nested traces in Pydantic Logfire and LangSmith across every node; Prometheus
  `/metrics`.
- **Security:** optional bearer-token auth; Redis-backed or in-memory rate limiting.
- **Evaluation:** RAGAS suite (6 metrics) plus True-vs-Noisy checks, with a Streamlit demo and a
  headless `evals/run_evals.py`.

## Quick start

```bash
cp .env.example .env     # set PORTKEY_API_KEY, JINA_API_KEY, QDRANT_URL, QDRANT_API_KEY
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.ingestion.pipeline              # DATA/ -> processed_data/
.venv/bin/python -m app.services.retrieval.indexer      # -> Qdrant
.venv/bin/uvicorn app.main:app --port 8000              # API
.venv/bin/streamlit run ui/app.py                       # chat UI on :8501
```

Or with Docker: `docker compose build && docker compose up -d` (run the two ingestion commands
first; see [DOCS/SETUP.md](DOCS/SETUP.md)).

## Project structure

```
├── app/
│   ├── agents/nodes/       # Planner, Retriever, Responder LangGraph nodes
│   ├── gateway/            # Portkey LLM gateway — primary + fallback model routing
│   ├── guardrails/         # NeMo Guardrails input/output filtering
│   ├── ingestion/
│   │   ├── chunking/       # Paragraph-based text splitter (1500 char max)
│   │   └── loaders/        # Local parsers — PDF (pypdf), HTML, TXT, DOCX, PPTX
│   ├── services/retrieval/ # Jina AI embeddings + Qdrant search + Jina AI reranking
│   ├── config.py           # Centralized environment variable management
│   ├── main.py             # FastAPI entrypoint — guardrails gate + /query endpoint
│   ├── security.py         # Bearer auth + rate limiting
│   ├── metrics.py          # Prometheus metrics
│   └── observability.py    # Logfire + LangSmith tracing
├── evals/                  # RAGAS evaluation suite + Streamlit 3-tab demo
├── ui/                     # Streamlit chat interface with reasoning-step transparency
├── processed_data/         # Auto-generated — parsed & chunked JSON per document
├── DOCS/                   # Architectural and operational guides
├── DATA/                   # Your documents (true_data/ vs noisy_data/) — not included, see DATA/README.md
├── Dockerfile, compose.yaml
└── requirements.txt        # Pinned dependencies
```

## Documentation

See [DOCS/](DOCS/README.md): architecture, setup, configuration, API, operations and evaluation.
