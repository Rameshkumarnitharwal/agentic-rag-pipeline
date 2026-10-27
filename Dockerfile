# Enterprise Agentic RAG — one image for the API, the chat UI and the eval demo.
#
#   docker build -t enterprise-rag .
#   docker run -p 8000:8000 --env-file .env enterprise-rag                     # API
#   docker run -p 8501:8501 --env-file .env enterprise-rag \
#       streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501     # chat UI
#
# Or use compose.yaml to run API + UI + Redis together.

# ---------- Stage 1: build the virtualenv ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only PyTorch first (the default wheel bundles CUDA and adds several GB). Same version as local dev.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the local fallback embedding model into the image so containers never download it at runtime
# (that huggingface.co call can hang). Build with --build-arg PRELOAD_LOCAL_EMBEDDINGS=false to skip
# it (~670 MB smaller) when LOCAL_EMBEDDINGS_ENABLED=false.
# The BM25 model for hybrid search is small and always baked in; OCR models ship inside rapidocr.
ARG PRELOAD_LOCAL_EMBEDDINGS=true
ENV HF_HOME=/opt/hf \
    FASTEMBED_CACHE_PATH=/opt/fastembed
RUN mkdir -p /opt/hf /opt/fastembed \
    && python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')" \
    && if [ "$PRELOAD_LOCAL_EMBEDDINGS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')"; \
    fi

# ---------- Stage 2: runtime ----------
FROM python:3.13-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    TOKENIZERS_PARALLELISM=false \
    NEMO_GUARDRAILS_NO_USAGE_STATS=1

RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY --from=builder /opt/fastembed /opt/fastembed

WORKDIR /app
COPY app/ app/
COPY ui/ ui/
COPY evals/ evals/

# Writable paths for ingestion output, embedded Qdrant, eval results and runtime state (conversation
# memory + feedback SQLite). Mount volumes over them.
RUN mkdir -p processed_data qdrant_local evals/results state && chown -R app:app processed_data qdrant_local evals/results state

USER app
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Conversation memory is SQLite on the state/ volume (one container). To run several replicas, use
# MEMORY_BACKEND=postgres and REDIS_URL so memory and rate limits are shared.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
