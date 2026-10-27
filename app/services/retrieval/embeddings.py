"""Embedding providers: Jina AI API (primary) with a local mxbai-embed-large-v1 fallback.

The two models produce vectors in *different* spaces, so they can't be mixed in one index.
Each chunk is stored with one named vector per provider in Qdrant, and a query is searched
against the vector that matches the provider that embedded it.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import httpx

from app.config import Settings

from ._http import JinaAPIError, post_json

logger = logging.getLogger(__name__)

JINA_VECTOR = "jina"
LOCAL_VECTOR = "local"


class Embedder(Protocol):
    vector_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class JinaEmbedder:
    vector_name = JINA_VECTOR

    def __init__(self, settings: Settings):
        if settings.jina_api_key is None:
            raise ValueError("JINA_API_KEY is not set")
        self.model = settings.jina_embedding_model
        self.dim = settings.embedding_dim
        self.batch_size = settings.embedding_batch_size
        self.max_retries = settings.jina_max_retries
        self._url = f"{settings.jina_base_url}/embeddings"
        self._client = httpx.Client(
            timeout=settings.jina_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.jina_api_key.get_secret_value()}"},
        )

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = post_json(
                self._client,
                self._url,
                {
                    "model": self.model,
                    "task": task,
                    "dimensions": self.dim,
                    "normalized": True,
                    "input": batch,
                },
                self.max_retries,
            )
            data = sorted(body["data"], key=lambda item: item["index"])
            if len(data) != len(batch):
                raise JinaAPIError(f"Expected {len(batch)} embeddings, got {len(data)}")
            vectors.extend(item["embedding"] for item in data)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task="retrieval.query")[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.query")  # one API call for all queries


class LocalEmbedder:
    """mxbai-embed-large-v1 via sentence-transformers. The model loads lazily on first use."""

    vector_name = LOCAL_VECTOR
    dim = 1024
    _QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

    def __init__(self, settings: Settings):
        self.model_name = settings.local_embedding_model
        self.batch_size = settings.embedding_batch_size
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # heavy import (torch)

                    logger.info("Loading local embedding model %s", self.model_name)
                    try:
                        # Use the cached copy without contacting huggingface.co (that check can hang).
                        self._model = SentenceTransformer(self.model_name, local_files_only=True)
                    except Exception:
                        logger.info("Model not cached locally; downloading %s", self.model_name)
                        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
        vectors = self.model.encode(
            texts, prompt=prompt, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], prompt=self._QUERY_PROMPT)[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, prompt=self._QUERY_PROMPT)


class EmbeddingService:
    """Picks the primary embedder (Jina) and falls back to the local model on failure."""

    def __init__(self, settings: Settings):
        self.embedders: list[Embedder] = []
        if settings.jina_api_key is not None:
            self.embedders.append(JinaEmbedder(settings))
        if settings.local_embeddings_enabled:
            self.embedders.append(LocalEmbedder(settings))
        if not self.embedders:
            raise ValueError("No embedding provider configured: set JINA_API_KEY or LOCAL_EMBEDDINGS_ENABLED=true")

    def warmup(self) -> None:
        """Load local models up front so the first request doesn't pay the load time."""
        for embedder in self.embedders:
            if isinstance(embedder, LocalEmbedder):
                embedder.embed_query("warmup")

    @property
    def vector_names(self) -> list[str]:
        return [e.vector_name for e in self.embedders]

    def embed_documents(self, texts: list[str]) -> dict[str, list[list[float]]]:
        """Embed with every configured provider so each fallback has vectors to search.

        A provider that fails is skipped (logged); raises only if all fail.
        """
        results: dict[str, list[list[float]]] = {}
        for embedder in self.embedders:
            try:
                results[embedder.vector_name] = embedder.embed_documents(texts)
            except Exception:
                logger.exception("Document embedding failed with %s", type(embedder).__name__)
        if not results:
            raise RuntimeError("All embedding providers failed")
        return results

    def embed_query(self, text: str) -> tuple[str, list[float]]:
        """Return ``(vector_name, vector)`` from the first provider that succeeds."""
        name, vectors = self.embed_queries([text])
        return name, vectors[0]

    def embed_queries(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        """Embed several queries in one call; returns ``(vector_name, vectors)`` from the first provider
        that succeeds. All vectors come from the same provider, so they search the same named vector."""
        last_error: Exception | None = None
        for embedder in self.embedders:
            try:
                return embedder.vector_name, embedder.embed_queries(texts)
            except Exception as exc:
                last_error = exc
                logger.warning("Query embedding failed with %s: %s — trying fallback", type(embedder).__name__, exc)
        raise RuntimeError("All embedding providers failed") from last_error
