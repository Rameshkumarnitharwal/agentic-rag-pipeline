"""Qdrant vector store with one named vector per embedding provider."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import Settings

from .embeddings import JINA_VECTOR, LOCAL_VECTOR

logger = logging.getLogger(__name__)

_KEYWORD_INDEXES = ("dataset", "source", "doc_id", "file_type", "content_type")
# Each point carries two 1024-dim vectors + text (~25 KB as JSON); keep requests well under a few MB
# so slow uplinks don't hit the client timeout.
UPSERT_SUB_BATCH = 32
UPSERT_ATTEMPTS = 3


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


def point_id(chunk_id: str) -> str:
    """Qdrant point id for a chunk (chunk ids are 32 hex chars, i.e. a UUID)."""
    return str(uuid.UUID(hex=chunk_id))


def build_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=settings.qdrant_timeout_seconds)
    if settings.qdrant_path:
        if settings.qdrant_path == ":memory:":
            return QdrantClient(location=":memory:")
        return QdrantClient(path=settings.qdrant_path)
    raise ValueError("Configure QDRANT_URL (Qdrant Cloud) or QDRANT_PATH (local)")


def _to_hit(point: Any) -> SearchHit:
    payload = dict(point.payload or {})
    text = payload.pop("text", "")
    return SearchHit(id=str(point.id), score=point.score, text=text, metadata=payload)


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Exact-match filter, e.g. ``{"dataset": "true"}``; list values match any of them."""
    if not filters:
        return None
    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value),
        )
        for key, value in filters.items()
    ]
    return models.Filter(must=conditions)


class QdrantStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.collection = settings.qdrant_collection
        self.client = client or build_client(settings)
        self.vector_dims = {JINA_VECTOR: settings.embedding_dim, LOCAL_VECTOR: 1024}

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        logger.info("Creating Qdrant collection %s", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                for name, dim in self.vector_dims.items()
            },
        )
        for field in _KEYWORD_INDEXES:
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def upsert_chunks(self, chunks: list[dict[str, Any]], vectors: dict[str, list[list[float]]]) -> None:
        """Upsert chunks (``{"chunk_id", "text", "metadata"}``) with their per-provider vectors."""
        points = [
            models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector={name: vecs[i] for name, vecs in vectors.items()},
                payload={"text": chunk["text"], **chunk["metadata"]},
            )
            for i, chunk in enumerate(chunks)
        ]
        for start in range(0, len(points), UPSERT_SUB_BATCH):
            self._upsert_with_retry(points[start : start + UPSERT_SUB_BATCH])

    def _upsert_with_retry(self, points: list[models.PointStruct]) -> None:
        # Upserts are idempotent (fixed point ids), so retrying after a timeout is safe.
        for attempt in range(1, UPSERT_ATTEMPTS + 1):
            try:
                self.client.upsert(self.collection, points=points, wait=True)
                return
            except ResponseHandlingException as exc:  # timeouts / dropped connections
                if attempt == UPSERT_ATTEMPTS:
                    raise
                delay = 2**attempt
                logger.warning("Qdrant upsert failed (%s); retry %d/%d in %ss", exc, attempt, UPSERT_ATTEMPTS - 1, delay)
                time.sleep(delay)

    def count(self, filters: dict[str, Any] | None = None) -> int:
        return self.client.count(self.collection, count_filter=build_filter(filters), exact=True).count

    def count_ids(self, chunk_ids: list[str], filters: dict[str, Any] | None = None) -> int:
        """How many of these chunk ids exist (optionally also matching ``filters``)."""
        if not chunk_ids:
            return 0
        base = build_filter(filters) or models.Filter()
        has_ids = models.HasIdCondition(has_id=[point_id(c) for c in chunk_ids])
        query = models.Filter(must=[*(base.must or []), has_ids])
        return self.client.count(self.collection, count_filter=query, exact=True).count

    def count_missing_vector(self, vector_name: str, filters: dict[str, Any] | None = None) -> int:
        """Points matching ``filters`` that have no ``vector_name`` vector (e.g. indexed during a Jina outage)."""
        base = build_filter(filters) or models.Filter()
        query = models.Filter(must=base.must, must_not=[models.HasVectorCondition(has_vector=vector_name)])
        return self.client.count(self.collection, count_filter=query, exact=True).count

    def delete(self, filters: dict[str, Any]) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=build_filter(filters)))

    def search_batch(
        self,
        vector_name: str,
        vectors: list[list[float]],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[list[SearchHit]]:
        """Run several searches in one request; one hit list per vector."""
        query_filter = build_filter(filters)
        requests = [
            models.QueryRequest(query=v, using=vector_name, filter=query_filter, limit=limit, with_payload=True)
            for v in vectors
        ]
        responses = self.client.query_batch_points(self.collection, requests=requests)
        return [[_to_hit(p) for p in r.points] for r in responses]

    def search(
        self,
        vector_name: str,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        response = self.client.query_points(
            self.collection,
            query=vector,
            using=vector_name,
            query_filter=build_filter(filters),
            limit=limit,
            with_payload=True,
        )
        return [_to_hit(p) for p in response.points]
