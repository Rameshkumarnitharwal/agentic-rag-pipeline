"""Retrieval pipeline: embed query -> Qdrant vector search -> Jina rerank -> relevance threshold."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.observability import annotate, span, traced

from .embeddings import EmbeddingService, SparseEmbedder
from .reranker import JinaReranker
from .vector_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    embedding_provider: str
    reranked: bool
    candidates: int  # hits returned by vector search
    filtered_out: int  # reranked hits dropped by the relevance threshold
    hybrid: bool = False  # dense + BM25 fused, or dense only
    timings_ms: dict[str, float] = field(default_factory=dict)


def _retrieval_outputs(result: RetrievalResult | None) -> dict:
    if result is None:
        return {}
    # LangSmith renders retriever runs as a document list.
    return {
        "documents": [
            {
                "type": "Document",
                "page_content": c.text,
                "metadata": {**c.metadata, "vector_score": c.vector_score, "rerank_score": c.rerank_score},
            }
            for c in result.chunks
        ]
    }


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: EmbeddingService | None = None,
        store: QdrantStore | None = None,
        reranker: JinaReranker | None = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings or EmbeddingService(self.settings)
        self.store = store or QdrantStore(self.settings)
        if reranker is None and self.settings.jina_api_key is not None:
            reranker = JinaReranker(self.settings)
        self.reranker = reranker
        self.sparse = SparseEmbedder(self.settings) if self.settings.hybrid_search_enabled else None
        self._hybrid_ready: bool | None = None  # checked lazily: does the collection have BM25 vectors?

    def _sparse_queries(self, queries: list[str]) -> list | None:
        if self.sparse is None:
            return None
        if self._hybrid_ready is None:
            try:
                self._hybrid_ready = self.store.has_sparse()
            except Exception:
                self._hybrid_ready = False
            if not self._hybrid_ready:
                logger.warning("Collection has no BM25 vectors yet; using dense-only search (run the indexer)")
        if not self._hybrid_ready:
            return None
        try:
            return self.sparse.embed_queries(queries)
        except Exception as exc:
            logger.warning("BM25 query embedding failed, using dense-only search: %s", exc)
            return None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
        acl_groups: list[str] | None = None,
    ) -> RetrievalResult:
        return self.retrieve_many(
            [query], top_k=top_k, top_n=top_n, min_score=min_score, filters=filters, acl_groups=acl_groups
        )

    @traced("retrieval", run_type="retriever", process_outputs=_retrieval_outputs)
    def retrieve_many(
        self,
        queries: list[str],
        rerank_query: str | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
        acl_groups: list[str] | None = None,
    ) -> RetrievalResult:
        """``acl_groups``: only chunks readable by these groups are returned (``*`` = unrestricted;
        ``None`` = no ACL filter, for internal tools only).

        Retrieve for several queries with a fixed number of round trips: one embedding call for all
        queries, one batched vector search, then a single rerank of the merged candidates against
        ``rerank_query`` (default: the first query) — instead of embed/search/rerank per query.
        """
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]  # dedupe, keep order
        rerank_query = rerank_query or queries[0]
        top_k = top_k or self.settings.retrieval_top_k
        top_n = top_n or self.settings.rerank_top_n
        min_score = self.settings.rerank_min_score if min_score is None else min_score
        timings: dict[str, float] = {}

        started = time.perf_counter()
        with span("retrieval.embed", queries=len(queries)) as s:
            vector_name, vectors = self.embeddings.embed_queries(queries)
            if s is not None:
                s.set_attribute("provider", vector_name)
        timings["embed"] = _ms(started)

        sparse = self._sparse_queries(queries)
        started = time.perf_counter()
        with span("retrieval.search", top_k=top_k, queries=len(queries), hybrid=sparse is not None, filters=str(filters or {})):
            hit_lists = self.store.search_batch(
                vector_name, vectors, limit=top_k, filters=filters, acl_groups=acl_groups, sparse=sparse
            )
        timings["search"] = _ms(started)

        # Merge: a chunk found by several queries keeps its best vector score.
        best: dict[str, RetrievedChunk] = {}
        for hits in hit_lists:
            for h in hits:
                if h.id not in best or h.score > best[h.id].vector_score:
                    best[h.id] = RetrievedChunk(h.id, h.text, h.metadata, vector_score=h.score)
        candidates = sorted(best.values(), key=lambda c: c.vector_score, reverse=True)
        candidates = candidates[: self.settings.rerank_max_candidates]

        with span("retrieval.rerank", candidates=len(candidates), top_n=top_n):
            chunks, reranked = self._rerank(rerank_query, candidates, top_n, timings)

        filtered_out = 0
        if reranked:
            kept = [c for c in chunks if c.rerank_score is not None and c.rerank_score >= min_score]
            filtered_out = len(chunks) - len(kept)
            chunks = kept

        annotate(
            embedding_provider=vector_name, hybrid=sparse is not None, reranked=reranked, queries=len(queries),
            candidates=len(candidates),
            kept=len(chunks), filtered_out=filtered_out,
            noisy_kept=sum(1 for c in chunks if c.metadata.get("dataset") == "noisy"),
        )
        logger.info(
            "Retrieved %d/%d chunks for %d queries (provider=%s, reranked=%s, filtered=%d)",
            len(chunks), len(candidates), len(queries), vector_name, reranked, filtered_out,
        )
        return RetrievalResult(
            query=rerank_query,
            chunks=chunks,
            embedding_provider=vector_name,
            reranked=reranked,
            hybrid=sparse is not None,
            candidates=len(candidates),
            filtered_out=filtered_out,
            timings_ms=timings,
        )

    def _rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int, timings: dict[str, float]
    ) -> tuple[list[RetrievedChunk], bool]:
        """Rerank with Jina; if unavailable, fall back to vector-score order (no threshold applied)."""
        if not candidates:
            return [], self.reranker is not None
        if self.reranker is None:
            return candidates[:top_n], False

        started = time.perf_counter()
        try:
            ranked = self.reranker.rerank(query, [c.text for c in candidates], top_n)
        except Exception as exc:
            logger.warning("Reranking failed, using vector order: %s", exc)
            return candidates[:top_n], False
        finally:
            timings["rerank"] = _ms(started)

        out = []
        for index, score in ranked:
            candidates[index].rerank_score = score
            out.append(candidates[index])
        return out, True


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
