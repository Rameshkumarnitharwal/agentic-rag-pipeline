"""Jina AI Reranker API client."""

from __future__ import annotations

import httpx

from app.config import Settings

from ._http import post_json


class JinaReranker:
    def __init__(self, settings: Settings):
        if settings.jina_api_key is None:
            raise ValueError("JINA_API_KEY is not set")
        self.model = settings.jina_reranker_model
        self.max_retries = settings.jina_max_retries
        self._url = f"{settings.jina_base_url}/rerank"
        self._client = httpx.Client(
            timeout=settings.jina_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.jina_api_key.get_secret_value()}"},
        )

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return ``(document_index, relevance_score)`` pairs, best first."""
        if not documents:
            return []
        body = post_json(
            self._client,
            self._url,
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
            self.max_retries,
        )
        results = [(item["index"], float(item["relevance_score"])) for item in body["results"]]
        return sorted(results, key=lambda pair: pair[1], reverse=True)
