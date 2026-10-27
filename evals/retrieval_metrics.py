"""Rank-based retrieval metrics computed against the golden dataset's ``expected_sources``.

Relevance labels are at document level: a retrieved chunk counts as relevant when it comes from one
of the question's expected source documents. The ranked list is the context the agent actually
passed to the responder (after reranking and the relevance threshold), best first.

- Precision@K: share of the top-K retrieved chunks that are relevant. When fewer than K chunks were
  retrieved, the denominator is the number retrieved (so a short, fully relevant context scores 1.0).
- Recall@K: share of the expected documents that appear among the top-K chunks.
- NDCG@K: rank-aware gain with binary relevance, normalised by the ideal ordering of the same list.
- Hit@K: 1 if any of the top-K chunks is relevant.
- MRR: 1 / rank of the first relevant chunk (0 if none).
"""

from __future__ import annotations

import math

DEFAULT_KS = (1, 3, 5)


def _relevance(ranked_sources: list[str], expected: set[str]) -> list[int]:
    return [1 if source in expected else 0 for source in ranked_sources]


def precision_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    top = _relevance(ranked_sources, expected)[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(ranked_sources[:k]) & expected) / len(expected)


def ndcg_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    rel = _relevance(ranked_sources, expected)[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def hit_at_k(ranked_sources: list[str], expected: set[str], k: int) -> float:
    return 1.0 if any(_relevance(ranked_sources, expected)[:k]) else 0.0


def mrr(ranked_sources: list[str], expected: set[str]) -> float:
    for i, r in enumerate(_relevance(ranked_sources, expected), start=1):
        if r:
            return 1.0 / i
    return 0.0


def retrieval_metrics(ranked_sources: list[str], expected_sources: list[str], ks=DEFAULT_KS) -> dict[str, float]:
    expected = set(expected_sources)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(ranked_sources, expected, k), 4)
        out[f"precision@{k}"] = round(precision_at_k(ranked_sources, expected, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked_sources, expected, k), 4)
        out[f"hit@{k}"] = hit_at_k(ranked_sources, expected, k)
    out["mrr"] = round(mrr(ranked_sources, expected), 4)
    return out
