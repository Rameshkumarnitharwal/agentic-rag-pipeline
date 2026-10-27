"""Copy the index into a new collection with BM25 sparse vectors (enables hybrid search).

Qdrant can't add a vector to an existing collection, so this creates a new one and copies every point
with its existing dense vectors and payload, computing BM25 from the stored text. No dense
re-embedding (no Jina calls). The source collection is left untouched.

Usage:
    python -m app.services.retrieval.migrate --to enterprise_rag_v2
    # then set QDRANT_COLLECTION=enterprise_rag_v2 and restart; delete the old collection when happy
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import models

from app.config import get_settings

from .embeddings import SPARSE_VECTOR, SparseEmbedder
from .vector_store import QdrantStore

logger = logging.getLogger(__name__)


def migrate(source: QdrantStore, target: QdrantStore, sparse: SparseEmbedder, batch_size: int = 64) -> int:
    if target.client.collection_exists(target.collection):
        raise SystemExit(f"Target collection {target.collection!r} already exists; choose a new name or delete it")
    target.ensure_collection(sparse=True)
    copied, offset = 0, None
    while True:
        points, offset = source.client.scroll(
            source.collection, limit=batch_size, with_payload=True, with_vectors=True, offset=offset
        )
        if not points:
            break
        bm25 = sparse.embed_documents([p.payload.get("text", "") for p in points])
        target.client.upsert(
            target.collection,
            points=[
                models.PointStruct(
                    id=p.id,
                    vector={**p.vector, SPARSE_VECTOR: models.SparseVector(indices=sv[0], values=sv[1])},
                    payload=p.payload,
                )
                for p, sv in zip(points, bm25)
            ],
            wait=True,
        )
        copied += len(points)
        logger.info("Copied %d points", copied)
        if offset is None:
            break
    src, dst = source.count(), target.count()
    if src != dst:
        raise SystemExit(f"Count mismatch after migration: source {src}, target {dst}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the index into a new collection with BM25 vectors.")
    parser.add_argument("--to", required=True, help="Name of the new collection")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    source = QdrantStore(settings)
    target = QdrantStore(settings.model_copy(update={"qdrant_collection": args.to}), client=source.client)
    if source.has_sparse():
        raise SystemExit(f"{source.collection!r} already has BM25 vectors; no migration needed")
    copied = migrate(source, target, SparseEmbedder(settings))
    logger.info("Done: %d points copied into %s. Set QDRANT_COLLECTION=%s to use it.", copied, args.to, args.to)


if __name__ == "__main__":
    main()
