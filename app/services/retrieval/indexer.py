"""Index processed_data/ JSON chunks into Qdrant.

Usage:
    python -m app.services.retrieval.indexer            # index new/changed documents
    python -m app.services.retrieval.indexer --force    # re-embed everything
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.config import Settings, get_settings

from .embeddings import SPARSE_VECTOR, EmbeddingService, SparseEmbedder
from .vector_store import QdrantStore

logger = logging.getLogger(__name__)

UPSERT_BATCH = 128


def index_processed_data(
    settings: Settings,
    embeddings: EmbeddingService,
    store: QdrantStore,
    processed_dir: Path | None = None,
    force: bool = False,
    prune: bool | None = None,
) -> dict[str, int]:
    # Pruning treats anything not in processed_dir as deleted, so by default it only runs when indexing
    # the full processed_data/ folder — never for an ad-hoc subset, which would wipe everything else.
    prune = (processed_dir is None) if prune is None else prune
    processed_dir = processed_dir or settings.processed_data_dir
    store.ensure_collection(sparse=settings.hybrid_search_enabled)
    sparse = SparseEmbedder(settings) if settings.hybrid_search_enabled else None
    if sparse is not None and not store.has_sparse():
        logger.warning(
            "Collection %s has no BM25 vector (created before hybrid search); indexing dense-only. "
            "Migrate with: python -m app.services.retrieval.migrate --to <new_collection>", store.collection,
        )
        sparse = None
    if sparse is not None:
        # Points indexed before hybrid search get BM25 vectors from their stored text (no dense re-embedding).
        filled = store.backfill_sparse(sparse.embed_documents)
        if filled:
            logger.info("Added BM25 vectors to %d existing points", filled)
    required_vectors = embeddings.vector_names + ([SPARSE_VECTOR] if sparse is not None else [])
    files = sorted(processed_dir.rglob("*.json"))
    stats = {"indexed": 0, "skipped": 0, "repaired": 0, "failed": 0, "chunks": 0}

    for i, path in enumerate(files, start=1):
        record = json.loads(path.read_text())
        source, chunks = record["source"], record["chunks"]

        if not force:
            # Chunk ids hash the source, position and text, so if every chunk id is present for this
            # source (and nothing extra), this exact chunking is already indexed. Comparing ids rather
            # than counts also catches re-chunking of an unchanged file.
            current = {"source": source}
            ids = [c["chunk_id"] for c in chunks]
            if store.count(current) == len(chunks) and store.count_ids(ids, current) == len(chunks):
                # ...but only if every enabled embedder's vector is present. A provider outage during an
                # earlier run leaves points without that vector, which normal searches can't see.
                missing = [n for n in required_vectors if store.count_missing_vector(n, current)]
                if not missing:
                    # Vectors are current; still apply ACL changes (payload-only, no re-embedding).
                    store.set_payload({"allowed_groups": record.get("allowed_groups", ["public"])}, current)
                    stats["skipped"] += 1
                    continue
                logger.info("[%d/%d] %s is missing %s vectors; re-embedding", i, len(files), source, missing)
                stats["repaired"] += 1

        try:
            store.delete({"source": source})  # drop chunks from older versions of this file
            for start in range(0, len(chunks), UPSERT_BATCH):
                batch = chunks[start : start + UPSERT_BATCH]
                texts = [c["text"] for c in batch]
                vectors = embeddings.embed_documents(texts)
                store.upsert_chunks(batch, vectors, sparse.embed_documents(texts) if sparse is not None else None)
        except Exception:
            logger.exception("[%d/%d] Failed to index %s", i, len(files), source)
            stats["failed"] += 1
            continue

        stats["indexed"] += 1
        stats["chunks"] += len(chunks)
        logger.info("[%d/%d] Indexed %s (%d chunks)", i, len(files), source, len(chunks))

    if prune:
        # Remove documents that are no longer in processed_data/ (deleted at the source).
        known = {json.loads(path.read_text())["source"] for path in files}
        for source in sorted(store.sources() - known):
            store.delete({"source": source})
            stats["pruned"] = stats.get("pruned", 0) + 1
            logger.info("Pruned %s from the index (document deleted)", source)
    logger.info("Done: %s (collection now has %d points)", stats, store.count())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed processed_data/ chunks and upsert them into Qdrant.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed documents that are already indexed")
    parser.add_argument("--no-prune", action="store_true", help="Keep index entries for documents deleted from processed_data/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    index_processed_data(settings, EmbeddingService(settings), QdrantStore(settings), args.processed_dir, args.force,
                         prune=False if args.no_prune else None)


if __name__ == "__main__":
    main()
