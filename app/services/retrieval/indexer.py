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

from .embeddings import EmbeddingService
from .vector_store import QdrantStore

logger = logging.getLogger(__name__)

UPSERT_BATCH = 128


def index_processed_data(
    settings: Settings,
    embeddings: EmbeddingService,
    store: QdrantStore,
    processed_dir: Path | None = None,
    force: bool = False,
) -> dict[str, int]:
    processed_dir = processed_dir or settings.processed_data_dir
    store.ensure_collection()
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
                missing = [n for n in embeddings.vector_names if store.count_missing_vector(n, current)]
                if not missing:
                    stats["skipped"] += 1
                    continue
                logger.info("[%d/%d] %s is missing %s vectors; re-embedding", i, len(files), source, missing)
                stats["repaired"] += 1

        try:
            store.delete({"source": source})  # drop chunks from older versions of this file
            for start in range(0, len(chunks), UPSERT_BATCH):
                batch = chunks[start : start + UPSERT_BATCH]
                vectors = embeddings.embed_documents([c["text"] for c in batch])
                store.upsert_chunks(batch, vectors)
        except Exception:
            logger.exception("[%d/%d] Failed to index %s", i, len(files), source)
            stats["failed"] += 1
            continue

        stats["indexed"] += 1
        stats["chunks"] += len(chunks)
        logger.info("[%d/%d] Indexed %s (%d chunks)", i, len(files), source, len(chunks))

    logger.info("Done: %s (collection now has %d points)", stats, store.count())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed processed_data/ chunks and upsert them into Qdrant.")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed documents that are already indexed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    index_processed_data(settings, EmbeddingService(settings), QdrantStore(settings), args.processed_dir, args.force)


if __name__ == "__main__":
    main()
