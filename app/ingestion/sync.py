"""Auto-ingestion: mirror document sources into DATA/, then parse and index only what changed.

Sources are declared in ``sources.yaml`` (see ``sources.example.yaml``). Each source is mirrored into
``DATA/<dataset>/<name>/`` so deletions are computed per source and a sync never touches files it
didn't create. Then the ingestion pipeline (incremental, prunes deleted files) and the indexer
(incremental, prunes deleted documents from Qdrant) run.

Usage:
    python -m app.ingestion.sync                     # all sources, once
    python -m app.ingestion.sync --source eng-wiki   # one source
    python -m app.ingestion.sync --every 3600        # loop: sync every hour
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config import PROJECT_ROOT, Settings, get_settings
from app.ingestion.connectors import Connector, build_connector

logger = logging.getLogger(__name__)

STATE_FILE = ".sync_state.json"  # hidden: ignored by the ingestion pipeline
DATASET_FOLDERS = {"true", "true_data", "noisy", "noisy_data"}


@dataclass
class SyncStats:
    source: str
    downloaded: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: list[str] = field(default_factory=list)


def load_sources(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"No sources file at {path}. Copy sources.example.yaml to sources.yaml and edit it.")
    sources = (yaml.safe_load(path.read_text()) or {}).get("sources", [])
    names = [s.get("name") for s in sources]
    if len(names) != len(set(names)) or not all(names):
        raise SystemExit(f"{path}: every source needs a unique 'name'")
    for s in sources:
        dataset = s.get("dataset", "true_data")
        if dataset not in DATASET_FOLDERS:
            raise SystemExit(f"Source {s['name']!r}: dataset must be true_data or noisy_data")
    return sources


def target_dir(data_dir: Path, source: dict[str, Any]) -> Path:
    dataset = source.get("dataset", "true_data")
    dataset = {"true": "true_data", "noisy": "noisy_data"}.get(dataset, dataset)
    return data_dir / dataset / source["name"]


def mirror(connector: Connector, dest: Path) -> SyncStats:
    """Download new/changed documents into ``dest`` and delete ones that disappeared from the source."""
    stats = SyncStats(source=connector.name)
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / STATE_FILE
    state: dict[str, str] = json.loads(state_path.read_text()) if state_path.exists() else {}
    seen: dict[str, str] = {}

    for doc in connector.list_documents():
        target = (dest / doc.path).resolve()
        if dest.resolve() not in target.parents:  # never write outside the source's folder
            logger.warning("Skipping %s: path escapes the source folder", doc.path)
            continue
        seen[doc.path] = doc.version
        if state.get(doc.path) == doc.version and target.exists():
            stats.unchanged += 1
            continue
        try:
            content = doc.fetch()
        except Exception as exc:
            logger.warning("[%s] Failed to fetch %s: %s", connector.name, doc.path, exc)
            stats.failed.append(doc.path)
            seen[doc.path] = state.get(doc.path, "")  # retry next run
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(target)  # atomic: the pipeline never sees half-written files
        stats.downloaded += 1

    for path in set(state) - set(seen):
        local = dest / path
        if local.exists():
            local.unlink()
        stats.deleted += 1
        logger.info("[%s] Removed %s (deleted at source)", connector.name, path)

    state_path.write_text(json.dumps(seen, indent=1, sort_keys=True))
    return stats


def sync_once(settings: Settings, sources: list[dict[str, Any]], only: str | None = None) -> dict[str, Any]:
    from app.ingestion.pipeline import run as run_pipeline
    from app.services.retrieval.embeddings import EmbeddingService
    from app.services.retrieval.indexer import index_processed_data
    from app.services.retrieval.vector_store import QdrantStore

    results = []
    for source in sources:
        if only and source["name"] != only:
            continue
        try:
            stats = mirror(build_connector(source), target_dir(settings.data_dir, source))
        except Exception as exc:
            logger.exception("Source %s failed", source["name"])
            stats = SyncStats(source=source["name"], failed=[f"source error: {exc}"])
        logger.info("[%s] downloaded=%d unchanged=%d deleted=%d failed=%d", stats.source, stats.downloaded,
                    stats.unchanged, stats.deleted, len(stats.failed))
        results.append(stats)

    changed = any(s.downloaded or s.deleted for s in results)
    summary: dict[str, Any] = {"sources": [vars(s) for s in results], "changed": changed}
    if changed:
        summary["pipeline"] = run_pipeline(settings.data_dir, settings.processed_data_dir)
        summary["index"] = index_processed_data(settings, EmbeddingService(settings), QdrantStore(settings))
    else:
        logger.info("No source changes; skipping parse and index")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync document sources and re-index what changed.")
    parser.add_argument("--sources", type=Path, default=PROJECT_ROOT / "sources.yaml")
    parser.add_argument("--source", help="Sync only this source name")
    parser.add_argument("--every", type=int, default=0, help="Repeat every N seconds (0 = run once)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    while True:
        started = time.time()
        try:
            summary = sync_once(settings, load_sources(args.sources), args.source)
            logger.info("Sync finished in %.0fs: %s", time.time() - started, json.dumps(summary, default=str))
        except SystemExit:
            raise
        except Exception:
            logger.exception("Sync run failed; will retry on the next interval")
        if not args.every:
            break
        time.sleep(max(0, args.every - (time.time() - started)))


if __name__ == "__main__":
    main()
