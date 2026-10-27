"""Ingestion pipeline: parse every document under a data dir, chunk it, write JSON to processed_data/.

Usage:
    python -m app.ingestion.pipeline --data-dir DATA --out-dir processed_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.acl import AclPolicy, load_acl_policy
from app.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_sections
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document

logger = logging.getLogger(__name__)

# Sub-folder name -> dataset label stored on every chunk (used by retrieval filters and evals).
DATASET_LABELS = {"true_data": "true", "noisy_data": "noisy"}
SKIP_FILES = {"readme.txt", "readme.md", "acl.yaml"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_for(path: Path, data_dir: Path) -> str:
    for part in path.relative_to(data_dir).parts[:-1]:
        if part in DATASET_LABELS:
            return DATASET_LABELS[part]
    return "unknown"


def discover_files(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name.lower() not in SKIP_FILES
        and not p.name.startswith((".", "~$"))
    )


def process_file(
    path: Path, data_dir: Path, max_chars: int = DEFAULT_MAX_CHARS, acl: AclPolicy | None = None
) -> dict[str, Any]:
    """Parse and chunk one file; returns the JSON-serializable record written to processed_data/."""
    started = time.perf_counter()
    sections = load_document(path)
    relative = path.relative_to(data_dir).as_posix()
    file_hash = _file_sha256(path)
    base_metadata = {
        "source": relative,
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "dataset": _dataset_for(path, data_dir),
        "doc_id": file_hash[:32],
        "allowed_groups": (acl or AclPolicy()).groups_for(relative),
    }
    title = next((s.metadata["title"] for s in sections if s.metadata.get("title")), None)
    if title:
        base_metadata["title"] = title

    chunks = chunk_sections(sections, base_metadata=base_metadata, max_chars=max_chars)

    return {
        **base_metadata,
        "sha256": file_hash,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "total_chars": sum(len(s.text) for s in sections),
        "parse_seconds": round(time.perf_counter() - started, 3),
        "chunks": [asdict(c) for c in chunks],
    }


def output_path_for(path: Path, data_dir: Path, out_dir: Path) -> Path:
    # Keep the original extension in the name: the same title exists as .pdf/.txt/.html in DATA/.
    relative = path.relative_to(data_dir)
    return out_dir / relative.parent / f"{relative.name}.json"


def run(data_dir: Path, out_dir: Path, max_chars: int = DEFAULT_MAX_CHARS, force: bool = False) -> dict[str, int]:
    files = discover_files(data_dir)
    acl = load_acl_policy(data_dir / "acl.yaml")
    stats = {"processed": 0, "skipped": 0, "failed": 0, "empty": 0, "chunks": 0}
    logger.info("Found %d files under %s", len(files), data_dir)

    for i, path in enumerate(files, start=1):
        target = output_path_for(path, data_dir, out_dir)
        if not force and target.exists():
            try:
                existing = json.loads(target.read_text())
                if existing["sha256"] == _file_sha256(path) and existing.get("allowed_groups") == acl.groups_for(
                    path.relative_to(data_dir).as_posix()
                ):
                    stats["skipped"] += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            record = process_file(path, data_dir, max_chars, acl)
        except Exception:
            logger.exception("[%d/%d] Failed to parse %s", i, len(files), path.name)
            stats["failed"] += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        stats["processed"] += 1
        stats["chunks"] += record["num_chunks"]
        if record["num_chunks"] == 0:
            stats["empty"] += 1
        logger.info(
            "[%d/%d] %s -> %d chunks (%.2fs)",
            i, len(files), path.name, record["num_chunks"], record["parse_seconds"],
        )

    stats["pruned"] = prune_processed(data_dir, out_dir)
    logger.info("Done: %s", stats)
    return stats


def prune_processed(data_dir: Path, out_dir: Path) -> int:
    """Delete processed JSON whose source file no longer exists (document removed from DATA/)."""
    removed = 0
    for target in out_dir.rglob("*.json"):
        relative = target.relative_to(out_dir)
        source = data_dir / relative.parent / relative.name.removesuffix(".json")
        if not source.exists():
            target.unlink()
            removed += 1
            logger.info("Pruned %s (source document deleted)", relative)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and chunk documents into processed_data/ JSON.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--force", action="store_true", help="Re-process files even if unchanged")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # pypdf is very chatty about malformed PDFs
    run(args.data_dir.resolve(), args.out_dir.resolve(), args.max_chars, args.force)


if __name__ == "__main__":
    main()
