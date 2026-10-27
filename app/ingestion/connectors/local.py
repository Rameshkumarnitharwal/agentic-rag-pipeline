"""Local or mounted folder (e.g. a network share mounted into the container)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.ingestion.loaders import SUPPORTED_EXTENSIONS

from .base import RemoteDocument


class LocalFolderConnector:
    def __init__(self, config: dict[str, Any]):
        self.name = config["name"]
        self.root = Path(config["path"]).expanduser()
        if not self.root.is_dir():
            raise ValueError(f"Source {self.name!r}: {self.root} is not a directory")

    def list_documents(self) -> Iterable[RemoteDocument]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith((".", "~$")):
                stat = path.stat()
                yield RemoteDocument(
                    path=path.relative_to(self.root).as_posix(),
                    version=f"{stat.st_mtime_ns}-{stat.st_size}",
                    fetch=path.read_bytes,
                )
