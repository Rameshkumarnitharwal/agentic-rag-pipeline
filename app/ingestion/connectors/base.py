"""Connector interface and factory.

A connector lists the documents a source currently holds (``path`` relative to the source, plus a
``version`` that changes when content changes) and fetches a document's bytes on demand. The sync job
only downloads documents whose version changed, and deletes local copies of documents that vanished.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol


@dataclass
class RemoteDocument:
    path: str  # relative path inside the source, e.g. "runbooks/cronjobs.docx"
    version: str  # etag / mtime / page version — anything that changes when the content does
    fetch: Callable[[], bytes]


class Connector(Protocol):
    name: str

    def list_documents(self) -> Iterable[RemoteDocument]: ...


def secret_from_env(config: dict[str, Any], key: str, required: bool = True) -> str | None:
    """Secrets are referenced by environment-variable name in sources.yaml, never stored in it."""
    env_name = config.get(key)
    value = os.environ.get(env_name) if env_name else None
    if required and not value:
        raise ValueError(f"Source {config.get('name')!r}: set environment variable named by {key!r} ({env_name})")
    return value


def build_connector(config: dict[str, Any]) -> Connector:
    kind = config.get("type")
    if kind == "local":
        from .local import LocalFolderConnector

        return LocalFolderConnector(config)
    if kind == "s3":
        from .s3 import S3Connector

        return S3Connector(config)
    if kind == "confluence":
        from .confluence import ConfluenceConnector

        return ConfluenceConnector(config)
    raise ValueError(f"Unknown source type {kind!r} (use local, s3 or confluence)")
