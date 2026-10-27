"""Conversation-memory backends (LangGraph checkpointers).

- ``memory``:   in-process only; lost on restart, not shared between workers (tests, local dev).
- ``sqlite``:   a local file; survives restarts, shared by workers on one machine (default).
- ``postgres``: shared by every replica; needs ``langgraph-checkpoint-postgres`` and MEMORY_POSTGRES_URL.
"""

from __future__ import annotations

import logging
import sqlite3

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings

logger = logging.getLogger(__name__)


def build_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    backend = settings.memory_backend
    if backend == "memory":
        return InMemorySaver()
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = settings.memory_sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across threads; SqliteSaver serialises access with its own lock.
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers while a request writes
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("Conversation memory: SQLite at %s", path)
        return saver
    if backend == "postgres":
        if settings.memory_postgres_url is None:
            raise ValueError("MEMORY_BACKEND=postgres requires MEMORY_POSTGRES_URL")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
        except ImportError as exc:
            raise ImportError("Install langgraph-checkpoint-postgres for MEMORY_BACKEND=postgres") from exc
        conn = Connection.connect(settings.memory_postgres_url.get_secret_value(), autocommit=True)
        saver = PostgresSaver(conn)
        saver.setup()
        logger.info("Conversation memory: Postgres")
        return saver
    raise ValueError(f"Unknown MEMORY_BACKEND {backend!r} (use memory, sqlite or postgres)")
