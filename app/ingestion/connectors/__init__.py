"""Document source connectors for auto-ingestion (see app/ingestion/sync.py)."""

from .base import Connector, RemoteDocument, build_connector

__all__ = ["Connector", "RemoteDocument", "build_connector"]
