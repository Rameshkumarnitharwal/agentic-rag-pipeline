"""Atlassian Confluence Cloud: pages of one space, saved as HTML (parsed by the HTML loader)."""

from __future__ import annotations

import html
import re
from typing import Any, Iterable

import httpx

from .base import RemoteDocument, secret_from_env

_UNSAFE = re.compile(r"[^\w.-]+")


class ConfluenceConnector:
    """Config: ``base_url`` (e.g. https://acme.atlassian.net/wiki), ``space`` (space key), and env var
    names ``email_env`` / ``token_env`` for an Atlassian API token (basic auth)."""

    page_size = 50

    def __init__(self, config: dict[str, Any], client: httpx.Client | None = None):
        self.name = config["name"]
        self.space = config["space"]
        self.base_url = config["base_url"].rstrip("/")
        if client is None:
            client = httpx.Client(
                auth=(secret_from_env(config, "email_env"), secret_from_env(config, "token_env")),
                timeout=30,
            )
        self.client = client

    def list_documents(self) -> Iterable[RemoteDocument]:
        url = f"{self.base_url}/rest/api/content"
        params: dict[str, Any] | None = {
            "spaceKey": self.space, "type": "page", "status": "current",
            "expand": "version", "limit": self.page_size, "start": 0,
        }
        while url:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            body = response.json()
            for page in body.get("results", []):
                page_id, title = page["id"], page.get("title", "untitled")
                yield RemoteDocument(
                    path=f"{_UNSAFE.sub('_', title).strip('_')[:80]}-{page_id}.html",
                    version=str(page.get("version", {}).get("number", "")),
                    fetch=lambda page_id=page_id, title=title: self._page_html(page_id, title),
                )
            nxt = body.get("_links", {}).get("next")
            url, params = (f"{self.base_url}{nxt}" if nxt else None), None

    def _page_html(self, page_id: str, title: str) -> bytes:
        response = self.client.get(f"{self.base_url}/rest/api/content/{page_id}", params={"expand": "body.storage"})
        response.raise_for_status()
        storage = response.json()["body"]["storage"]["value"]
        # Confluence storage format is XHTML; wrap it so the title becomes the document's <h1>.
        return (f"<html><head><title>{html.escape(title)}</title></head>"
                f"<body><main><h1>{html.escape(title)}</h1>{storage}</main></body></html>").encode()
