"""Small HTTP helper with retry/backoff for the Jina AI API."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class JinaAPIError(RuntimeError):
    pass


def post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    """POST ``payload`` and return the JSON body, retrying transient failures with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=payload)
        except httpx.TransportError as exc:
            error: str = f"transport error: {exc}"
        else:
            if response.status_code < 400:
                return response.json()
            error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in _RETRYABLE_STATUS:
                raise JinaAPIError(error)

        if attempt == max_retries:
            raise JinaAPIError(f"{url} failed after {max_retries + 1} attempts ({error})")
        delay = min(2**attempt, 10)
        logger.warning("Jina API %s; retrying in %ss", error, delay)
        time.sleep(delay)
    raise AssertionError("unreachable")
