"""Optional bearer-token auth and per-client rate limiting (Redis-backed or in-memory)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request, status

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    id: str  # hash of the API key (never the key itself), or "ip:<addr>" when auth is off
    groups: tuple[str, ...]  # ACL groups this caller may read


def authenticate(request: Request, settings: Settings) -> Principal:
    """Identify the caller; raise 401 if auth is enabled and the token is wrong."""
    key_groups = settings.api_key_groups
    keys = list(key_groups)
    if not keys:
        host = request.client.host if request.client else "unknown"
        return Principal(id=f"ip:{host}", groups=tuple(settings.acl_default_user_groups))

    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    # Compare against every key so timing doesn't reveal which/how many keys exist.
    matched: str | None = None
    for key in keys:
        if hmac.compare_digest(token.encode(), key.encode()):
            matched = key
    if matched is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})
    return Principal(id="key:" + hashlib.sha256(token.encode()).hexdigest()[:16], groups=tuple(key_groups[matched]))


class RateLimiter(Protocol):
    def hit(self, key: str) -> tuple[bool, int]:
        """Record a request; return (allowed, seconds until the client may retry)."""
        ...


class InMemoryRateLimiter:
    """Sliding-window limiter. Per-process only — use Redis when running several workers."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window - now) + 1)
            hits.append(now)
            return True, 0


class RedisRateLimiter:
    """Fixed-window limiter shared by every API worker/replica."""

    def __init__(self, url: str, limit: int, window_seconds: int = 60):
        import redis

        self.client = redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
        self.limit = limit
        self.window = window_seconds

    def hit(self, key: str) -> tuple[bool, int]:
        window_start = int(time.time()) // self.window
        redis_key = f"ratelimit:{key}:{window_start}"
        try:
            pipe = self.client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, self.window + 1)
            count, _ = pipe.execute()
        except Exception as exc:
            # Don't take the API down with Redis: allow the request and log it.
            logger.warning("Rate limiter unavailable (allowing request): %s", exc)
            return True, 0
        if count > self.limit:
            return False, self.window - int(time.time()) % self.window
        return True, 0


def build_rate_limiter(settings: Settings) -> RateLimiter | None:
    if settings.rate_limit_per_minute <= 0:
        return None
    if settings.redis_url:
        return RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute)
    return InMemoryRateLimiter(settings.rate_limit_per_minute)
