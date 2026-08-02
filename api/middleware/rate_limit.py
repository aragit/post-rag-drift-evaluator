from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from evaluator.config import config
from evaluator.logging_config import get_logger

logger = get_logger("RateLimiter")

_WINDOW_SECONDS: float = 60.0
_requests: dict[str, list[float]] = defaultdict(list)
_lock = asyncio.Lock()


def _get_client_id(request: Request) -> str:
    """Determine the rate-limit key: API key when present, otherwise client IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def reset_rate_limits() -> None:
    """Clear all rate-limit state (primarily for testing)."""
    _requests.clear()


async def rate_limit(request: Request) -> str:
    """FastAPI dependency enforcing a sliding-window rate limit.

    Uses an in-memory fixed-window counter keyed by API key (when present)
    or client IP.  Returns the client identifier; raises ``429`` when the
    configured per-minute limit is exceeded.
    """
    client_id = _get_client_id(request)
    now = time.monotonic()
    window_start = now - _WINDOW_SECONDS

    async with _lock:
        entries = _requests[client_id]
        entries[:] = [t for t in entries if t > window_start]

        max_requests = config.RATE_LIMIT_PER_MINUTE
        if len(entries) >= max_requests:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - entries[0])))
            logger.warning("Rate limit exceeded for client %s", client_id)
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        entries.append(now)
        _requests[client_id] = entries

    return client_id
