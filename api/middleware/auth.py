from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request
from starlette.status import HTTP_401_UNAUTHORIZED

from evaluator.config import config

HEALTH_PATHS = {"/healthz", "/readyz"}
METRICS_PATH = "/metrics"


def _is_public_path(path: str) -> bool:
    """Return True for health probes and the metrics endpoint."""
    return path in HEALTH_PATHS or path == METRICS_PATH


def verify_api_key(request: Request) -> Optional[str]:
    """FastAPI dependency: verify the request carries a valid API key.

    Reads the key from the ``X-API-Key`` header or an
    ``Authorization: Bearer <key>`` header.  When
    ``config.API_KEY_REQUIRED`` is ``False`` the dependency is a no-op
    and returns ``None``.
    """
    if not config.API_KEY_REQUIRED or _is_public_path(request.url.path):
        return None

    api_key: Optional[str] = request.headers.get("X-API-Key")
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer ") :]

    if not api_key or api_key not in config.API_KEYS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return api_key
