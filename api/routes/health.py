from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from evaluator.db import pool as db_pool
from evaluator.logging_config import get_logger

logger = get_logger("api.routes.health")

router = APIRouter()


@router.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    """Liveness probe — the process is alive as long as this responds."""
    return {"status": "alive"}


@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Liveness/readiness alias used by the container and k8s probes."""
    return {"status": "alive"}


@router.get("/readyz", tags=["health"])
async def readyz(request: Request) -> dict[str, Any]:
    """Readiness probe — verifies DB pool reachability and buffer state."""
    db_ok = False
    db_error: str | None = None
    try:
        pool = await db_pool.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as exc:  # noqa: BLE001 - probe must never crash
        db_error = str(exc)
        logger.warning("Readiness probe DB check failed: %s", exc)

    buffer = getattr(request.app.state, "ingestion_buffer", None)
    buffer_ready = buffer is not None and getattr(buffer, "is_running", False)

    if not db_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "db": "disconnected",
                "db_error": db_error,
                "buffer": "ok" if buffer_ready else "not running",
            },
        )

    if not buffer_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "db": "connected",
                "buffer": "not running",
            },
        )

    return {"status": "ready", "db": "connected", "buffer": "running"}
