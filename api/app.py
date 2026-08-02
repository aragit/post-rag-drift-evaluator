from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from api.routes.telemetry import router as telemetry_router
from evaluator.db import pool as db_pool
from evaluator.drift_monitor import DriftMonitor
from evaluator.drift_store import DriftStore
from evaluator.logging_config import get_logger
from ingestion.queue import AsyncIngestionBuffer

logger = get_logger("api.app")


def create_app(
    store: Optional[DriftStore] = None,
    buffer: Optional[AsyncIngestionBuffer] = None,
    monitor: Optional[DriftMonitor] = None,
) -> FastAPI:
    """Build the drift ingestion gateway.

    ``store``/``buffer``/``monitor`` may be injected (e.g. fakes) for tests;
    defaults wire up the real Postgres-backed ``DriftStore``.
    """
    store = store or DriftStore()
    buffer = buffer or AsyncIngestionBuffer()
    monitor = monitor or DriftMonitor(store=store)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if isinstance(store, DriftStore):
            try:
                await db_pool.get_pool()
            except Exception as exc:  # noqa: BLE001 - API stays up without DB
                logger.warning("Database pool warmup failed: %s", exc)
        await buffer.start_worker(store)
        try:
            yield
        finally:
            await buffer.stop_worker()
            await store.close()
            if isinstance(store, DriftStore):
                await db_pool.close_pool()

    application = FastAPI(
        title="RAG & Agent Swarm Drift Engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.ingestion_buffer = buffer
    application.state.drift_store = store
    application.state.drift_monitor = monitor
    application.include_router(telemetry_router)
    return application


app = create_app()
