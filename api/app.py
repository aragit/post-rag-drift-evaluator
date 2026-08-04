from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.telemetry import router as telemetry_router
from evaluator.baseline_service import DynamicBaselineService
from evaluator.config import config
from evaluator.db import pool as db_pool
from evaluator.drift_monitor import DriftMonitor
from evaluator.drift_store import DriftStore
from evaluator.logging_config import get_logger, set_correlation_id
from ingestion.queue import AsyncIngestionBuffer, get_ingestion_buffer
from ingestion.redis_queue import RedisStreamBuffer

logger = get_logger("api.app")


def create_app(
    store: DriftStore | None = None,
    buffer: AsyncIngestionBuffer | RedisStreamBuffer | None = None,
    monitor: DriftMonitor | None = None,
) -> FastAPI:
    """Build the drift ingestion gateway.

    ``store``/``buffer``/``monitor`` may be injected (e.g. fakes) for tests;
    defaults wire up the real Postgres-backed ``DriftStore`` and a durable
    Redis Streams buffer (with graceful fallback to the in-memory queue).
    """
    store = store or DriftStore()
    if buffer is None:
        buffer = get_ingestion_buffer(config, store)
    baseline_service = DynamicBaselineService(store=store)
    monitor = monitor or DriftMonitor(store=store, baseline_service=baseline_service)

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

    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def _correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID") or uuid4().hex
        set_correlation_id(correlation_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response

    application.include_router(health_router)
    application.include_router(metrics_router)
    application.include_router(telemetry_router)
    return application


app = create_app()
