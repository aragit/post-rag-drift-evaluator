from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.app import create_app
from evaluator.config import config
from evaluator.db import pool as db_pool
from ingestion.queue import AsyncIngestionBuffer
from tests.test_api_ingestion import FakeRepo, api_client


@asynccontextmanager
async def _app_with_store():
    store = FakeRepo()
    app = create_app(store=store, buffer=AsyncIngestionBuffer())
    async with api_client(app) as client:
        yield client, app


async def test_healthz_returns_alive():
    async with _app_with_store() as (client, _):
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readyz_returns_ready_when_db_connected(postgres_available):
    if not postgres_available:
        pytest.skip("PostgreSQL is not available")

    async with _app_with_store() as (client, _):
        response = await client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["db"] == "connected"


async def test_readyz_returns_503_when_db_disconnected():
    async with _app_with_store() as (client, _):
        with patch.object(
            db_pool, "get_pool", side_effect=RuntimeError("simulated outage")
        ):
            response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["db"] == "disconnected"


async def test_readyz_returns_503_when_buffer_not_running(postgres_available):
    if not postgres_available:
        pytest.skip("PostgreSQL is not available")

    store = FakeRepo()
    buffer_mock = MagicMock()
    buffer_mock.is_running = False
    buffer_mock.buffer_type = "mock"
    buffer_mock.pending = 0
    buffer_mock.start_worker = AsyncMock()
    buffer_mock.stop_worker = AsyncMock()
    app = create_app(store=store, buffer=buffer_mock)

    async with api_client(app) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503


def test_config_loads_app_settings():
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 8000
    assert config.LOG_LEVEL == "INFO"
    assert config.WORKER_BATCH_SIZE == 50
    assert config.WORKER_FLUSH_INTERVAL == 5.0


def test_config_log_level_int():
    assert config.LOG_LEVEL_INT == 20


def test_config_database_url_derived_from_pg_params(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "myuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "mypass")
    monkeypatch.setenv("POSTGRES_DB", "mydb")
    monkeypatch.setenv("POSTGRES_HOST", "myhost")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from evaluator.config import EvaluatorConfig

    cfg = EvaluatorConfig(_env_file=None)
    assert cfg.DATABASE_URL == "postgresql://myuser:mypass@myhost:5433/mydb"


def test_config_explicit_database_url_takes_precedence(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://override:secret@overridehost:9999/db"
    )
    monkeypatch.delenv("POSTGRES_USER", raising=False)

    from evaluator.config import EvaluatorConfig

    cfg = EvaluatorConfig(_env_file=None)
    assert cfg.DATABASE_URL == "postgresql://override:secret@overridehost:9999/db"
