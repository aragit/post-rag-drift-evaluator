from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from evaluator.config import EvaluatorConfig
from ingestion.queue import AsyncIngestionBuffer, get_ingestion_buffer
from ingestion.redis_queue import RedisStreamBuffer
from tests.test_api_ingestion import FakeRepo, _frame


def _make_config(**overrides: object) -> EvaluatorConfig:
    base: dict[str, object] = {
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "test_db",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": 5432,
    }
    base.update(overrides)
    return EvaluatorConfig(_env_file=None, **base)


# -- RedisStreamBuffer tests with fakeredis --


@pytest.mark.asyncio
async def test_enqueue_publishes_frames_to_stream():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="test:frames",
        consumer_group="test_group",
        consumer_name="worker_1",
        batch_size=10,
        client=fake,
    )

    frames = [_frame(), _frame(rag_type="agentic")]
    await buffer.enqueue(frames)

    assert buffer.pending == 2
    length = await fake.xlen("test:frames")
    assert length == 2

    await fake.aclose()


@pytest.mark.asyncio
async def test_worker_persists_frames_and_acks():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    repo = FakeRepo()
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="test:frames",
        consumer_group="test_group",
        consumer_name="worker_1",
        batch_size=10,
        flush_interval=0.5,
        client=fake,
    )

    await buffer.start_worker(repo)

    frames = [_frame(), _frame(rag_type="agentic"), _frame()]
    await buffer.enqueue(frames)

    for _ in range(200):
        if repo.flush_attempts >= 1:
            break
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.6)
    await buffer.stop_worker()

    stored = [f for batch in repo.batches for f in batch]
    assert len(stored) == 3
    assert buffer.pending == 0

    pending_info = await fake.xpending("test:frames", "test_group")
    assert pending_info["pending"] == 0

    await fake.aclose()


@pytest.mark.asyncio
async def test_worker_survives_persistence_failure():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    repo = FakeRepo()
    repo.fail_batches = True
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="test:frames",
        consumer_group="test_group",
        consumer_name="worker_1",
        batch_size=10,
        flush_interval=0.5,
        client=fake,
    )

    await buffer.start_worker(repo)

    frames = [_frame()] * 2
    await buffer.enqueue(frames)

    for _ in range(200):
        if repo.flush_attempts >= 1:
            break
        await asyncio.sleep(0.05)

    assert repo.flush_attempts >= 1
    assert buffer.is_running is True

    # Messages remain in the PEL (pending) because persistence failed
    pending_info = await fake.xpending("test:frames", "test_group")
    assert pending_info["pending"] == 2

    await buffer.stop_worker()

    await fake.aclose()


@pytest.mark.asyncio
async def test_stop_worker_drains_remaining_messages():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    repo = FakeRepo()
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="test:frames",
        consumer_group="test_group",
        consumer_name="worker_1",
        batch_size=100,
        flush_interval=60.0,
        client=fake,
    )

    await buffer.start_worker(repo)

    frames = [_frame() for _ in range(3)]
    await buffer.enqueue(frames)

    await buffer.stop_worker()

    stored = [f for batch in repo.batches for f in batch]
    assert len(stored) == 3

    await fake.aclose()


@pytest.mark.asyncio
async def test_is_running_and_pending_properties():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    repo = FakeRepo()
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="test:frames",
        consumer_group="test_group",
        consumer_name="worker_1",
        batch_size=10,
        flush_interval=0.5,
        client=fake,
    )

    assert buffer.is_running is False
    assert buffer.pending == 0

    await buffer.start_worker(repo)
    assert buffer.is_running is True

    await buffer.enqueue([_frame()])
    assert buffer.pending == 1

    await buffer.stop_worker()
    assert buffer.is_running is False

    await fake.aclose()


@pytest.mark.asyncio
async def test_consumer_group_created_once():
    fake = fakeredis.aioredis.FakeRedis()
    await fake.flushall()
    repo = FakeRepo()
    buffer = RedisStreamBuffer(
        redis_url="redis://localhost:6379/0",
        stream_key="cg:frames",
        consumer_group="cg_group",
        consumer_name="worker_1",
        batch_size=10,
        client=fake,
    )

    await buffer.start_worker(repo)

    # Calling start_worker again should not raise (group already exists)
    await buffer.start_worker(repo)

    await buffer.stop_worker()
    await fake.aclose()


# -- Fallback factory tests --


def test_get_ingestion_buffer_returns_memory_when_redis_url_none():
    cfg = _make_config(REDIS_URL=None)
    buf = get_ingestion_buffer(cfg)
    assert isinstance(buf, AsyncIngestionBuffer)


def test_get_ingestion_buffer_falls_back_when_redis_unreachable():
    cfg = _make_config(REDIS_URL="redis://127.0.0.1:1")
    buf = get_ingestion_buffer(cfg)
    assert isinstance(buf, AsyncIngestionBuffer)


def test_get_ingestion_buffer_returns_redis_buffer_when_reachable(monkeypatch):
    cfg = _make_config(REDIS_URL="redis://localhost:6379/0")

    mock_client = MagicMock()
    mock_client.ping = MagicMock(return_value=True)

    def fake_from_url(url, **kwargs):
        return mock_client

    monkeypatch.setattr("redis.Redis.from_url", fake_from_url)

    buf = get_ingestion_buffer(cfg)
    assert isinstance(buf, RedisStreamBuffer)
    assert buf._redis_url == "redis://localhost:6379/0"
