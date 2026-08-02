from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional, Union

import redis

from api.metrics import (
    DB_BATCH_WRITE_LATENCY_SECONDS,
    INGESTION_BUFFER_DEPTH,
)
from evaluator.config import EvaluatorConfig, config
from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import RAGEvaluationFrame
from ingestion.redis_queue import RedisStreamBuffer

logger = get_logger("AsyncIngestionBuffer")

_STOP = object()


class AsyncIngestionBuffer:
    """Non-blocking async ingestion buffer for ``RAGEvaluationFrame`` telemetry.

    Incoming frames are pushed onto an :class:`asyncio.Queue` without
    awaiting persistence, so client HTTP responses are not tied to database
    write latency. A background worker drains the queue in batches and
    writes them through the persistence repository via
    ``repo.batch_store_frames()``.
    """

    def __init__(self, batch_size: int = 50, flush_interval: float = 5.0):
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.0, flush_interval)
        self._worker_task: Optional[asyncio.Task] = None
        self._repo: Optional[Any] = None
        self._stopped = False

    @property
    def pending(self) -> int:
        """Number of frames currently buffered but not yet persisted."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the background persistence worker is active."""
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def buffer_type(self) -> str:
        return "memory"

    async def enqueue(self, frames: List[RAGEvaluationFrame]) -> None:
        """Push frames onto the buffer. Returns without awaiting persistence."""
        for frame in frames:
            self._queue.put_nowait(frame)
        INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(self.pending)
        logger.debug("Enqueued %d frames (pending=%d).", len(frames), self.pending)

    async def start_worker(self, persistence_repo: Any) -> None:
        """Spawn the background persistence worker against ``persistence_repo``."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._repo = persistence_repo
        self._stopped = False
        self._worker_task = asyncio.create_task(self._run())
        INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(0)
        logger.info("Ingestion worker started.")

    async def _run(self) -> None:
        while not self._stopped:
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval
                )
            except asyncio.TimeoutError:
                continue
            batch: List[RAGEvaluationFrame] = []
            if first is _STOP:
                self._stopped = True
            else:
                batch.append(first)
                while len(batch) < self._batch_size:
                    try:
                        item = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is _STOP:
                        self._stopped = True
                        break
                    batch.append(item)
            if batch:
                await self._flush(batch)
        logger.info("Ingestion worker stopped.")

    async def _flush(self, frames: List[RAGEvaluationFrame]) -> None:
        if self._repo is None or not frames:
            return
        start = time.monotonic()
        try:
            await self._repo.batch_store_frames(frames)
        except Exception as exc:  # noqa: BLE001 - keep worker alive across DB hiccups
            DB_BATCH_WRITE_LATENCY_SECONDS.observe(time.monotonic() - start)
            logger.error("Failed to persist %d frames: %s", len(frames), exc)
        else:
            DB_BATCH_WRITE_LATENCY_SECONDS.observe(time.monotonic() - start)
            INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(self.pending)
            logger.info("Flushed %d frames to persistence.", len(frames))

    async def stop_worker(self) -> None:
        """Signal the worker to stop, flushing all remaining buffered frames."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = None
            return
        await self._queue.put(_STOP)
        await self._worker_task
        self._worker_task = None
        logger.info("Ingestion worker flushed remaining frames and stopped.")


def _make_memory_buffer(
    settings: EvaluatorConfig,
) -> AsyncIngestionBuffer:
    return AsyncIngestionBuffer(
        batch_size=settings.WORKER_BATCH_SIZE,
        flush_interval=settings.WORKER_FLUSH_INTERVAL,
    )


def get_ingestion_buffer(
    settings: Optional[EvaluatorConfig] = None,
    repo: Any = None,
) -> Union[RedisStreamBuffer, AsyncIngestionBuffer]:
    """Return a durable Redis Streams buffer when Redis is reachable.

    Falls back to the in-memory :class:`AsyncIngestionBuffer` when
    ``settings.REDIS_URL`` is ``None`` or the Redis server cannot be
    reached.
    """
    settings = settings or config

    if not settings.REDIS_URL:
        logger.info("REDIS_URL not set; using in-memory AsyncIngestionBuffer.")
        return _make_memory_buffer(settings)

    try:
        checker = redis.Redis.from_url(
            settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2
        )
        checker.ping()
        checker.close()
    except Exception as exc:  # noqa: BLE001 - graceful fallback
        logger.warning(
            "Redis ping failed (%s); falling back to AsyncIngestionBuffer.",
            exc,
        )
        return _make_memory_buffer(settings)

    return RedisStreamBuffer(
        redis_url=settings.REDIS_URL,
        stream_key=settings.REDIS_STREAM_KEY,
        consumer_group=settings.REDIS_CONSUMER_GROUP,
        consumer_name=settings.REDIS_CONSUMER_NAME,
        batch_size=settings.WORKER_BATCH_SIZE,
        flush_interval=settings.WORKER_FLUSH_INTERVAL,
    )
