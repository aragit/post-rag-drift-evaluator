from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

import redis
import redis.asyncio

from api.metrics import (
    DB_BATCH_WRITE_LATENCY_SECONDS,
    INGESTION_BUFFER_DEPTH,
)
from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import RAGEvaluationFrame

logger = get_logger("RedisStreamBuffer")


class RedisStreamBuffer:
    """Durable async ingestion buffer backed by Redis Streams.

    Frames are written to a Redis Stream via ``XADD`` and consumed by a
    background worker using consumer-group semantics (``XREADGROUP``).
    Each successfully persisted batch is acknowledged with ``XACK``,
    guaranteeing at-least-once delivery with no frame loss across
    process restarts.
    """

    def __init__(
        self,
        redis_url: Optional[str],
        stream_key: str = "telemetry:frames:stream",
        consumer_group: str = "drift_engine_workers",
        consumer_name: str = "worker_1",
        batch_size: int = 50,
        flush_interval: float = 5.0,
        client: Optional[redis.asyncio.Redis] = None,
    ):
        self._redis_url = redis_url
        self._stream_key = stream_key
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.0, flush_interval)
        self._client: Optional[redis.asyncio.Redis] = client
        self._worker_task: Optional[asyncio.Task] = None
        self._repo: Optional[Any] = None
        self._stopped = False
        self._pending = 0

    @property
    def pending(self) -> int:
        """Approximate number of frames enqueued but not yet acknowledged."""
        return self._pending

    @property
    def is_running(self) -> bool:
        """Whether the background persistence worker is active."""
        return self._worker_task is not None and not self._worker_task.done()

    @property
    def buffer_type(self) -> str:
        return "redis_stream"

    async def _ensure_client(self) -> redis.asyncio.Redis:
        if self._client is None:
            self._client = redis.asyncio.from_url(self._redis_url)
        return self._client

    async def enqueue(self, frames: List[RAGEvaluationFrame]) -> None:
        """Serialize frames to JSON and publish them to the Redis Stream."""
        if not frames:
            return
        client = await self._ensure_client()
        pipe = client.pipeline()
        for frame in frames:
            pipe.xadd(self._stream_key, {"frame": frame.model_dump_json()})
        await pipe.execute()
        self._pending += len(frames)
        INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(self.pending)
        logger.debug(
            "Enqueued %d frames to stream %s (pending=%d).",
            len(frames),
            self._stream_key,
            self.pending,
        )

    async def start_worker(self, persistence_repo: Any) -> None:
        """Ensure the consumer group exists and start the background worker."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._repo = persistence_repo
        self._stopped = False

        await self._ensure_consumer_group()
        self._worker_task = asyncio.create_task(self._run())
        INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(self._pending)
        logger.info("Redis Streams ingestion worker started.")

    async def _ensure_consumer_group(self) -> None:
        client = await self._ensure_client()
        try:
            await client.xgroup_create(
                name=self._stream_key,
                groupname=self._consumer_group,
                id="$",
                mkstream=True,
            )
            logger.info(
                "Created consumer group '%s' on stream '%s'.",
                self._consumer_group,
                self._stream_key,
            )
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug("Consumer group '%s' already exists.", self._consumer_group)

    async def _run(self) -> None:
        client = await self._ensure_client()
        block_ms = int(self._flush_interval * 1000) or 1000

        while not self._stopped:
            try:
                response = await client.xreadgroup(
                    groupname=self._consumer_group,
                    consumername=self._consumer_name,
                    streams={self._stream_key: ">"},
                    count=self._batch_size,
                    block=block_ms,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error reading from stream: %s", exc)
                await asyncio.sleep(self._flush_interval)
                continue

            if not response:
                continue

            for _stream_name, messages in response:
                if not messages:
                    continue
                frames: List[RAGEvaluationFrame] = []
                msg_ids: List[str] = []
                for msg_id, fields in messages:
                    raw = fields.get("frame") or fields.get(b"frame")
                    if raw is None:
                        logger.warning(
                            "Stream message %s has no 'frame' field.", msg_id
                        )
                        await client.xack(
                            self._stream_key, self._consumer_group, msg_id
                        )
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        frame = RAGEvaluationFrame.model_validate_json(raw)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Failed to deserialize frame for message %s: %s",
                            msg_id,
                            exc,
                        )
                        await client.xack(
                            self._stream_key, self._consumer_group, msg_id
                        )
                        continue
                    frames.append(frame)
                    msg_ids.append(msg_id)

                if frames:
                    await self._flush(frames, msg_ids, client)

        logger.info("Redis Streams ingestion worker stopped.")

    async def _flush(
        self,
        frames: List[RAGEvaluationFrame],
        msg_ids: List[str],
        client: redis.asyncio.Redis,
    ) -> None:
        if self._repo is None or not frames:
            return
        start = time.monotonic()
        try:
            await self._repo.batch_store_frames(frames)
        except Exception as exc:  # noqa: BLE001 - keep worker alive across DB hiccups
            DB_BATCH_WRITE_LATENCY_SECONDS.observe(time.monotonic() - start)
            logger.error(
                "Failed to persist %d frames from stream: %s", len(frames), exc
            )
        else:
            DB_BATCH_WRITE_LATENCY_SECONDS.observe(time.monotonic() - start)
            await client.xack(self._stream_key, self._consumer_group, *msg_ids)
            self._pending = max(0, self._pending - len(frames))
            INGESTION_BUFFER_DEPTH.labels(buffer_type=self.buffer_type).set(
                self.pending
            )
            logger.info("Flushed and acknowledged %d frames from stream.", len(frames))

    async def stop_worker(self) -> None:
        """Signal the worker to stop, draining remaining messages first."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = None
            return
        self._stopped = True
        await self._worker_task
        self._worker_task = None

        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("Redis Streams ingestion worker flushed and stopped.")
