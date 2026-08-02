from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import RAGEvaluationFrame

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

    def __init__(self, batch_size: int = 50):
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._batch_size = max(1, batch_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._repo: Optional[Any] = None
        self._stopped = False

    @property
    def pending(self) -> int:
        """Number of frames currently buffered but not yet persisted."""
        return self._queue.qsize()

    async def enqueue(self, frames: List[RAGEvaluationFrame]) -> None:
        """Push frames onto the buffer. Returns without awaiting persistence."""
        for frame in frames:
            self._queue.put_nowait(frame)
        logger.debug("Enqueued %d frames (pending=%d).", len(frames), self.pending)

    async def start_worker(self, persistence_repo: Any) -> None:
        """Spawn the background persistence worker against ``persistence_repo``."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._repo = persistence_repo
        self._stopped = False
        self._worker_task = asyncio.create_task(self._run())
        logger.info("Ingestion worker started.")

    async def _run(self) -> None:
        while not self._stopped:
            first = await self._queue.get()
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
        try:
            await self._repo.batch_store_frames(frames)
        except Exception as exc:  # noqa: BLE001 - keep worker alive across DB hiccups
            logger.error("Failed to persist %d frames: %s", len(frames), exc)
        else:
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
