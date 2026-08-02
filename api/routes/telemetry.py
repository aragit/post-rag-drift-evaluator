from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from evaluator.schemas.telemetry import RAGEvaluationFrame

router = APIRouter(prefix="/v1/telemetry", tags=["telemetry"])

DEFAULT_BASELINE_LIMIT = 100


class IngestFramesPayload(BaseModel):
    frames: List[RAGEvaluationFrame]


class EvaluateRequestPayload(BaseModel):
    baseline_batch_id: Optional[str] = None
    baseline_frames: Optional[List[RAGEvaluationFrame]] = None
    current_frames: List[RAGEvaluationFrame]


@router.post("/frames", status_code=status.HTTP_202_ACCEPTED)
async def ingest_frames(
    request: Request,
    payload: Union[IngestFramesPayload, RAGEvaluationFrame],
) -> Dict[str, Any]:
    """Ingest one or many ``RAGEvaluationFrame`` objects asynchronously.

    Frames are enqueued onto the app's ``AsyncIngestionBuffer`` and
    persisted by the background worker; the HTTP response returns
    immediately with ``202 Accepted``.
    """
    buffer = request.app.state.ingestion_buffer
    frames = payload.frames if isinstance(payload, IngestFramesPayload) else [payload]
    await buffer.enqueue(frames)
    return {"status": "accepted", "count": len(frames)}


@router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_telemetry(
    request: Request,
    payload: EvaluateRequestPayload,
) -> Dict[str, Any]:
    """Run multi-modal drift evaluation over the provided frame windows.

    Baseline frames come from ``baseline_frames`` when supplied, otherwise
    from the most recent persisted window referenced by
    ``baseline_batch_id``.
    """
    store = request.app.state.drift_store
    monitor = request.app.state.drift_monitor

    if payload.baseline_frames:
        baseline_frames = payload.baseline_frames
    elif payload.baseline_batch_id:
        baseline_frames = await store.get_recent_frames(limit=DEFAULT_BASELINE_LIMIT)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either baseline_frames or baseline_batch_id",
        )

    return await monitor.evaluate_frames(baseline_frames, payload.current_frames)
