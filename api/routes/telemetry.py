from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from api.middleware.auth import verify_api_key
from api.middleware.rate_limit import rate_limit
from api.metrics import (
    DRIFT_SCORE_GAUGE,
    EVALUATION_LATENCY_SECONDS,
    FRAME_INGESTION_TOTAL,
)
from evaluator.schemas.telemetry import RAGEvaluationFrame

router = APIRouter(
    prefix="/v1/telemetry",
    tags=["telemetry"],
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)

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
    FRAME_INGESTION_TOTAL.labels(
        status="accepted", buffer_type=buffer.buffer_type
    ).inc(len(frames))
    return {"status": "accepted", "count": len(frames)}


@router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_telemetry(
    request: Request,
    payload: EvaluateRequestPayload,
) -> Dict[str, Any]:
    """Run multi-modal drift evaluation over the provided frame windows.

    Baseline frames come from ``baseline_frames`` when supplied, otherwise
    from the most recent persisted window referenced by
    ``baseline_batch_id``.  When neither is provided, a sliding-window
    baseline is fetched dynamically and auto-calibrated thresholds are
    computed.
    """
    store = request.app.state.drift_store
    monitor = request.app.state.drift_monitor

    if payload.baseline_frames:
        baseline_frames = payload.baseline_frames
    elif payload.baseline_batch_id:
        baseline_frames = await store.get_recent_frames(limit=DEFAULT_BASELINE_LIMIT)
    else:
        baseline_frames = None

    start = time.monotonic()
    try:
        result = await monitor.evaluate_frames(baseline_frames, payload.current_frames)
    except Exception:
        EVALUATION_LATENCY_SECONDS.labels(status="error").observe(time.monotonic() - start)
        raise
    EVALUATION_LATENCY_SECONDS.labels(status="success").observe(time.monotonic() - start)

    _update_drift_gauges(result)
    return result


def _update_drift_gauges(result: Dict[str, Any]) -> None:
    vector = result.get("vector_drift", {})
    graph = result.get("graph_drift", {})
    swarm = result.get("swarm_drift", {})

    DRIFT_SCORE_GAUGE.labels(metric_type="vector_jsd").set(
        float(vector.get("js_divergence") or 0.0)
    )
    DRIFT_SCORE_GAUGE.labels(metric_type="vector_mmd").set(
        float(vector.get("mmd_score") or 0.0)
    )
    DRIFT_SCORE_GAUGE.labels(metric_type="graph_spectral").set(
        float(graph.get("spectral_distance") or 0.0)
    )
    DRIFT_SCORE_GAUGE.labels(metric_type="swarm_entropy").set(
        float(swarm.get("transition_entropy_delta") or 0.0)
    )
