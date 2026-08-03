from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_store
from api.schemas import DriftRequest, DriftResponse
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.drift_detection import detect_drift_from_store

router = APIRouter(tags=["drift"])


@router.post("/drift", response_model=DriftResponse)
def detect_drift(
    request: DriftRequest,
    store: JSONHistoryStore = Depends(get_store),
) -> DriftResponse:
    """Detect drift events from the history store.

    Uses a sliding-window mean-shift algorithm on the specified metric.
    """
    events = detect_drift_from_store(
        store=store,
        metric_name=request.metric_name,
        window_size=request.window_size,
        threshold=request.threshold,
    )

    if not events:
        return DriftResponse(
            events=[],
            metric_name=request.metric_name,
            count=0,
        )

    serialized = [e.to_dict() for e in events]

    return DriftResponse(
        events=serialized,
        metric_name=request.metric_name,
        count=len(events),
    )
