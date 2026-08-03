from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.schemas import AttributionRequest, AttributionResponse
from evaluator.causal.attribution import attribute_drift
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.models import DriftEvent

router = APIRouter(tags=["attribution"])


@router.post("/attribution", response_model=AttributionResponse)
def get_attribution(
    request: AttributionRequest,
    store: JSONHistoryStore = Depends(get_store),
) -> AttributionResponse:
    """Run causal attribution for a given drift event.

    Accepts the serialized :class:`DriftEvent` returned by the ``/drift``
    endpoint and returns ranked causal factors.
    """
    try:
        drift_event = DriftEvent.from_dict(request.drift_event)
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid drift_event payload: {exc}",
        )

    try:
        attribution = attribute_drift(drift_event, store)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Attribution failed: {exc}",
        )

    return AttributionResponse(
        attribution=attribution.to_dict(),
        drift_event_id=attribution.drift_event_id,
        metric_name=attribution.metric_name,
        num_factors=len(attribution.factors),
        confidence=attribution.confidence,
    )
