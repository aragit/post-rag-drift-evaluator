from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_store
from api.schemas import CounterfactualRequest, CounterfactualResponse
from evaluator.causal.models import CausalAttribution
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.models import DriftEvent

router = APIRouter(tags=["counterfactual"])


@router.post("/counterfactual", response_model=CounterfactualResponse)
def run_counterfactual(
    request: CounterfactualRequest,
    store: JSONHistoryStore = Depends(get_store),
) -> CounterfactualResponse:
    """Run counterfactual simulation for a drift event.

    Accepts the serialized ``DriftEvent`` and ``CausalAttribution``
    (both produced by earlier pipeline stages) and returns simulation
    results — "what if this change had not occurred?"
    """
    try:
        drift_event = DriftEvent.from_dict(request.drift_event)
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid drift_event payload: {exc}",
        )

    try:
        attribution = CausalAttribution.from_dict(request.attribution)
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attribution payload: {exc}",
        )

    results = run_counterfactual_analysis(
        drift_event=drift_event,
        attribution=attribution,
        store=store,
        top_k=request.top_k,
    )

    serialized = [r.to_dict() for r in results]

    return CounterfactualResponse(
        results=serialized,
        drift_event_id=drift_event.event_id or "",
        count=len(results),
    )
