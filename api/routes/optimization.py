from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_store
from api.schemas import OptimizationRequest, OptimizationResponse
from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.optimization.optimizer import generate_optimization_plan
from evaluator.storage import JSONHistoryStore
from evaluator.temporal.drift_detection import detect_drift_from_store

router = APIRouter(tags=["optimization"])


@router.post("/optimize", response_model=OptimizationResponse)
def optimize(
    request: OptimizationRequest,
    store: JSONHistoryStore = Depends(get_store),
) -> OptimizationResponse:
    """Run the full optimization pipeline end-to-end.

    Steps:
    1. Detect drift from the history store.
    2. Run causal attribution on each drift event.
    3. Run counterfactual simulation.
    4. Generate ranked optimization recommendations.
    """
    drift_events = detect_drift_from_store(
        store=store,
        metric_name=request.metric_name,
        window_size=request.window_size,
        threshold=request.threshold,
    )

    if not drift_events:
        return OptimizationResponse(
            plan={
                "drift_event_id": "",
                "recommendations": [],
                "summary": "No drift detected — no optimization needed.",
                "metadata": {
                    "drift_events_found": 0,
                    "attribution_factors": 0,
                    "counterfactual_results": 0,
                    "recommendations": 0,
                },
            },
            drift_events_found=0,
            attribution_factors=0,
            counterfactual_results=0,
            recommendations=0,
        )

    # Use the first (most significant) drift event for full pipeline
    drift_event = drift_events[0]

    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(
        drift_event,
        attribution,
        store,
        top_k=request.top_k,
    )
    plan = generate_optimization_plan(drift_event, attribution, counterfx)

    return OptimizationResponse(
        plan=plan.to_dict(),
        drift_events_found=len(drift_events),
        attribution_factors=len(attribution.factors),
        counterfactual_results=len(counterfx),
        recommendations=len(plan.recommendations),
    )
