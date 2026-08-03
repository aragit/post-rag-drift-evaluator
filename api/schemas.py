from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DriftRequest(BaseModel):
    """Request to detect drift in the history store."""

    metric_name: str = Field(default="js_divergence", description="Metric to monitor")
    window_size: int = Field(default=3, ge=1, description="Sliding window size")
    threshold: float = Field(default=0.15, ge=0.0, description="Drift threshold")


class DriftResponse(BaseModel):
    """Response containing detected drift events."""

    events: list[dict[str, Any]] = Field(default_factory=list)
    metric_name: str = ""
    count: int = 0


class AttributionRequest(BaseModel):
    """Request to run causal attribution for a drift event.

    The ``drift_event`` is the serialized :class:`DriftEvent` (via ``to_dict``)
    returned by the ``/drift`` endpoint.
    """

    drift_event: dict[str, Any]


class AttributionResponse(BaseModel):
    """Response containing ranked causal factors."""

    attribution: dict[str, Any]
    drift_event_id: str
    metric_name: str
    num_factors: int
    confidence: float


class CounterfactualRequest(BaseModel):
    """Request to run counterfactual simulation."""

    drift_event: dict[str, Any]
    attribution: dict[str, Any]
    top_k: int = Field(default=3, ge=1)


class CounterfactualResponse(BaseModel):
    """Response containing counterfactual simulation results."""

    results: list[dict[str, Any]]
    drift_event_id: str
    count: int


class OptimizationRequest(BaseModel):
    """Request to run the full optimization pipeline."""

    metric_name: str = Field(default="js_divergence", description="Metric to monitor")
    window_size: int = Field(default=3, ge=1, description="Sliding window size")
    threshold: float = Field(default=0.15, ge=0.0, description="Drift threshold")
    top_k: int = Field(default=3, ge=1)


class OptimizationResponse(BaseModel):
    """Response containing the full optimization plan."""

    plan: dict[str, Any]
    drift_events_found: int
    attribution_factors: int
    counterfactual_results: int
    recommendations: int
