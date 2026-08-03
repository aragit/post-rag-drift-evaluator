from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.optimization.actions import generate_actions
from evaluator.optimization.models import (
    OptimizationPlan,
    OptimizationRecommendation,
)
from evaluator.optimization.scorer import score_actions

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution
    from evaluator.counterfactual.models import CounterfactualResult
    from evaluator.temporal.models import DriftEvent


def generate_optimization_plan(
    drift_event: DriftEvent,
    attribution: CausalAttribution,
    counterfactuals: list[CounterfactualResult],
) -> OptimizationPlan:
    """Generate a ranked, actionable optimization plan for a drift event.

    Full pipeline:
    1. Generate remediation actions from causal factors.
    2. Score actions using counterfactual impact estimates.
    3. Rank recommendations by expected improvement.
    4. Build a human-readable summary.

    Args:
        drift_event: The drift event that triggered the analysis.
        attribution: The causal attribution for this drift event.
        counterfactuals: Counterfactual simulation results for the drift event.

    Returns:
        An :class:`OptimizationPlan` with ranked recommendations.
    """
    actions = generate_actions(attribution)
    recommendations = score_actions(actions, counterfactuals)

    summary = _build_summary(recommendations, drift_event)

    plan = OptimizationPlan(
        drift_event_id=drift_event.event_id or "",
        recommendations=recommendations,
        summary=summary,
        metadata={
            "metric_name": attribution.metric_name,
            "num_actions": len(actions),
            "num_recommendations": len(recommendations),
            "drift_magnitude": drift_event.magnitude,
        },
    )

    return plan


def _build_summary(
    recommendations: list[OptimizationRecommendation],
    drift_event: DriftEvent,
) -> str:
    """Build a human-readable summary of the top recommendation."""
    if not recommendations:
        return (
            f"No actionable recommendations for drift event "
            f"{drift_event.event_id}. No causal factors identified."
        )

    top = recommendations[0]
    action = top.action
    improvement = round(top.expected_improvement, 4)

    parts = [f"Top recommendation: {action.description.lower()}"]

    if action.target_run_id:
        parts.append(f"in run {action.target_run_id}")

    parts.append(f"(expected improvement: {improvement})")

    if top.confidence > 0:
        parts.append(f"confidence: {top.confidence}")

    return " ".join(parts)
