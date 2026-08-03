from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.causal.models import CausalAttribution
from evaluator.counterfactual.models import (
    CounterfactualScenario,
    Intervention,
)

if TYPE_CHECKING:
    pass

_MAX_TOP_K = 3


def build_counterfactual_scenarios(
    attribution: CausalAttribution,
    top_k: int = _MAX_TOP_K,
) -> list[CounterfactualScenario]:
    """Build counterfactual scenarios from a causal attribution.

    Converts the top-ranked causal factors into individual
    "remove this change" scenarios, plus an optional combined
    scenario that removes *all* top factors simultaneously.

    Scenarios are deterministic: given the same attribution,
    the same scenarios (with the same IDs) are produced.

    Args:
        attribution: The output of :func:`attribute_drift`.
        top_k: Maximum number of individual factors to include
            as separate scenarios (default 3).

    Returns:
        A list of :class:`CounterfactualScenario` objects.
    """
    if not attribution.factors:
        return []

    factors = sorted(attribution.factors, key=lambda f: f.score, reverse=True)[:top_k]

    scenarios: list[CounterfactualScenario] = []

    for factor in factors:
        change_id = str(factor.metadata.get("change_id", ""))
        intervention = Intervention(
            action="remove",
            change_id=change_id,
            metadata={
                "factor_name": factor.factor_name,
                "factor_score": factor.score,
                "related_run_ids": list(factor.related_run_ids),
            },
        )
        scenario = CounterfactualScenario(
            drift_event_id=attribution.drift_event_id,
            interventions=[intervention],
            description=(
                f"Remove change '{factor.factor_name}' "
                f"(score={factor.score:.4f}, change_id={change_id})"
            ),
        )
        scenarios.append(scenario)

    if len(factors) > 1:
        interventions = []
        for factor in factors:
            change_id = str(factor.metadata.get("change_id", ""))
            interventions.append(
                Intervention(
                    action="remove",
                    change_id=change_id,
                    metadata={
                        "factor_name": factor.factor_name,
                        "factor_score": factor.score,
                    },
                )
            )
        combined = CounterfactualScenario(
            drift_event_id=attribution.drift_event_id,
            interventions=interventions,
            description=(f"Remove all {len(interventions)} top-ranked changes"),
        )
        scenarios.append(combined)

    return scenarios
