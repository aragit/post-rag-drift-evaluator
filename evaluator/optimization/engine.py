"""Track-aware optimization engine for selecting remediation actions.

Dispatches targeted interventions based on the drift track
(``"retrieval"`` vs ``"generation"``) and validates counterfactual
confidence before selecting an action.
"""

from __future__ import annotations

from typing import Any

from evaluator.counterfactual.models import CounterfactualResult
from evaluator.optimization.models import (
    OptimizationAction,
)
from evaluator.temporal.models import DriftEvent

RETRIEVAL_ACTIONS = {"adjust_top_k", "update_reranker_cutoff", "fallback_embedding"}
GENERATION_ACTIONS = {"adjust_temperature", "switch_prompt_template", "fallback_model"}


class OptimizationEngine:
    """Selects optimal remediation actions based on drift track.

    Attributes:
        min_confidence: Minimum counterfactual confidence required
            to consider an action (default 0.70).
        default_action_type: Fallback action type for unified track.
    """

    def __init__(
        self,
        min_confidence: float = 0.70,
        default_action_type: str = "rollback_config",
    ):
        self.min_confidence = min_confidence
        self.default_action_type = default_action_type

    def select_action(
        self,
        drift_event: DriftEvent,
        counterfactual_results: list[CounterfactualResult],
        attribution_factors: list[Any] | None = None,
    ) -> OptimizationAction | None:
        """Select the best action based on drift track and counterfactual impact.

        Args:
            drift_event: The drift event that triggered optimization.
            counterfactual_results: Results from counterfactual analysis.
            attribution_factors: Optional causal factors for additional context.

        Returns:
            An :class:`OptimizationAction` if one meets the confidence
            threshold, otherwise ``None``.
        """
        if not counterfactual_results:
            return None

        track = drift_event.metadata.get("track", "unified")

        valid_results = [
            r for r in counterfactual_results if r.confidence >= self.min_confidence
        ]

        if not valid_results:
            return None

        best = max(valid_results, key=lambda r: r.delta)

        action_type = self._select_action_type(track, best)

        change_id = ""
        if best.metadata.get("change_ids"):
            change_id = str(best.metadata["change_ids"][0])

        return OptimizationAction(
            action_type=action_type,
            target_run_id=best.metadata.get("run_id", ""),
            change_id=change_id,
            description=self._build_description(track, action_type, best),
            metadata={
                "counterfactual_delta": best.delta,
                "counterfactual_confidence": best.confidence,
                "track": track,
                "source": "optimization_engine",
            },
        )

    def _select_action_type(
        self,
        track: str,
        best_result: CounterfactualResult,
    ) -> str:
        """Determine the appropriate action type based on drift track."""
        if track == "retrieval":
            return "adjust_top_k"
        elif track == "generation":
            return "adjust_temperature"
        else:
            return self.default_action_type

    def _build_description(
        self,
        track: str,
        action_type: str,
        result: CounterfactualResult,
    ) -> str:
        """Build a human-readable description for the selected action."""
        improvement = result.delta
        if track == "retrieval":
            return (
                f"Adjust retrieval parameters to reduce drift "
                f"(expected improvement: {improvement:.4f})"
            )
        elif track == "generation":
            return (
                f"Adjust generation parameters to reduce drift "
                f"(expected improvement: {improvement:.4f})"
            )
        else:
            return (
                f"Rollback configuration to reduce drift "
                f"(expected improvement: {improvement:.4f})"
            )

    def generate_plan(
        self,
        drift_event: DriftEvent,
        counterfactual_results: list[CounterfactualResult],
        attribution_factors: list[Any] | None = None,
    ) -> list[OptimizationAction]:
        """Generate a ranked list of candidate actions (no policy filtering).

        Unlike :meth:`select_action`, this returns all viable actions
        ranked by expected improvement, allowing the runner to apply
        policy filtering separately.
        """
        track = drift_event.metadata.get("track", "unified")

        valid_results = [
            r for r in counterfactual_results if r.confidence >= self.min_confidence
        ]

        actions: list[OptimizationAction] = []
        for result in sorted(valid_results, key=lambda r: r.delta, reverse=True):
            action_type = self._select_action_type(track, result)
            change_id = ""
            if result.metadata.get("change_ids"):
                change_id = str(result.metadata["change_ids"][0])

            action = OptimizationAction(
                action_type=action_type,
                target_run_id=result.metadata.get("run_id", ""),
                change_id=change_id,
                description=self._build_description(track, action_type, result),
                metadata={
                    "counterfactual_delta": result.delta,
                    "counterfactual_confidence": result.confidence,
                    "track": track,
                    "source": "optimization_engine",
                },
            )
            actions.append(action)

        return actions
