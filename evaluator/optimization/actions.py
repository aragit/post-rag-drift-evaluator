from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.optimization.models import OptimizationAction

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution

_CHANGE_TYPE_TO_ACTION: dict[str, str] = {
    "model_update": "revert_model",
    "config_change": "rollback_config",
    "version_change": "restore_version",
}

_ACTION_DESCRIPTIONS: dict[str, str] = {
    "revert_model": "Revert model change",
    "rollback_config": "Rollback configuration change",
    "restore_version": "Restore previous version",
}


def generate_actions(
    attribution: CausalAttribution,
) -> list[OptimizationAction]:
    """Generate concrete remediation actions from causal factors.

    Maps each top-ranked causal factor to an actionable remediation
    step based on the change type:

    - ``model_update`` → ``revert_model``
    - ``config_change`` → ``rollback_config``
    - ``version_change`` → ``restore_version``
    - unknown → ``rollback_config`` (safe default)

    Args:
        attribution: The causal attribution result for a drift event.

    Returns:
        A list of :class:`OptimizationAction` objects, one per factor.
    """
    actions: list[OptimizationAction] = []

    factors = sorted(
        attribution.factors, key=lambda f: f.score, reverse=True
    )

    for factor in factors:
        action_type = _CHANGE_TYPE_TO_ACTION.get(
            factor.factor_name, "rollback_config"
        )
        description = _ACTION_DESCRIPTIONS.get(
            action_type, "Rollback configuration change"
        )
        change_id = str(factor.metadata.get("change_id", ""))
        target_run_id = (
            factor.related_run_ids[0] if factor.related_run_ids else ""
        )

        action = OptimizationAction(
            action_type=action_type,
            target_run_id=target_run_id,
            change_id=change_id,
            description=description,
            metadata={
                "factor_name": factor.factor_name,
                "factor_score": factor.score,
                "change_type": factor.factor_name,
            },
        )
        actions.append(action)

    return actions
