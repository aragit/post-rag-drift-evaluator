"""Policy guardrails for optimization action validation.

Provides :class:`PolicyEvaluator` which enforces safety checks before
actions are dispatched to a RAG pipeline:

1. **Cooldown Period** — identical actions must respect a minimum
   time gap (default 300 s).
2. **Max Action Flapping** — prevents toggling the same parameter
   more than *N* times within a 1-hour window (default 5).
3. **Parameter Bounds** — validates scalar parameters against hard
   safety limits (e.g. temperature ∈ [0.0, 1.0], top_k ∈ [1, 50]).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evaluator.optimization.models import OptimizationAction

COOLDOWN_PERIOD_S: float = 300.0
MAX_FLAPPING_PER_HOUR: int = 5
FLAPPING_WINDOW_S: float = 3600.0

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 1.0),
    "top_k": (1.0, 50.0),
    "reranker_cutoff": (0.0, 1.0),
    "retrieval_depth": (1.0, 20.0),
}


@dataclass
class PolicyDecision:
    """Result of a policy evaluation.

    Attributes:
        allowed: Whether the action passed all policy checks.
        reason: Human-readable explanation.
        rule_violated: Name of the rule that was violated, or ``None``.
    """

    allowed: bool
    reason: str = ""
    rule_violated: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyEvaluator:
    """Validates optimization actions against policy rules.

    Args:
        cooldown_period_s: Minimum seconds between identical actions.
        max_flapping_per_hour: Max times a parameter can be flapped
            (changed back and forth) within the flapping window.
        flapping_window_s: Time window for flapping detection (seconds).
        custom_bounds: Override the default parameter bounds.
    """

    def __init__(
        self,
        cooldown_period_s: float = COOLDOWN_PERIOD_S,
        max_flapping_per_hour: int = MAX_FLAPPING_PER_HOUR,
        flapping_window_s: float = FLAPPING_WINDOW_S,
        custom_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.cooldown_period_s = cooldown_period_s
        self.max_flapping_per_hour = max_flapping_per_hour
        self.flapping_window_s = flapping_window_s
        self.parameter_bounds = {
            **PARAMETER_BOUNDS,
            **(custom_bounds or {}),
        }

    def validate_action(
        self,
        action: OptimizationAction,
        execution_history: list[OptimizationAction],
    ) -> PolicyDecision:
        """Validate an action against all policy rules.

        Checks are evaluated in order:
        1. Parameter bounds
        2. Cooldown period
        3. Flapping protection

        The first violated rule causes the action to be rejected.

        Args:
            action: The optimization action to validate.
            execution_history: Previously executed actions (most recent first).

        Returns:
            :class:`PolicyDecision` with ``allowed`` flag.
        """
        # 1. Parameter bounds check
        bounds_result = self._check_parameter_bounds(action)
        if not bounds_result.allowed:
            return bounds_result

        # 2. Cooldown check
        cooldown_result = self._check_cooldown(action, execution_history)
        if not cooldown_result.allowed:
            return cooldown_result

        # 3. Flapping check
        flap_result = self._check_flapping(action, execution_history)
        if not flap_result.allowed:
            return flap_result

        return PolicyDecision(
            allowed=True,
            reason="Action passed all policy checks.",
        )

    def _check_parameter_bounds(self, action: OptimizationAction) -> PolicyDecision:
        """Validate scalar parameters are within hard safety limits."""
        params = action.metadata.get("params", {})

        for param_name, value in params.items():
            if param_name in self.parameter_bounds:
                lo, hi = self.parameter_bounds[param_name]
                if isinstance(value, int | float):
                    if value < lo or value > hi:
                        return PolicyDecision(
                            allowed=False,
                            reason=(
                                f"Parameter '{param_name}'={value} is outside "
                                f"bounds [{lo}, {hi}]."
                            ),
                            rule_violated="parameter_bounds",
                        )

        return PolicyDecision(allowed=True)

    def _check_cooldown(
        self,
        action: OptimizationAction,
        execution_history: list[OptimizationAction],
    ) -> PolicyDecision:
        """Reject identical actions within the cooldown window."""
        now = time.time()
        action_key = self._action_signature(action)

        for past_action in execution_history:
            past_ts = past_action.metadata.get("executed_at", now)
            if isinstance(past_ts, int | float):
                elapsed = now - past_ts
            else:
                # If no timestamp, assume it was just now
                elapsed = 0.0

            if elapsed < self.cooldown_period_s:
                if self._action_signature(past_action) == action_key:
                    return PolicyDecision(
                        allowed=False,
                        reason=(
                            f"Action '{action.action_type}' is within "
                            f"cooldown period ({elapsed:.1f}s < "
                            f"{self.cooldown_period_s}s)."
                        ),
                        rule_violated="cooldown_period",
                    )

        return PolicyDecision(allowed=True)

    def _check_flapping(
        self,
        action: OptimizationAction,
        execution_history: list[OptimizationAction],
    ) -> PolicyDecision:
        """Prevent excessive toggling of the same parameter."""
        now = time.time()
        target_run = action.target_run_id
        flapped_count = 0

        for past_action in execution_history:
            past_ts = past_action.metadata.get("executed_at", now)
            if isinstance(past_ts, int | float):
                elapsed = now - past_ts
            else:
                elapsed = 0.0

            if elapsed > self.flapping_window_s:
                continue

            if past_action.target_run_id == target_run:
                flapped_count += 1

        if flapped_count >= self.max_flapping_per_hour:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Action targeting run '{target_run}' has been "
                    f"executed {flapped_count} times in the last "
                    f"{self.flapping_window_s}s (max "
                    f"{self.max_flapping_per_hour})."
                ),
                rule_violated="max_flapping",
            )

        return PolicyDecision(allowed=True)

    @staticmethod
    def _action_signature(action: OptimizationAction) -> str:
        """Generate a signature to identify identical actions."""
        return f"{action.action_type}:{action.target_run_id}:{action.change_id}"
