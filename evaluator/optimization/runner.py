"""Closed-loop optimization runner.

Integrates counterfactual simulation, track-aware action selection,
and policy guardrails into a single execution cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.guardrails.policy import PolicyDecision, PolicyEvaluator
from evaluator.optimization.engine import OptimizationEngine
from evaluator.optimization.models import OptimizationAction
from evaluator.temporal.models import DriftEvent

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution
    from evaluator.storage import InMemoryHistoryStore

STATUS_APPROVED = "approved"
STATUS_BLOCKED_BY_GUARDRAIL = "blocked_by_guardrail"
STATUS_NO_ACTION_NEEDED = "no_action_needed"


@dataclass
class OptimizationResult:
    """Result of a single closed-loop optimization cycle.

    Attributes:
        status: ``STATUS_APPROVED``, ``STATUS_BLOCKED_BY_GUARDRAIL``,
            or ``STATUS_NO_ACTION_NEEDED``.
        action: The selected action (if any).
        policy_decision: The policy evaluation result (if action was checked).
        counterfactual_results: All counterfactual results from simulation.
        metadata: Additional diagnostic information.
    """

    status: str
    action: OptimizationAction | None = None
    policy_decision: PolicyDecision | None = None
    counterfactual_results: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action.to_dict() if self.action else None,
            "policy_decision": {
                "allowed": self.policy_decision.allowed,
                "reason": self.policy_decision.reason,
                "rule_violated": self.policy_decision.rule_violated,
            } if self.policy_decision else None,
            "counterfactual_count": len(self.counterfactual_results),
            "metadata": self.metadata,
        }


class OptimizationRunner:
    """Orchestrates the full closed-loop optimization cycle.

    Flow:
    1. Run counterfactual simulations on the in-memory store.
    2. Use :class:`OptimizationEngine` to select the best action.
    3. Validate the action with :class:`PolicyEvaluator`.
    4. Return an :class:`OptimizationResult` with the final status.

    Args:
        engine: The optimization engine for action selection.
        policy_evaluator: The policy evaluator for guardrail checks.
        execution_history: Mutable list of previously executed actions
            (for cooldown and flapping detection).
    """

    def __init__(
        self,
        engine: OptimizationEngine | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
        execution_history: list[OptimizationAction] | None = None,
    ):
        self.engine = engine or OptimizationEngine()
        self.policy_evaluator = policy_evaluator or PolicyEvaluator()
        self.execution_history: list[OptimizationAction] = execution_history or []

    def run_optimization_cycle(
        self,
        drift_event: DriftEvent,
        attribution: CausalAttribution,
        store: InMemoryHistoryStore,
        top_k: int = 3,
    ) -> OptimizationResult:
        """Run a complete optimization cycle from drift detection to action.

        Args:
            drift_event: The drift event that triggered optimization.
            attribution: The causal attribution for this drift event.
            store: The history store (in-memory preferred).
            top_k: Maximum number of counterfactual scenarios to generate.

        Returns:
            :class:`OptimizationResult` with status and action.
        """
        start_time = time.time()

        counterfactual_results = run_counterfactual_analysis(
            drift_event, attribution, store, top_k=top_k
        )

        if not counterfactual_results:
            return OptimizationResult(
                status=STATUS_NO_ACTION_NEEDED,
                counterfactual_results=counterfactual_results,
                metadata={
                    "reason": "No counterfactual results generated",
                    "cycle_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )

        action = self.engine.select_action(
            drift_event, counterfactual_results, attribution.factors
        )

        if action is None:
            return OptimizationResult(
                status=STATUS_NO_ACTION_NEEDED,
                counterfactual_results=counterfactual_results,
                metadata={
                    "reason": "No action met confidence threshold",
                    "cycle_time_ms": round((time.time() - start_time) * 1000, 2),
                    "min_confidence": self.engine.min_confidence,
                },
            )

        policy_decision = self.policy_evaluator.validate_action(
            action, self.execution_history
        )

        if not policy_decision.allowed:
            tagged_action = OptimizationAction(
                action_type=action.action_type,
                target_run_id=action.target_run_id,
                change_id=action.change_id,
                description=action.description,
                metadata={
                    **action.metadata,
                    "status": STATUS_BLOCKED_BY_GUARDRAIL,
                    "guardrail_reason": policy_decision.reason,
                    "rule_violated": policy_decision.rule_violated,
                },
            )
            self.execution_history.append(tagged_action)

            return OptimizationResult(
                status=STATUS_BLOCKED_BY_GUARDRAIL,
                action=tagged_action,
                policy_decision=policy_decision,
                counterfactual_results=counterfactual_results,
                metadata={
                    "cycle_time_ms": round((time.time() - start_time) * 1000, 2),
                },
            )

        approved_action = OptimizationAction(
            action_type=action.action_type,
            target_run_id=action.target_run_id,
            change_id=action.change_id,
            description=action.description,
            metadata={
                **action.metadata,
                "status": STATUS_APPROVED,
                "executed_at": time.time(),
            },
        )
        self.execution_history.append(approved_action)

        return OptimizationResult(
            status=STATUS_APPROVED,
            action=approved_action,
            policy_decision=policy_decision,
            counterfactual_results=counterfactual_results,
            metadata={
                "cycle_time_ms": round((time.time() - start_time) * 1000, 2),
            },
        )
