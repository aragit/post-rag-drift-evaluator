from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationAction:
    """A concrete action that can be taken to address a detected drift.

    Each action targets a specific change event (identified by
    ``change_id``) and proposes a remediation step.

    Attributes:
        action_type: The category of action.  Currently supported:
            - ``"revert_model"`` — undo a model version/model update
            - ``"rollback_config"`` — undo a configuration parameter change
            - ``"restore_version"`` — undo a version roll-forward
        target_run_id: The run_id of the record affected by the change.
        change_id: Identifier of the change event this action addresses.
        description: Human-readable description of the action.
        metadata: Additional context (e.g. old/new values).
    """

    action_type: str
    target_run_id: str
    change_id: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None

    def __post_init__(self) -> None:
        if self.action_id is None:
            self.action_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "target_run_id": self.target_run_id,
            "change_id": self.change_id,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationAction:
        return cls(
            action_id=data.get("action_id"),
            action_type=data.get("action_type", ""),
            target_run_id=data.get("target_run_id", ""),
            change_id=data.get("change_id", ""),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OptimizationRecommendation:
    """A ranked recommendation tying an action to its expected impact.

    Attributes:
        action: The :class:`OptimizationAction` to execute.
        expected_improvement: How much the drift metric is expected
            to improve (reduce) if this action is taken.
        confidence: Confidence in the recommendation, derived from
            the counterfactual confidence.
        priority: 1-based priority rank (1 = highest priority).
        metadata: Additional diagnostic fields.
    """

    action: OptimizationAction
    expected_improvement: float
    confidence: float
    priority: int
    metadata: dict[str, Any] = field(default_factory=dict)
    recommendation_id: str | None = None

    def __post_init__(self) -> None:
        if self.recommendation_id is None:
            self.recommendation_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "action": self.action.to_dict(),
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationRecommendation:
        return cls(
            recommendation_id=data.get("recommendation_id"),
            action=OptimizationAction.from_dict(data["action"]),
            expected_improvement=data["expected_improvement"],
            confidence=data["confidence"],
            priority=data["priority"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OptimizationPlan:
    """A complete optimization plan for addressing a drift event.

    Attributes:
        drift_event_id: The drift event this plan addresses.
        recommendations: Ranked list of actionable recommendations.
        summary: A human-readable summary of the top recommendation.
        metadata: Additional plan-level metadata.
    """

    drift_event_id: str
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.plan_id is None:
            self.plan_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "drift_event_id": self.drift_event_id,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationPlan:
        return cls(
            plan_id=data.get("plan_id"),
            drift_event_id=data["drift_event_id"],
            recommendations=[
                OptimizationRecommendation.from_dict(r)
                for r in data.get("recommendations", [])
            ],
            summary=data.get("summary", ""),
            metadata=dict(data.get("metadata", {})),
        )
