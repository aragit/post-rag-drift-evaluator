from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intervention:
    """A hypothetical action applied to a change event.

    Used to simulate "what if this change had not happened?"
    Each intervention targets a specific change (by ``change_id``)
    and specifies an action.

    Supported actions:
        - ``"remove"``: revert the change entirely (revert to pre-change state).
        - ``"modify"``: override specific metadata fields on the change.
    """

    action: str = "remove"
    change_id: str = ""
    override_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    intervention_id: str | None = None

    def __post_init__(self) -> None:
        if self.intervention_id is None:
            self.intervention_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "change_id": self.change_id,
            "action": self.action,
            "override_metadata": dict(self.override_metadata),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Intervention:
        return cls(
            intervention_id=data.get("intervention_id"),
            change_id=data.get("change_id", ""),
            action=data.get("action", "remove"),
            override_metadata=dict(data.get("override_metadata", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CounterfactualScenario:
    """A 'what-if' hypothesis built from one or more interventions.

    Each scenario is attached to a specific drift event and describes
    the interventions whose hypothetical application is being tested.
    """

    drift_event_id: str
    interventions: list[Intervention] = field(default_factory=list)
    description: str = ""
    scenario_id: str | None = None

    def __post_init__(self) -> None:
        if self.scenario_id is None:
            self.scenario_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "drift_event_id": self.drift_event_id,
            "interventions": [i.to_dict() for i in self.interventions],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CounterfactualScenario:
        return cls(
            scenario_id=data.get("scenario_id"),
            drift_event_id=data["drift_event_id"],
            interventions=[
                Intervention.from_dict(i) for i in data.get("interventions", [])
            ],
            description=data.get("description", ""),
        )


@dataclass
class CounterfactualResult:
    """Output of a single counterfactual simulation run.

    Attributes:
        scenario_id: Identifier of the scenario that produced this result.
        original_metric: The observed metric value (drift magnitude)
            before counterfactual adjustment.
        counterfactual_metric: The estimated metric value after the
            intervention(s) are applied.
        delta: ``original_metric - counterfactual_metric`` — a positive
            delta indicates the change contributed to the drift.
        confidence: Confidence in the counterfactual estimate, derived
            from the attribution score(s).
        metadata: Additional diagnostic information.
    """

    scenario_id: str
    original_metric: float
    counterfactual_metric: float
    delta: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    result_id: str | None = None

    def __post_init__(self) -> None:
        if self.result_id is None:
            self.result_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "scenario_id": self.scenario_id,
            "original_metric": self.original_metric,
            "counterfactual_metric": self.counterfactual_metric,
            "delta": self.delta,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CounterfactualResult:
        return cls(
            result_id=data.get("result_id"),
            scenario_id=data["scenario_id"],
            original_metric=data["original_metric"],
            counterfactual_metric=data["counterfactual_metric"],
            delta=data["delta"],
            confidence=data["confidence"],
            metadata=dict(data.get("metadata", {})),
        )
