from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChangeEvent:
    """Represents a system change detected in the evaluation history.

    A change event is emitted when consecutive runs differ in
    ``system_version``, ``metadata``, or ``system_info`` fields.
    """

    change_id: str | None = None
    timestamp: float = 0.0
    run_id: str = ""
    change_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.change_id is None:
            import uuid

            self.change_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{self.run_id}:{self.timestamp}:{self.change_type}",
                )
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "change_type": self.change_type,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeEvent:
        return cls(
            change_id=data.get("change_id"),
            timestamp=data.get("timestamp", 0.0),
            run_id=data.get("run_id", ""),
            change_type=data.get("change_type", ""),
            details=dict(data.get("details", {})),
        )


@dataclass
class CausalFactor:
    """A candidate cause ranked by its attribution score."""

    factor_name: str
    score: float
    related_run_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "score": self.score,
            "related_run_ids": list(self.related_run_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalFactor:
        return cls(
            factor_name=data["factor_name"],
            score=data["score"],
            related_run_ids=list(data.get("related_run_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CausalAttribution:
    """Full attribution result for a single drift event."""

    drift_event_id: str
    metric_name: str
    factors: list[CausalFactor] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    attribution_id: str | None = None

    def __post_init__(self) -> None:
        if self.attribution_id is None:
            self.attribution_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribution_id": self.attribution_id,
            "drift_event_id": self.drift_event_id,
            "metric_name": self.metric_name,
            "factors": [f.to_dict() for f in self.factors],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalAttribution:
        return cls(
            attribution_id=data.get("attribution_id"),
            drift_event_id=data["drift_event_id"],
            metric_name=data["metric_name"],
            factors=[CausalFactor.from_dict(f) for f in data.get("factors", [])],
            confidence=data.get("confidence", 0.0),
            metadata=dict(data.get("metadata", {})),
        )
