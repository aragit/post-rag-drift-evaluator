from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftEvent:
    """Represents a detected drift over a time window.

    Produced by :func:`evaluator.temporal.drift_detection.detect_drift_events`.
    Each event captures which metric changed, when, by how much, and
    which runs contributed to the shifted window.
    """

    metric_name: str
    start_timestamp: float
    end_timestamp: float
    magnitude: float
    involved_run_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "metric_name": self.metric_name,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "magnitude": self.magnitude,
            "involved_run_ids": list(self.involved_run_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftEvent:
        """Deserialize from a dictionary created by :meth:`to_dict`."""
        return cls(
            event_id=data.get("event_id"),
            metric_name=data["metric_name"],
            start_timestamp=data["start_timestamp"],
            end_timestamp=data["end_timestamp"],
            magnitude=data["magnitude"],
            involved_run_ids=list(data.get("involved_run_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )
