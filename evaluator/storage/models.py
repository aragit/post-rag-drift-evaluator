from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from evaluator.metrics.results import MetricResult


@dataclass
class EvaluationRecord:
    """Persistent record of one evaluation session.

    Wraps a collection of :class:`MetricResult` objects so that
    historical analysis can group results by run, time window, or
    metric type.
    """

    run_id: str
    metrics: list[MetricResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    system_version: str | None = None
    record_id: str | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.record_id is None:
            self.record_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Nested ``MetricResult`` objects are serialized via their own
        ``to_dict()`` to preserve subclass type information.
        """
        return {
            "record_id": self.record_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "system_version": self.system_version,
            "metrics": [
                m.to_dict() if isinstance(m, MetricResult) else dict(m)
                for m in self.metrics
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRecord:
        """Deserialize from a dict created by :meth:`to_dict`."""
        metrics_data = data.get("metrics", [])
        metrics: list[MetricResult] = [
            MetricResult.from_dict(m) if isinstance(m, dict) else m
            for m in metrics_data
        ]
        return cls(
            record_id=data.get("record_id"),
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp"),
            system_version=data.get("system_version"),
            metrics=metrics,
            metadata=data.get("metadata", {}),
        )
