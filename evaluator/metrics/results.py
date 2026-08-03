from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Base result object for all metric computations.

    All metrics eventually return structured results instead of raw
    floats.  ``metadata`` is a catch-all for implementation-specific
    details (threshold used, p-value, dimension, etc.).
    """

    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict, including the class name."""
        data = asdict(self)
        data["_type"] = type(self).__name__
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Deserialize from a dict created by :meth:`to_dict`.

        Dispatches on the ``_type`` field to reconstruct the correct
        subclass (``DriftResult`` or ``QualityResult``).
        """
        type_name = data.pop("_type", None)
        if type_name == "DriftResult":
            from evaluator.metrics.results import DriftResult

            return DriftResult(**data)
        if type_name == "QualityResult":
            from evaluator.metrics.results import QualityResult

            return QualityResult(**data)
        return cls(**data)


@dataclass
class DriftResult(MetricResult):
    """Result of a drift comparison between two runs."""

    baseline_run_id: str | None = None
    current_run_id: str | None = None


@dataclass
class QualityResult(MetricResult):
    """Result of a quality metric on a single run."""

    run_id: str | None = None


def metric_result_from_json(payload: str) -> MetricResult:
    """Convenience: JSON string → MetricResult (with subclass dispatch)."""
    return MetricResult.from_dict(json.loads(payload))


def metric_result_to_json(result: MetricResult) -> str:
    """Convenience: MetricResult → JSON string."""
    return json.dumps(result.to_dict())
