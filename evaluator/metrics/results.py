from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class DriftResult(MetricResult):
    """Result of a drift comparison between two runs."""

    baseline_run_id: str | None = None
    current_run_id: str | None = None


@dataclass
class QualityResult(MetricResult):
    """Result of a quality metric on a single run."""

    run_id: str | None = None
