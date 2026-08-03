from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np


@dataclass
class EmbeddingBatch:
    """A batch of embedding vectors with a timestamp.

    Attributes:
        vectors: 2-D array of shape ``(n_samples, dim)``.
        timestamp: When the batch was collected.
        metadata: Optional context (run_ids, model name, etc.).
        track: Which embedding track this batch belongs to:
            ``"retrieval"`` (chunks/documents), ``"generation"``
            (LLM answers), or ``"unified"`` (combined/default).
    """

    vectors: np.ndarray
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    track: Literal["retrieval", "generation", "unified"] = "unified"

    def __post_init__(self) -> None:
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.reshape(1, -1)


@dataclass
class LatentDriftResult:
    """Result of a latent drift computation.

    Attributes:
        drift_score: Divergence score between baseline and current
            embedding distributions.  Bounded ``[0, 1]``.
        drift_detected: ``True`` when ``drift_score > threshold``.
        threshold: The configured drift threshold.
        n_samples_baseline: Number of baseline embedding vectors.
        n_samples_current: Number of current embedding vectors.
        metric_used: Which distance metric was used (``"mmd"``,
            ``"swd"``, or ``"jsd"``).
        track: The embedding track this result pertains to.
        metric_breakdown: Side-by-side metric scores if multiple
            metrics were computed (e.g. on each track independently).
        metadata: Additional diagnostic information (PCA explained
            variance, KDE bandwidth, etc.).
    """

    drift_score: float
    drift_detected: bool
    threshold: float
    n_samples_baseline: int
    n_samples_current: int
    metric_used: str = "jsd"
    track: str = "unified"
    metric_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
