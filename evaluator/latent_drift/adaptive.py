"""Adaptive threshold management for latent drift detection.

Uses a rolling window of drift scores and computes a dynamic threshold
based on the z-score (mean + z * std).  The threshold is clamped to
user-configured absolute bounds to prevent extreme values.
"""

from __future__ import annotations

import math
from collections import deque


class AdaptiveThresholdManager:
    """Manages dynamic drift thresholds using rolling statistics.

    Maintains a rolling window of observed drift scores and computes
    a threshold as ``mean(window) + sensitivity_z * std(window)``.

    The threshold is bounded to ``[min_threshold, max_threshold]``
    to ensure stable behavior across low-variance and high-variance
    regimes.

    Attributes:
        base_threshold: Initial/fallback threshold when window is empty.
        sensitivity_z: Z-score multiplier (higher = more sensitive).
        window_size: Number of historical scores to consider.
        min_threshold: Absolute minimum threshold.
        max_threshold: Absolute maximum threshold.
    """

    def __init__(
        self,
        base_threshold: float = 0.15,
        sensitivity_z: float = 2.0,
        window_size: int = 30,
        min_threshold: float = 0.05,
        max_threshold: float = 0.50,
    ):
        self.base_threshold = base_threshold
        self.sensitivity_z = sensitivity_z
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._scores: deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> None:
        """Append a drift score to the rolling window.

        Args:
            score: The latest drift score (0–1).
        """
        self._scores.append(float(score))

    def get_threshold(self) -> float:
        """Compute the current adaptive threshold.

        Returns:
            Dynamic threshold ``mean + z * std``, clamped to
            ``[min_threshold, max_threshold]``.

            If fewer than 2 scores are available, returns
            ``base_threshold``.
        """
        if len(self._scores) < 2:
            return self.base_threshold

        n = len(self._scores)
        mean_score = sum(self._scores) / n
        variance = sum((s - mean_score) ** 2 for s in self._scores) / (n - 1)
        std_score = math.sqrt(variance)

        threshold = mean_score + self.sensitivity_z * std_score

        # Clamp to absolute bounds
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))

        return threshold

    def reset(self) -> None:
        """Clear all historical scores."""
        self._scores.clear()

    @property
    def window(self) -> list[float]:
        """Return a copy of the current score window."""
        return list(self._scores)
