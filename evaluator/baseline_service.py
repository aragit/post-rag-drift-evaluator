from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from .config import config
from .drift_store import DriftStore

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_HOURS = 24
DEFAULT_K_SIGMA = 2.0
DEFAULT_BOOTSTRAP_ITERS = 20
DEFAULT_BASELINE_LIMIT = 100


class DynamicBaselineService:
    """Fetch historical frame windows and auto-calibrate drift thresholds.

    When sufficient historical frames are available, thresholds are computed as
    ``mu + k*sigma`` over bootstrap intra-baseline splits.  Below the minimum
    sample size the service signals a *fallback* so callers retain static safety
    thresholds.
    """

    def __init__(self, store: DriftStore | None = None) -> None:
        self._store = store

    async def fetch_sliding_baseline_frames(
        self,
        window_hours: int = DEFAULT_WINDOW_HOURS,
        limit: int = DEFAULT_BASELINE_LIMIT,
    ) -> list[Any]:
        """Retrieve frames for the configured sliding time window."""
        if self._store is None:
            logger.debug("No DriftStore injected; returning empty baseline.")
            return []
        return await self._store.get_frames_by_time_window(
            hours=window_hours, limit=limit
        )

    def compute_calibrated_thresholds(
        self,
        baseline_frames: list[Any],
        k_sigma: float = DEFAULT_K_SIGMA,
        iterations: int = DEFAULT_BOOTSTRAP_ITERS,
    ) -> dict[str, float]:
        """Bootstrap-calibrate thresholds from intra-baseline splits.

        Returns a dict with keys ``vector_jsd_threshold``,
        ``vector_mmd_threshold``, ``graph_spectral_threshold`` and
        ``swarm_entropy_threshold`` (only for metrics that could be computed).
        """
        min_frames = config.MIN_BASELINE_FRAMES
        if len(baseline_frames) < min_frames:
            logger.info(
                "Only %d baseline frames (< %d); signaling fallback to static thresholds.",
                len(baseline_frames),
                min_frames,
            )
            return {}

        # Lazy import to avoid circular dependency with DriftMonitor.
        from .drift_monitor import DriftMonitor

        monitor = DriftMonitor(store=self._store, notifier=None)

        jsd_values: list[float] = []
        mmd_values: list[float] = []
        spectral_values: list[float] = []
        entropy_values: list[float] = []

        actual_iters = min(iterations, len(baseline_frames) // 2)
        frames = list(baseline_frames)

        for _ in range(actual_iters):
            random.shuffle(frames)
            mid = len(frames) // 2
            split_1 = frames[:mid]
            split_2 = frames[mid:]

            vector_result = monitor._evaluate_vector_drift(split_1, split_2)
            graph_result = monitor._evaluate_graph_drift(split_1, split_2)
            swarm_result = monitor._evaluate_swarm_drift(split_1, split_2)

            jsd_values.append(vector_result["js_divergence"])
            mmd_values.append(vector_result["mmd_score"])
            spectral_values.append(graph_result["spectral_distance"])
            entropy_values.append(swarm_result["transition_entropy_delta"])

        thresholds: dict[str, float] = {}

        def _calibrate(metric: str, values: list[float]) -> None:
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            thresholds[metric] = mu + k_sigma * sigma

        if jsd_values:
            _calibrate("vector_jsd_threshold", jsd_values)
        if mmd_values:
            _calibrate("vector_mmd_threshold", mmd_values)
        if spectral_values:
            _calibrate("graph_spectral_threshold", spectral_values)
        if entropy_values:
            _calibrate("swarm_entropy_threshold", entropy_values)

        logger.info(
            "Calibrated thresholds from %d frames over %d bootstrap iterations: %s",
            len(baseline_frames),
            actual_iters,
            thresholds,
        )
        return thresholds
