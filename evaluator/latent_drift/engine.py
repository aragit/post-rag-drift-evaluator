"""Latent drift detection engine using PCA + KDE + JSD.

This module implements a statistical drift detector that operates directly
on embedding vectors.  It detects distribution-level semantic drift by:

1. Projecting embeddings into a stable latent manifold (PCA).
2. Estimating probability densities (Gaussian KDE).
3. Computing Jensen-Shannon Divergence between distributions.

Results are converted to :class:`~evaluator.temporal.models.DriftEvent`
objects so they integrate seamlessly with the existing causal attribution,
counterfactual simulation, and optimization pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluator.latent_drift.jsd import compute_jsd
from evaluator.latent_drift.kde import evaluate_density, fit_kde
from evaluator.latent_drift.pca import fit_pca, project_vectors
from evaluator.latent_drift.schemas import EmbeddingBatch, LatentDriftResult
from evaluator.temporal.models import DriftEvent

_EPSILON = 1e-12
_DEFAULT_KDE_SAMPLE_SIZE = 1000


class LatentDriftEngine:
    """Detect distribution-level drift on embedding vectors.

    The engine fits a PCA model + KDE on baseline embeddings during
    :meth:`fit`, then compares incoming embeddings during
    :meth:`compute_drift`.

    Attributes:
        threshold: Drift threshold for the JSD score.
        pca_components: Number of PCA components to retain.
        kde_sample_size: Number of grid points for KDE evaluation.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        pca_components: int = 5,
        kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
    ):
        self.threshold = threshold
        self.pca_components = pca_components
        self.kde_sample_size = kde_sample_size
        self.pca: Any = None
        self.kde_baseline: Any = None
        self._baseline_proj: np.ndarray | None = None

    def fit(self, baseline_vectors: np.ndarray) -> None:
        """Fit PCA and KDE on baseline embedding vectors.

        Args:
            baseline_vectors: 2-D array of shape ``(n_samples, dim)``.
        """
        baseline_vectors = np.atleast_2d(baseline_vectors)

        self.pca = fit_pca(baseline_vectors, n_components=self.pca_components)
        self._baseline_proj = project_vectors(self.pca, baseline_vectors)

        self.kde_baseline = fit_kde(self._baseline_proj)

    def compute_drift(
        self,
        current_vectors: np.ndarray,
    ) -> LatentDriftResult:
        """Compute latent drift between baseline and current embeddings.

        Args:
            current_vectors: 2-D array of shape ``(n_samples, dim)``.
                Must have the same dimensionality as the baseline.

        Returns:
            A :class:`LatentDriftResult` with the drift score and metadata.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self.pca is None or self.kde_baseline is None:
            raise RuntimeError(
                "LatentDriftEngine must be fitted first. Call fit() before compute_drift()."
            )

        current_vectors = np.atleast_2d(current_vectors)
        baseline_proj = self._baseline_proj if self._baseline_proj is not None else np.empty((0, 0))

        current_proj = project_vectors(self.pca, current_vectors)

        grid = _build_shared_grid(baseline_proj, current_proj)

        p_density = evaluate_density(self.kde_baseline, grid)

        try:
            kde_current = fit_kde(current_proj)
            q_density = evaluate_density(kde_current, grid)
        except (ValueError, np.linalg.LinAlgError):
            q_density = np.full_like(p_density, _EPSILON)

        drift_score = compute_jsd(p_density, q_density)

        metadata = {
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "kde_sample_size": self.kde_sample_size,
        }

        return LatentDriftResult(
            drift_score=drift_score,
            drift_detected=drift_score > self.threshold,
            threshold=self.threshold,
            n_samples_baseline=baseline_proj.shape[0],
            n_samples_current=current_proj.shape[0],
            metadata=metadata,
        )

    def fit_compute(
        self,
        baseline_vectors: np.ndarray,
        current_vectors: np.ndarray,
    ) -> LatentDriftResult:
        """Convenience: fit on baseline, then compute drift on current.

        Args:
            baseline_vectors: Baseline embedding batch.
            current_vectors: Current embedding batch.

        Returns:
            :class:`LatentDriftResult`
        """
        self.fit(baseline_vectors)
        return self.compute_drift(current_vectors)

    def to_drift_event(
        self,
        result: LatentDriftResult,
        drift_event_id: str | None = None,
    ) -> DriftEvent:
        """Convert a :class:`LatentDriftResult` to a :class:`DriftEvent`.

        This allows latent drift results to feed into the existing causal
        attribution → counterfactual → optimization pipeline.

        Args:
            result: The latent drift computation result.
            drift_event_id: Optional explicit event ID.

        Returns:
            A :class:`DriftEvent` with ``metric_name="latent_jsd"``.
        """
        return DriftEvent(
            event_id=drift_event_id,
            metric_name="latent_jsd",
            start_timestamp=0.0,
            end_timestamp=0.0,
            magnitude=result.drift_score,
            metadata={
                "latent_drift": True,
                "drift_score": result.drift_score,
                "threshold": result.threshold,
                "n_samples_baseline": result.n_samples_baseline,
                "n_samples_current": result.n_samples_current,
                "pca_components": result.metadata.get("pca_components", 0),
                "engine_config": {
                    "threshold": self.threshold,
                    "pca_components": self.pca_components,
                    "kde_sample_size": self.kde_sample_size,
                },
            },
        )


def compute_latent_drift(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    pca_components: int = 5,
    kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> LatentDriftResult:
    """Convenience function: detect latent drift from two embedding batches.

    Args:
        baseline: Baseline :class:`EmbeddingBatch`.
        current: Current :class:`EmbeddingBatch`.
        threshold: JSD threshold for drift detection.
        pca_components: PCA components to retain.
        kde_sample_size: Grid size for KDE evaluation.

    Returns:
        :class:`LatentDriftResult`
    """
    engine = LatentDriftEngine(
        threshold=threshold,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )
    return engine.fit_compute(baseline.vectors, current.vectors)


def detect_latent_drift_events(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    pca_components: int = 5,
    kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> list[DriftEvent]:
    """Full pipeline: detect latent drift and return DriftEvent(s).

    This integrates the latent drift engine with the existing temporal
    drift detection system.  When drift is detected, a :class:`DriftEvent`
    is produced and can be consumed by the causal attribution pipeline.

    Args:
        baseline: Baseline :class:`EmbeddingBatch`.
        current: Current :class:`EmbeddingBatch`.
        threshold: JSD threshold.
        pca_components: PCA components.
        kde_sample_size: Grid size for KDE evaluation.

    Returns:
        A list containing one :class:`DriftEvent` if drift is detected,
        otherwise an empty list.
    """
    result = compute_latent_drift(
        baseline=baseline,
        current=current,
        threshold=threshold,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )

    if not result.drift_detected:
        return []

    start_ts = (
        current.timestamp.timestamp() if current.timestamp else 0.0
    )

    event = DriftEvent(
        metric_name="latent_jsd",
        start_timestamp=start_ts,
        end_timestamp=start_ts,
        magnitude=result.drift_score,
        metadata={
            "latent_drift": True,
            "drift_score": result.drift_score,
            "threshold": result.threshold,
            "n_samples_baseline": result.n_samples_baseline,
            "n_samples_current": result.n_samples_current,
            "pca_components": result.metadata.get("pca_components", 0),
        },
    )

    return [event]


def _build_shared_grid(
    baseline_proj: np.ndarray,
    current_proj: np.ndarray,
    max_points: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> np.ndarray:
    """Build a shared evaluation grid from actual projected data points.

    Uses the actual projected baseline and current vectors as evaluation
    points, which yields meaningful density comparisons.  When the combined
    set exceeds ``max_points``, a deterministic subsample is taken.

    Args:
        baseline_proj: PCA-projected baseline vectors.
        current_proj: PCA-projected current vectors.
        max_points: Maximum number of evaluation points.

    Returns:
        Grid of shape ``(n_points, n_features)``.
    """
    combined = np.vstack([baseline_proj, current_proj])

    if combined.shape[0] > max_points:
        rng = np.random.RandomState(0)
        indices = rng.choice(combined.shape[0], size=max_points, replace=False)
        combined = combined[indices]

    return combined
