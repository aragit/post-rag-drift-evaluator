"""Latent drift detection engine using PCA + configurable metrics.

This module implements a statistical drift detector that operates directly
on embedding vectors.  It detects distribution-level semantic drift by:

1. Projecting embeddings into a stable latent manifold (PCA).
2. Computing a divergence score using one of several metrics:
   - MMD (Maximum Mean Discrepancy) with RBF kernel — default
   - SWD (Sliced Wasserstein Distance)
   - JSD (Jensen-Shannon Divergence via KDE)
3. Comparing the score against a threshold.

Results are converted to :class:`~evaluator.temporal.models.DriftEvent`
objects so they integrate seamlessly with the existing causal attribution,
counterfactual simulation, and optimization pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from evaluator.latent_drift.distance import compute_mmd, compute_swd
from evaluator.latent_drift.jsd import compute_jsd
from evaluator.latent_drift.kde import evaluate_density, fit_kde
from evaluator.latent_drift.pca import fit_pca, project_vectors
from evaluator.latent_drift.schemas import EmbeddingBatch, LatentDriftResult
from evaluator.temporal.models import DriftEvent

if TYPE_CHECKING:
    from evaluator.latent_drift.adaptive import AdaptiveThresholdManager

_EPSILON = 1e-12
_DEFAULT_KDE_SAMPLE_SIZE = 1000
_VALID_METRICS = {"mmd", "swd", "jsd"}


class LatentDriftEngine:
    """Detect distribution-level drift on embedding vectors.

    The engine fits a PCA model + KDE on baseline embeddings during
    :meth:`fit`, then compares incoming embeddings during
    :meth:`compute_drift`.

    Attributes:
        threshold: Drift threshold for the score.
        metric: Distance metric to use (``"mmd"``, ``"swd"``, or ``"jsd""``).
        pca_components: Number of PCA components to retain.
        kde_sample_size: Maximum grid points for KDE evaluation (JSD only).
        threshold_manager: Optional :class:`AdaptiveThresholdManager` for
            dynamic threshold computation.  When set, the adaptive
            threshold overrides the static ``threshold`` during
            :meth:`compute_drift`.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        metric: str = "mmd",
        pca_components: int = 5,
        kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
        threshold_manager: AdaptiveThresholdManager | None = None,
    ):
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {_VALID_METRICS}"
            )
        self.threshold = threshold
        self.metric = metric
        self.pca_components = pca_components
        self.kde_sample_size = kde_sample_size
        self.threshold_manager = threshold_manager
        self.pca: Any = None
        self.kde_baseline: Any = None
        self._baseline_proj: np.ndarray | None = None

    def fit(self, baseline_vectors: np.ndarray) -> None:
        """Fit PCA (and KDE for JSD mode) on baseline embedding vectors.

        Args:
            baseline_vectors: 2-D array of shape ``(n_samples, dim)``.
        """
        baseline_vectors = np.atleast_2d(baseline_vectors)

        self.pca = fit_pca(baseline_vectors, n_components=self.pca_components)
        self._baseline_proj = project_vectors(self.pca, baseline_vectors)

        if self.metric == "jsd":
            self.kde_baseline = fit_kde(self._baseline_proj)

    def compute_drift(
        self,
        current_vectors: np.ndarray,
    ) -> LatentDriftResult:
        """Compute latent drift between baseline and current embeddings.

        Dispatches to the configured metric (``mmd``, ``swd``, or ``jsd``)
        on PCA-projected vectors.

        Args:
            current_vectors: 2-D array of shape ``(n_samples, dim)``.
                Must have the same dimensionality as the baseline.

        Returns:
            A :class:`LatentDriftResult` with the drift score and metadata.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.pca is None:
            raise RuntimeError(
                "LatentDriftEngine must be fitted first. Call fit() "
                "before compute_drift()."
            )

        if self.metric == "jsd" and self.kde_baseline is None:
            raise RuntimeError(
                "LatentDriftEngine must be fitted first. Call fit() "
                "before compute_drift()."
            )

        current_vectors = np.atleast_2d(current_vectors)
        baseline_proj = (
            self._baseline_proj if self._baseline_proj is not None else np.empty((0, 0))
        )

        current_proj = project_vectors(self.pca, current_vectors)

        drift_score = self._compute_metric(baseline_proj, current_proj)

        # Use adaptive threshold if manager is configured
        if self.threshold_manager is not None:
            self.threshold_manager.update(drift_score)
            effective_threshold = self.threshold_manager.get_threshold()
        else:
            effective_threshold = self.threshold

        metadata = {
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "metric": self.metric,
            "kde_sample_size": self.kde_sample_size,
            "adaptive_threshold": self.threshold_manager is not None,
        }

        return LatentDriftResult(
            drift_score=drift_score,
            drift_detected=drift_score > effective_threshold,
            threshold=effective_threshold,
            n_samples_baseline=baseline_proj.shape[0],
            n_samples_current=current_proj.shape[0],
            metric_used=self.metric,
            metadata=metadata,
        )

    def _compute_metric(
        self,
        baseline_proj: np.ndarray,
        current_proj: np.ndarray,
    ) -> float:
        """Dispatch to the configured divergence metric."""
        if self.metric == "mmd":
            return compute_mmd(baseline_proj, current_proj)
        elif self.metric == "swd":
            return compute_swd(baseline_proj, current_proj)
        elif self.metric == "jsd":
            grid = _build_shared_grid(baseline_proj, current_proj)
            p_density = evaluate_density(self.kde_baseline, grid)
            try:
                kde_current = fit_kde(current_proj)
                q_density = evaluate_density(kde_current, grid)
            except (ValueError, np.linalg.LinAlgError):
                q_density = np.full_like(p_density, _EPSILON)
            return compute_jsd(p_density, q_density)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

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

    def compute_dual_track_drift(
        self,
        baseline_retrieval: np.ndarray,
        current_retrieval: np.ndarray,
        baseline_generation: np.ndarray,
        current_generation: np.ndarray,
    ) -> dict[str, LatentDriftResult]:
        """Compute drift separately for retrieval and generation tracks.

        Fits PCA on the combined baseline (retrieval + generation), then
        computes drift for each track independently.  Also computes a
        unified score on the combined embeddings.

        Args:
            baseline_retrieval: Baseline retrieval embeddings.
            current_retrieval: Current retrieval embeddings.
            baseline_generation: Baseline generation embeddings.
            current_generation: Current generation embeddings.

        Returns:
            A dict with keys ``"retrieval"``, ``"generation"``, and
            ``"unified"``, each mapping to a :class:`LatentDriftResult`.
        """
        baseline_vectors = np.vstack(
            [
                np.atleast_2d(baseline_retrieval),
                np.atleast_2d(baseline_generation),
            ]
        )

        self.fit(baseline_vectors)

        base_proj = np.atleast_2d(
            self._baseline_proj if self._baseline_proj is not None else np.empty((0, 0))
        )
        n_base_retrieval = np.atleast_2d(baseline_retrieval).shape[0]
        n_base_generation = np.atleast_2d(baseline_generation).shape[0]
        base_ret_proj = base_proj[:n_base_retrieval]
        base_gen_proj = base_proj[
            n_base_retrieval : n_base_retrieval + n_base_generation
        ]

        cur_ret_proj = project_vectors(self.pca, np.atleast_2d(current_retrieval))
        cur_gen_proj = project_vectors(self.pca, np.atleast_2d(current_generation))

        metric_breakdown: dict[str, float] = {}

        ret_score = self._compute_metric(base_ret_proj, cur_ret_proj)
        metric_breakdown["retrieval"] = ret_score

        gen_score = self._compute_metric(base_gen_proj, cur_gen_proj)
        metric_breakdown["generation"] = gen_score

        combined_base = base_proj
        combined_curr = np.vstack([cur_ret_proj, cur_gen_proj])
        unified_score = self._compute_metric(combined_base, combined_curr)
        metric_breakdown["unified"] = unified_score

        results: dict[str, LatentDriftResult] = {
            "retrieval": LatentDriftResult(
                drift_score=ret_score,
                drift_detected=ret_score > self.threshold,
                threshold=self.threshold,
                n_samples_baseline=base_ret_proj.shape[0],
                n_samples_current=cur_ret_proj.shape[0],
                metric_used=self.metric,
                track="retrieval",
                metric_breakdown=metric_breakdown.copy(),
                metadata={"pca_components": self.pca.n_components_},
            ),
            "generation": LatentDriftResult(
                drift_score=gen_score,
                drift_detected=gen_score > self.threshold,
                threshold=self.threshold,
                n_samples_baseline=base_gen_proj.shape[0],
                n_samples_current=cur_gen_proj.shape[0],
                metric_used=self.metric,
                track="generation",
                metric_breakdown=metric_breakdown.copy(),
                metadata={"pca_components": self.pca.n_components_},
            ),
            "unified": LatentDriftResult(
                drift_score=unified_score,
                drift_detected=unified_score > self.threshold,
                threshold=self.threshold,
                n_samples_baseline=combined_base.shape[0],
                n_samples_current=combined_curr.shape[0],
                metric_used=self.metric,
                track="unified",
                metric_breakdown=metric_breakdown.copy(),
                metadata={"pca_components": self.pca.n_components_},
            ),
        }

        return results

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
            A :class:`DriftEvent` with ``metric_name="latent_jsd"``
            (or ``"latent_mmd"`` / ``"latent_swd"`` depending on metric).
        """
        metric_name = f"latent_{result.metric_used}"

        return DriftEvent(
            event_id=drift_event_id,
            metric_name=metric_name,
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
                "metric_used": result.metric_used,
                "track": result.track,
                "metric_breakdown": result.metric_breakdown,
                "engine_config": {
                    "threshold": self.threshold,
                    "metric": self.metric,
                    "pca_components": self.pca_components,
                    "kde_sample_size": self.kde_sample_size,
                },
            },
        )


def compute_latent_drift(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    metric: str = "mmd",
    pca_components: int = 5,
    kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> LatentDriftResult:
    """Convenience function: detect latent drift from two embedding batches.

    Args:
        baseline: Baseline :class:`EmbeddingBatch`.
        current: Current :class:`EmbeddingBatch`.
        threshold: Score threshold for drift detection.
        metric: Distance metric (``"mmd"``, ``"swd"``, or ``"jsd"``).
        pca_components: PCA components to retain.
        kde_sample_size: Grid size for KDE evaluation (JSD only).

    Returns:
        :class:`LatentDriftResult`
    """
    engine = LatentDriftEngine(
        threshold=threshold,
        metric=metric,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )
    result = engine.fit_compute(baseline.vectors, current.vectors)
    result.track = baseline.track
    return result


def detect_latent_drift_events(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    metric: str = "mmd",
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
        threshold: Score threshold.
        metric: Distance metric (``"mmd"``, ``"swd"``, or ``"jsd"``).
        pca_components: PCA components.
        kde_sample_size: Grid size for KDE evaluation (JSD only).

    Returns:
        A list containing one :class:`DriftEvent` if drift is detected,
        otherwise an empty list.
    """
    result = compute_latent_drift(
        baseline=baseline,
        current=current,
        threshold=threshold,
        metric=metric,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )

    if not result.drift_detected:
        return []

    start_ts = current.timestamp.timestamp() if current.timestamp else 0.0

    event = DriftEvent(
        metric_name=f"latent_{metric}",
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
            "metric_used": result.metric_used,
            "track": result.track,
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
