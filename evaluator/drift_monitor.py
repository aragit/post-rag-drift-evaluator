import numpy as np
import polars as pl
from scipy.spatial.distance import cdist, jensenshannon, pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from typing import Any, Dict, List, Optional, Tuple

from alerting.notifier import DriftAlertNotifier
from evaluator.config import config
from evaluator.drift.graph_drift import GraphDriftCalculator
from evaluator.drift.swarm_drift import SwarmDriftCalculator
from evaluator.drift_store import DriftStore
from evaluator.logging_config import get_logger
from evaluator.schemas.telemetry import RAGEvaluationFrame

logger = get_logger("DriftMonitor")


class DriftMonitor:
    def __init__(
        self,
        threshold: float = config.DRIFT_THRESHOLD,
        store: DriftStore | None = None,
        notifier: Optional[DriftAlertNotifier] = None,
        baseline_service: Optional[object] = None,
    ):
        self.threshold = threshold
        self.mmd_threshold = config.MMD_THRESHOLD
        self.per_component_kl_threshold = config.PER_COMPONENT_KL_THRESHOLD
        self.calibrated_thresholds: Dict[str, float] = {}
        self._calibrated = False
        self._store = store or DriftStore()
        self.notifier = notifier
        self._graph_calculator = GraphDriftCalculator()
        self._swarm_calculator = SwarmDriftCalculator()
        self._baseline_service = baseline_service
        self._saved_thresholds: Optional[Dict[str, Any]] = None

    def calibrate_thresholds(
        self, baseline_embeddings: np.ndarray, n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        baseline_embeddings = np.atleast_2d(baseline_embeddings)
        n_samples = baseline_embeddings.shape[0]

        js_scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = baseline_embeddings[idx]

            p = self._calculate_empirical_distribution(baseline_embeddings)
            q = self._calculate_empirical_distribution(bootstrap_sample)
            js_score = float(jensenshannon(p, q))
            js_scores.append(js_score)

        js_scores = np.array(js_scores)
        js_mean = float(np.mean(js_scores))
        js_std = float(np.std(js_scores))
        calibrated_js_threshold = js_mean + 3 * js_std

        mmd_scores = []
        for _ in range(min(n_bootstrap, 100)):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = baseline_embeddings[idx]

            pairwise_bb = squareform(pdist(baseline_embeddings, metric="sqeuclidean"))
            pairwise_bs = squareform(pdist(bootstrap_sample, metric="sqeuclidean"))
            cross_b = cdist(baseline_embeddings, bootstrap_sample, metric="sqeuclidean")

            median_sq_dist = np.median(
                np.concatenate(
                    [
                        pairwise_bb[pairwise_bb > 0],
                        pairwise_bs[pairwise_bs > 0],
                        cross_b.ravel(),
                    ]
                )
            )
            bandwidth = median_sq_dist / (2 * np.log(max(n_samples, n_samples) + 1))
            if bandwidth <= 0:
                bandwidth = 1.0

            gamma = 1.0 / (2 * bandwidth)

            def rbf_kernel(X, Y, g):
                return np.exp(-g * cdist(X, Y, metric="sqeuclidean"))

            k_bb = rbf_kernel(baseline_embeddings, baseline_embeddings, gamma)
            k_bs = rbf_kernel(bootstrap_sample, bootstrap_sample, gamma)
            k_bb_cross = rbf_kernel(baseline_embeddings, bootstrap_sample, gamma)

            mmd_sq = max(0.0, np.mean(k_bb) + np.mean(k_bs) - 2 * np.mean(k_bb_cross))
            mmd_scores.append(np.sqrt(mmd_sq))

        mmd_scores = np.array(mmd_scores)
        mmd_mean = float(np.mean(mmd_scores))
        mmd_std = float(np.std(mmd_scores))
        calibrated_mmd_threshold = mmd_mean + 3 * mmd_std

        kl_scores = []
        n_components = min(10, baseline_embeddings.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(baseline_embeddings)
        baseline_proj = pca.transform(baseline_embeddings)

        for i in range(n_components):
            col = baseline_proj[:, i].reshape(-1, 1)
            try:
                kde = gaussian_kde(col.ravel())
                eval_points = np.linspace(col.min(), col.max(), 100).reshape(-1, 1)
                densities = kde(eval_points.T)
                densities = np.clip(densities, 1e-12, None)
                entropy = -float(np.sum(densities * np.log(densities)))
                kl_scores.append(entropy)
            except Exception:
                kl_scores.append(0.0)

        kl_scores = np.array(kl_scores)
        kl_mean = float(np.mean(kl_scores))
        kl_std = float(np.std(kl_scores))
        calibrated_kl_threshold = kl_mean + 3 * kl_std

        self.calibrated_thresholds = {
            "js_divergence": calibrated_js_threshold,
            "mmd_score": calibrated_mmd_threshold,
            "per_component_kl": calibrated_kl_threshold,
        }
        self._calibrated = True

        logger.info(
            f"Thresholds calibrated: JSD={calibrated_js_threshold:.4f}, "
            f"MMD={calibrated_mmd_threshold:.4f}, Per-component KL={calibrated_kl_threshold:.4f}"
        )

        return self.calibrated_thresholds

    def _get_threshold(self, metric: str) -> float:
        if self._calibrated and metric in self.calibrated_thresholds:
            return self.calibrated_thresholds[metric]
        if metric == "js_divergence":
            return self.threshold
        if metric == "mmd_score":
            return self.mmd_threshold
        if metric == "per_component_kl":
            return self.per_component_kl_threshold
        return 0.0

    def is_drifted(self, js_score: float, mmd_score: float, max_kl: float) -> bool:
        js_threshold = self._get_threshold("js_divergence")
        mmd_threshold = self._get_threshold("mmd_score")
        kl_threshold = self._get_threshold("per_component_kl")

        if not self._calibrated:
            logger.warning(
                "Thresholds not calibrated. Using defaults. "
                "Call calibrate_thresholds() for data-driven thresholds."
            )

        return (
            js_score > js_threshold
            or mmd_score > mmd_threshold
            or max_kl > kl_threshold
        )

    def _calculate_empirical_distribution(
        self, embeddings: np.ndarray, bins: int = 20
    ) -> np.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        pca = PCA(n_components=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            reduced = pca.fit_transform(embeddings).flatten()
        hist, _ = np.histogram(reduced, bins=bins, density=True)
        prob_dist = hist + 1e-12
        return prob_dist / np.sum(prob_dist)

    def _extract_embeddings(self, df: pl.DataFrame, embedding_col: str) -> np.ndarray:
        return np.array(df[embedding_col].to_list())

    def compute_jensen_shannon_drift(
        self,
        baseline_df: pl.DataFrame,
        current_df: pl.DataFrame,
        embedding_col: str = "embedding",
    ) -> Tuple[float, bool]:
        baseline_matrix = self._extract_embeddings(baseline_df, embedding_col)
        current_matrix = self._extract_embeddings(current_df, embedding_col)

        p = self._calculate_empirical_distribution(baseline_matrix)
        q = self._calculate_empirical_distribution(current_matrix)

        js_divergence = float(jensenshannon(p, q))

        is_drifted = self.is_drifted(js_divergence, 0.0, 0.0)
        if is_drifted:
            logger.critical(
                f"Drift detected: JSD={js_divergence:.4f} exceeds threshold={self._get_threshold('js_divergence'):.4f}"
            )
        else:
            logger.info(f"System stable. JSD={js_divergence:.4f}")

        return js_divergence, is_drifted

    def compute_mmd_drift(
        self,
        baseline_df: pl.DataFrame,
        current_df: pl.DataFrame,
        embedding_col: str = "embedding",
    ) -> Tuple[float, float, bool]:
        baseline_matrix = self._extract_embeddings(baseline_df, embedding_col)
        current_matrix = self._extract_embeddings(current_df, embedding_col)

        baseline_matrix = np.atleast_2d(baseline_matrix)
        current_matrix = np.atleast_2d(current_matrix)

        n_baseline = baseline_matrix.shape[0]
        n_current = current_matrix.shape[0]

        pairwise_baseline = squareform(pdist(baseline_matrix, metric="sqeuclidean"))
        pairwise_current = squareform(pdist(current_matrix, metric="sqeuclidean"))
        cross_distances = cdist(baseline_matrix, current_matrix, metric="sqeuclidean")

        median_sq_dist = np.median(
            np.concatenate(
                [
                    pairwise_baseline[pairwise_baseline > 0],
                    pairwise_current[pairwise_current > 0],
                    cross_distances.ravel(),
                ]
            )
        )
        bandwidth = median_sq_dist / (2 * np.log(max(n_baseline, n_current) + 1))

        if bandwidth <= 0:
            bandwidth = 1.0

        def rbf_kernel(X, Y, gamma):
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            return np.exp(-gamma * sq_dists)

        gamma = 1.0 / (2 * bandwidth)

        k_bb = rbf_kernel(baseline_matrix, baseline_matrix, gamma)
        k_cc = rbf_kernel(current_matrix, current_matrix, gamma)
        k_bc = rbf_kernel(baseline_matrix, current_matrix, gamma)

        mmd_sq = np.mean(k_bb) + np.mean(k_cc) - 2 * np.mean(k_bc)
        mmd_sq = max(0.0, mmd_sq)
        mmd_score = float(np.sqrt(mmd_sq))

        n_permutations = 100
        perm_scores = []
        combined = np.vstack([baseline_matrix, current_matrix])
        n_total = combined.shape[0]
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(n_total)
            perm_b = combined[perm_idx[:n_baseline]]
            perm_c = combined[perm_idx[n_baseline:]]

            perm_bb = rbf_kernel(perm_b, perm_b, gamma)
            perm_cc = rbf_kernel(perm_c, perm_c, gamma)
            perm_bc = rbf_kernel(perm_b, perm_c, gamma)

            perm_mmd = np.mean(perm_bb) + np.mean(perm_cc) - 2 * np.mean(perm_bc)
            perm_scores.append(max(0.0, perm_mmd))

        perm_scores = np.array(perm_scores)
        p_value = float(np.mean(perm_scores >= mmd_sq))

        mmd_threshold = self._get_threshold("mmd_score")
        is_drifted = p_value < 0.05 or mmd_score > mmd_threshold
        if is_drifted:
            logger.critical(
                f"MMD drift detected: score={mmd_score:.4f}, p_value={p_value:.4f}, "
                f"threshold={mmd_threshold:.4f}"
            )
        else:
            logger.info(f"MMD stable: score={mmd_score:.4f}, p_value={p_value:.4f}")

        return mmd_score, p_value, is_drifted

    def compute_per_component_drift(
        self,
        baseline_df: pl.DataFrame,
        current_df: pl.DataFrame,
        embedding_col: str = "embedding",
    ) -> Tuple[float, float, bool]:
        baseline_matrix = self._extract_embeddings(baseline_df, embedding_col)
        current_matrix = self._extract_embeddings(current_df, embedding_col)

        baseline_matrix = np.atleast_2d(baseline_matrix)
        current_matrix = np.atleast_2d(current_matrix)

        n_components = min(10, baseline_matrix.shape[1], current_matrix.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(baseline_matrix)

        baseline_proj = pca.transform(baseline_matrix)
        current_proj = pca.transform(current_matrix)

        kl_divergences = []
        for i in range(n_components):
            baseline_col = baseline_proj[:, i]
            current_col = current_proj[:, i]

            baseline_col = baseline_col.reshape(-1, 1)
            current_col = current_col.reshape(-1, 1)

            try:
                kde_b = gaussian_kde(baseline_col.ravel())
                kde_c = gaussian_kde(current_col.ravel())

                eval_points = np.linspace(
                    min(baseline_col.min(), current_col.min()),
                    max(baseline_col.max(), current_col.max()),
                    100,
                ).reshape(-1, 1)

                p_b = kde_b(eval_points.T)
                p_c = kde_c(eval_points.T)

                p_b = np.clip(p_b, 1e-12, None)
                p_c = np.clip(p_c, 1e-12, None)

                kl = float(np.sum(p_b * np.log(p_b / p_c)))
                kl_divergences.append(kl)
            except Exception:
                kl_divergences.append(0.0)

        kl_array = np.array(kl_divergences)
        max_kl = float(np.max(kl_array)) if len(kl_array) > 0 else 0.0
        mean_kl = float(np.mean(kl_array)) if len(kl_array) > 0 else 0.0

        kl_threshold = self._get_threshold("per_component_kl")
        is_drifted = max_kl > kl_threshold
        if is_drifted:
            logger.critical(
                f"Per-component drift detected: max_kl={max_kl:.4f}, "
                f"threshold={kl_threshold:.4f}"
            )
        else:
            logger.info(f"Per-component stable: max_kl={max_kl:.4f}")

        return max_kl, mean_kl, is_drifted

    async def trend_analysis(self, hours: int = 24, window: int = 30) -> Dict[str, Any]:
        history = await self._store.get_recent_history(hours=hours)
        trend = await self._store.get_trend(window=window)
        anomaly = await self._store.detect_anomaly(window=window)
        return {
            "history": history,
            "trend": trend,
            "anomaly_detected": anomaly,
        }

    async def compute_comprehensive_drift(
        self,
        baseline_df: pl.DataFrame,
        current_df: pl.DataFrame,
        embedding_col: str = "embedding",
    ) -> Dict[str, Any]:
        js_score, js_drifted = self.compute_jensen_shannon_drift(
            baseline_df, current_df, embedding_col
        )
        mmd_score, mmd_p_value, mmd_drifted = self.compute_mmd_drift(
            baseline_df, current_df, embedding_col
        )
        max_kl, mean_kl, pc_drifted = self.compute_per_component_drift(
            baseline_df, current_df, embedding_col
        )

        is_drifted = js_drifted or mmd_drifted or pc_drifted

        result = {
            "js_divergence": js_score,
            "js_drifted": js_drifted,
            "mmd_score": mmd_score,
            "mmd_p_value": mmd_p_value,
            "mmd_drifted": mmd_drifted,
            "max_component_kl": max_kl,
            "mean_component_kl": mean_kl,
            "per_component_drifted": pc_drifted,
            "is_drifted": is_drifted,
        }

        await self._store.record_drift(result)

        return result

    @staticmethod
    def _collect_vector_embeddings(
        frames: List[RAGEvaluationFrame],
    ) -> Optional[np.ndarray]:
        vectors = []
        for frame in frames:
            embedding = frame.query.embedding
            if embedding is None:
                embedding = frame.output.response_embedding
            if embedding is None and frame.context.dense_embeddings:
                embedding = frame.context.dense_embeddings[0]
            if embedding:
                vectors.append(embedding)
        if not vectors:
            return None
        return np.array(vectors, dtype=float)

    def _evaluate_vector_drift(
        self,
        baseline_frames: List[RAGEvaluationFrame],
        current_frames: List[RAGEvaluationFrame],
    ) -> Dict[str, Any]:
        result = {
            "js_divergence": 0.0,
            "mmd_score": 0.0,
            "is_drifted": False,
        }
        baseline_vectors = self._collect_vector_embeddings(baseline_frames)
        current_vectors = self._collect_vector_embeddings(current_frames)
        if (
            baseline_vectors is None
            or current_vectors is None
            or baseline_vectors.shape[1] != current_vectors.shape[1]
        ):
            return result

        baseline_df = pl.DataFrame({"embedding": baseline_vectors.tolist()})
        current_df = pl.DataFrame({"embedding": current_vectors.tolist()})

        js_score, js_drifted = self.compute_jensen_shannon_drift(
            baseline_df, current_df, "embedding"
        )
        if len(baseline_vectors) < 2 or len(current_vectors) < 2:
            mmd_score, mmd_drifted = 0.0, False
        else:
            mmd_score, _, mmd_drifted = self.compute_mmd_drift(
                baseline_df, current_df, "embedding"
            )

        return {
            "js_divergence": js_score,
            "mmd_score": mmd_score,
            "is_drifted": js_drifted or mmd_drifted,
        }

    def _evaluate_graph_drift(
        self,
        baseline_frames: List[RAGEvaluationFrame],
        current_frames: List[RAGEvaluationFrame],
    ) -> Dict[str, Any]:
        baseline_graphs = [
            frame.context.graph_topology
            for frame in baseline_frames
            if frame.context.graph_topology is not None
        ]
        current_graphs = [
            frame.context.graph_topology
            for frame in current_frames
            if frame.context.graph_topology is not None
        ]
        if not baseline_graphs and not current_graphs:
            return {
                "spectral_distance": 0.0,
                "density_delta": 0.0,
                "node_count_delta": 0,
                "is_graph_drifted": False,
            }
        return self._graph_calculator.compute_graph_drift(
            baseline_graphs, current_graphs
        )

    def _evaluate_swarm_drift(
        self,
        baseline_frames: List[RAGEvaluationFrame],
        current_frames: List[RAGEvaluationFrame],
    ) -> Dict[str, Any]:
        baseline_metadata = [
            frame.metadata
            for frame in baseline_frames
            if frame.metadata.agent_hops or frame.metadata.reflection_iterations
        ]
        current_metadata = [
            frame.metadata
            for frame in current_frames
            if frame.metadata.agent_hops or frame.metadata.reflection_iterations
        ]
        if not baseline_metadata and not current_metadata:
            return {
                "transition_entropy_delta": 0.0,
                "avg_reflection_iterations_delta": 0.0,
                "is_swarm_drifted": False,
            }
        return self._swarm_calculator.compute_swarm_drift(
            baseline_metadata, current_metadata
        )

    def _apply_calibrated_thresholds(self, thresholds: Dict[str, float]) -> None:
        if not thresholds:
            return
        self._saved_thresholds = {
            "jsd": self.threshold,
            "mmd": self.mmd_threshold,
            "kl": self.per_component_kl_threshold,
            "spectral": self._graph_calculator.spectral_threshold,
            "graph_density": self._graph_calculator.density_threshold,
            "entropy": self._swarm_calculator.entropy_threshold,
            "reflection": self._swarm_calculator.reflection_threshold,
            "calibrated": self._calibrated,
            "calibrated_thresholds": dict(self.calibrated_thresholds),
        }
        self.threshold = thresholds.get(
            "vector_jsd_threshold", self._saved_thresholds["jsd"]
        )
        self.mmd_threshold = thresholds.get(
            "vector_mmd_threshold", self._saved_thresholds["mmd"]
        )
        self._graph_calculator.spectral_threshold = thresholds.get(
            "graph_spectral_threshold", self._saved_thresholds["spectral"]
        )
        self._swarm_calculator.entropy_threshold = thresholds.get(
            "swarm_entropy_threshold", self._saved_thresholds["entropy"]
        )
        self._calibrated = True
        self.calibrated_thresholds = dict(thresholds)

    def _restore_thresholds(self) -> None:
        if self._saved_thresholds is None:
            return
        self.threshold = self._saved_thresholds["jsd"]
        self.mmd_threshold = self._saved_thresholds["mmd"]
        self.per_component_kl_threshold = self._saved_thresholds["kl"]
        self._graph_calculator.spectral_threshold = self._saved_thresholds["spectral"]
        self._graph_calculator.density_threshold = self._saved_thresholds["graph_density"]
        self._swarm_calculator.entropy_threshold = self._saved_thresholds["entropy"]
        self._swarm_calculator.reflection_threshold = self._saved_thresholds["reflection"]
        self._calibrated = self._saved_thresholds["calibrated"]
        self.calibrated_thresholds = self._saved_thresholds["calibrated_thresholds"]
        self._saved_thresholds = None

    async def evaluate_frames(
        self,
        baseline_frames: Optional[List[RAGEvaluationFrame]] = None,
        current_frames: Optional[List[RAGEvaluationFrame]] = None,
    ) -> Dict[str, Any]:
        """Evaluate hybrid vector/graph/swarm drift between two windows.

        Results are keyed as ``vector_drift``, ``graph_drift``,
        ``swarm_drift``, and ``is_drifted``. When ``current_frames`` is
        non-empty, each current frame is persisted via ``record_evaluation``.

        If *baseline_frames* is ``None`` or empty and a ``baseline_service``
        is attached, a sliding-window baseline is fetched and dynamically
        calibrated thresholds (``mu + k*sigma``) are applied.  When fewer
        than ``MIN_BASELINE_FRAMES`` are available, static safety thresholds
        are used as a graceful fallback.
        """
        if current_frames is None:
            current_frames = []

        baseline_source: str = "explicit"

        if not baseline_frames and self._baseline_service is not None:
            baseline_frames = await self._baseline_service.fetch_sliding_baseline_frames()
            baseline_source = "dynamic"
            logger.info(
                "Fetched %d dynamic baseline frames for evaluation.",
                len(baseline_frames) if baseline_frames else 0,
            )

            thresholds = self._baseline_service.compute_calibrated_thresholds(
                baseline_frames or []
            )
            if thresholds:
                self._apply_calibrated_thresholds(thresholds)
                logger.info(
                    "Applied dynamically calibrated thresholds: %s", thresholds
                )
            else:
                logger.info(
                    "Falling back to static thresholds (insufficient baseline frames)."
                )

        if not baseline_frames:
            baseline_frames = []

        try:
            vector_drift = self._evaluate_vector_drift(baseline_frames, current_frames)
            graph_drift = self._evaluate_graph_drift(baseline_frames, current_frames)
            swarm_drift = self._evaluate_swarm_drift(baseline_frames, current_frames)
        finally:
            if baseline_source == "dynamic":
                self._restore_thresholds()

        is_drifted = (
            bool(vector_drift["is_drifted"])
            or bool(graph_drift["is_graph_drifted"])
            or bool(swarm_drift["is_swarm_drifted"])
        )

        result = {
            "vector_drift": vector_drift,
            "graph_drift": graph_drift,
            "swarm_drift": swarm_drift,
            "is_drifted": is_drifted,
        }

        if self.notifier is not None:
            await self.notifier.notify_if_drifted(result)

        if current_frames:
            metrics = {
                "js_divergence": vector_drift["js_divergence"],
                "mmd_score": vector_drift["mmd_score"],
                "wasserstein_distance": None,
                "is_drifted": is_drifted,
                "spectral_distance": graph_drift["spectral_distance"],
                "density_delta": graph_drift["density_delta"],
                "node_count_delta": graph_drift["node_count_delta"],
                "transition_entropy_delta": swarm_drift["transition_entropy_delta"],
                "avg_reflection_iterations_delta": swarm_drift[
                    "avg_reflection_iterations_delta"
                ],
            }
            for frame in current_frames:
                await self._store.record_evaluation(frame, metrics)

        return result
