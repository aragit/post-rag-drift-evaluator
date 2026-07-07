import logging
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple

logger = logging.getLogger("DriftMonitor")
logging.basicConfig(level=logging.INFO)

class DriftMonitor:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def _calculate_empirical_distribution(self, embeddings: np.ndarray, bins: int = 20) -> np.ndarray:
        """Projects high-dim embeddings to 1D via PCA to compute an empirical probability distribution."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Reduce dimensionality to find dominant variance axis
        pca = PCA(n_components=1)
        reduced = pca.fit_transform(embeddings).flatten()
        
        # Calculate histogram density
        hist, _ = np.histogram(reduced, bins=bins, density=True)
        # Convert to true probability vector (add epsilon to avoid log(0) issues)
        prob_dist = hist + 1e-12
        return prob_dist / np.sum(prob_dist)

    def compute_jensen_shannon_drift(self, baseline_df: pl.DataFrame, current_df: pl.DataFrame, embedding_col: str = "embedding") -> Tuple[float, bool]:
        """
        Calculates JS Divergence between baseline traffic and current traffic embeddings.
        Returns:
            Tuple[float, bool]: The divergence score and a boolean flag indicating if drift exceeds threshold.
        """
        # Extract embedding matrices out of Polars series
        baseline_matrix = np.array(baseline_df[embedding_col].to_list())
        current_matrix = np.array(current_df[embedding_col].to_list())
        
        # Compute empirical 1D distributions
        p = self._calculate_empirical_distribution(baseline_matrix)
        q = self._calculate_empirical_distribution(current_matrix)
        
        # Compute JS Divergence (ranges from 0.0 to 1.0)
        js_divergence = float(jensenshannon(p, q))
        
        is_drifted = js_divergence > self.threshold
        if is_drifted:
            logger.critical(
                f"🚨 SYSTEM ALERT: Latent Space Model Drift Detected! "
                f"Score: {js_divergence:.4f} exceeds threshold: {self.threshold}"
            )
        else:
            logger.info(f"System stable. Current Latent Space Drift Score: {js_divergence:.4f}")
            
        return js_divergence, is_drifted
