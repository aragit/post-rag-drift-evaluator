"""PCA projection utilities for latent drift detection."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(
    baseline_vectors: np.ndarray,
    n_components: int = 10,
) -> PCA:
    """Fit a PCA model on baseline embedding vectors.

    The PCA is fit **only** on the baseline data.  It can then be used
    to project both baseline and current vectors into the same latent
    space.

    Args:
        baseline_vectors: 2-D array of shape ``(n_samples, dim)``.
        n_components: Maximum number of principal components.  Defaults
            to 10, but is automatically capped at the available rank
            of the input.

    Returns:
        A fitted :class:`sklearn.decomposition.PCA` instance.
    """
    baseline_vectors = np.atleast_2d(baseline_vectors)
    max_components = min(baseline_vectors.shape)
    actual_components = min(n_components, max_components)

    pca = PCA(n_components=actual_components)
    pca.fit(baseline_vectors)
    return pca


def project_vectors(
    pca: PCA,
    vectors: np.ndarray,
) -> np.ndarray:
    """Project vectors into the PCA-fitted latent space.

    Args:
        pca: A fitted :class:`~sklearn.decomposition.PCA` instance.
        vectors: 2-D array of shape ``(n_samples, dim)``.

    Returns:
        Projected vectors of shape ``(n_samples, n_components)``.
    """
    vectors = np.atleast_2d(vectors)
    return pca.transform(vectors)
