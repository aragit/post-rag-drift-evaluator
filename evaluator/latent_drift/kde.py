"""KDE density estimation utilities for latent drift detection."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def fit_kde(vectors: np.ndarray) -> gaussian_kde:
    """Fit a Gaussian KDE on (PCA-projected) vectors.

    Uses Scott's rule for bandwidth selection (the default in SciPy).

    Args:
        vectors: 2-D array of shape ``(n_samples, n_features)``.

    Returns:
        A fitted :class:`scipy.stats.gaussian_kde` instance.

    Raises:
        ValueError: If fewer than 2 samples are provided (KDE requires
            at least as many samples as dimensions for a valid covariance
            estimate).
    """
    vectors = np.atleast_2d(vectors)
    n_samples, n_features = vectors.shape

    if n_samples < 2:
        raise ValueError(
            f"KDE requires at least 2 samples, got {n_samples}. "
            "Consider collecting more baseline data."
        )

    # gaussian_kde expects shape (n_features, n_samples)
    return gaussian_kde(vectors.T)


def evaluate_density(
    kde: gaussian_kde,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate a KDE on a set of grid points.

    Args:
        kde: A fitted :class:`~scipy.stats.gaussian_kde` instance.
        points: 2-D array of shape ``(n_points, n_features)``.

    Returns:
        1-D array of density values of length ``n_points``.
    """
    points = np.atleast_2d(points)
    if points.shape[0] == 1:
        points = points.T
    return kde(points.T)
