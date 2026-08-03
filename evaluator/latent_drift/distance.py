"""Non-parametric distance metrics for latent drift detection.

Implements Maximum Mean Discrepancy (MMD) with a Gaussian RBF kernel
and Sliced Wasserstein Distance (SWD) — both operate directly on
embedding distributions without requiring grid-based density evaluation.
"""

from __future__ import annotations

import numpy as np


def _rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute the Gaussian RBF kernel matrix between X and Y.

    K(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        X: Array of shape ``(n_x, d)``.
        Y: Array of shape ``(n_y, d)``.
        gamma: Kernel coefficient.

    Returns:
        Kernel matrix of shape ``(n_x, n_y)``.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # (n_x, 1, d) - (1, n_y, d) -> (n_x, n_y, d) -> sum -> (n_x, n_y)
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    cross = 2.0 * X @ Y.T
    dist_sq = XX - cross + YY
    dist_sq = np.maximum(dist_sq, 0.0)  # numerical guard

    return np.exp(-gamma * dist_sq)


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute gamma via the median heuristic.

    gamma = 1 / (2 * median(distance)^2)

    Uses pairwise distances between X and Y samples.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    cross = 2.0 * X @ Y.T
    dist_sq = XX - cross + YY
    dist_sq = np.maximum(dist_sq, 0.0)

    median_dist = np.sqrt(np.median(dist_sq))

    # Guard against zero median distance
    if median_dist < 1e-12:
        return 1.0

    return 1.0 / (2.0 * median_dist ** 2)


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Compute the Maximum Mean Discrepancy with an RBF kernel.

    MMD measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS).  The squared MMD statistic is:

        MMD^2 = mean(K(X, X)) + mean(K(Y, Y)) - 2 * mean(K(X, Y))

    The result is clipped to ``[0, 1]`` for a bounded drift score.

    Args:
        X: Baseline samples of shape ``(n_x, d)``.
        Y: Current samples of shape ``(n_y, d)``.
        gamma: RBF kernel coefficient.  If ``None``, uses the median
            heuristic: ``gamma = 1 / (2 * median_dist^2)``.

    Returns:
        MMD score as a float in ``[0, 1]``.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: X has {X.shape[1]} features, "
            f"Y has {Y.shape[1]} features."
        )

    if gamma is None:
        gamma = _median_heuristic_gamma(X, Y)

    K_xx = _rbf_kernel(X, X, gamma)
    K_yy = _rbf_kernel(Y, Y, gamma)
    K_xy = _rbf_kernel(X, Y, gamma)

    mmd_sq = (
        np.mean(K_xx)
        + np.mean(K_yy)
        - 2.0 * np.mean(K_xy)
    )

    # Square root to get MMD (not squared), then clip
    mmd = float(np.sqrt(max(mmd_sq, 0.0)))
    return float(np.clip(mmd, 0.0, 1.0))


def _random_projections(n_features: int, n_projections: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors for Sliced Wasserstein Distance.

    Samples vectors from a unit sphere in ``n_features`` dimensions.

    Args:
        n_features: Dimensionality of the embedding space.
        n_projections: Number of random projections.
        seed: Random seed for reproducibility.

    Returns:
        Projection matrix of shape ``(n_features, n_projections)``,
        where each column is a unit vector.
    """
    rng = np.random.RandomState(seed)
    projections = rng.normal(0, 1, size=(n_features, n_projections))
    norms = np.linalg.norm(projections, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    return projections / norms


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the 1-D Wasserstein-1 (Earth Mover's) distance.

    Uses the sorted CDF approach: sort both 1-D arrays, compute
    the L1 distance between the empirical CDFs.

    Args:
        a: 1-D array of projected values from distribution X.
        b: 1-D array of projected values from distribution Y.

    Returns:
        Wasserstein-1 distance as a float.
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    n_a = len(a_sorted)
    n_b = len(b_sorted)

    # Merge and sort all values to build a common quantile grid
    all_values = np.concatenate([a_sorted, b_sorted])
    all_values.sort()

    # Compute empirical CDFs at the merged grid points
    cdf_a = np.searchsorted(a_sorted, all_values, side="right") / n_a
    cdf_b = np.searchsorted(b_sorted, all_values, side="right") / n_b

    # Wasserstein-1 = integral of |CDF_a - CDF_b| dx
    # Using the trapezoidal rule on the sorted unique values
    deltas = np.diff(all_values)
    cdf_diff = np.abs(cdf_a[:-1] - cdf_b[:-1])

    return float(np.sum(deltas * cdf_diff))


def compute_swd(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> float:
    """Compute the Sliced Wasserstein Distance between two distributions.

    Projects both distributions onto ``n_projections`` random unit vectors,
    computes the 1-D Wasserstein distance on each projection, and returns
    the mean.  Uses a fixed seed for deterministic results.

    Args:
        X: Baseline samples of shape ``(n_x, d)``.
        Y: Current samples of shape ``(n_y, d)``.
        n_projections: Number of random projections (default 100).
        seed: Random seed for projection generation (default 42).

    Returns:
        SWD score as a float.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: X has {X.shape[1]} features, "
            f"Y has {Y.shape[1]} features."
        )

    n_features = X.shape[1]
    projections = _random_projections(n_features, n_projections, seed)

    distances = []
    for i in range(n_projections):
        proj_dir = projections[:, i]
        X_proj = X @ proj_dir
        Y_proj = Y @ proj_dir
        w_dist = _wasserstein_1d(X_proj, Y_proj)
        distances.append(w_dist)

    return float(np.mean(distances))
