"""Jensen-Shannon Divergence computation for latent drift detection."""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def compute_jsd(
    p_density: np.ndarray,
    q_density: np.ndarray,
) -> float:
    """Compute the Jensen-Shannon Divergence between two density arrays.

    Steps:
    1. Normalize densities so they sum to 1.
    2. Compute the midpoint distribution ``M = 0.5 * (P + Q)``.
    3. Compute ``JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)``.

    The result is bounded ``[0, 1]`` (base-2 logarithm).

    Args:
        p_density: 1-D array of density values from the baseline distribution.
        q_density: 1-D array of density values from the current distribution.
            Must have the same length as ``p_density`` (evaluated on a
            shared support grid).

    Returns:
        The Jensen-Shannon divergence as a float in ``[0, 1]``.
    """
    p = np.asarray(p_density, dtype=float)
    q = np.asarray(q_density, dtype=float)

    p = np.clip(p, _EPSILON, None)
    q = np.clip(q, _EPSILON, None)

    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    m = np.clip(m, _EPSILON, None)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    jsd = float(np.clip(jsd, 0.0, 1.0))

    return jsd
