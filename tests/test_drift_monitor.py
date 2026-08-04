from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from evaluator.drift_monitor import DriftMonitor, _freedman_diaconis_bin_count

# ── Continuous KL divergence integration step (dx) ─────────────────────────


def test_kl_divergence_is_grid_invariant():
    """KL divergence approximated as a Riemann sum must be ~invariant to the
    evaluation grid resolution once scaled by ``dx``."""
    rng = np.random.RandomState(0)
    baseline = rng.normal(0.0, 1.0, 500)
    current = rng.normal(0.6, 1.0, 500)

    kl_100 = DriftMonitor._kl_divergence_kde(baseline, current, n_points=100)
    kl_500 = DriftMonitor._kl_divergence_kde(baseline, current, n_points=500)

    assert kl_100 >= 0.0
    assert kl_500 >= 0.0
    # Same integrand, different grid resolution -> converged values.
    assert kl_100 == pytest.approx(kl_500, abs=0.1)


def test_kl_divergence_identical_distributions_is_zero():
    rng = np.random.RandomState(42)
    sample = rng.normal(0.0, 1.0, 300)
    kl = DriftMonitor._kl_divergence_kde(sample, sample, n_points=200)
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_shifted_distributions_is_positive():
    rng = np.random.RandomState(1)
    baseline = rng.normal(0.0, 1.0, 500)
    current = rng.normal(2.0, 1.0, 500)
    kl = DriftMonitor._kl_divergence_kde(baseline, current)
    assert kl > 0.0


# ── Freedman-Diaconis binning ────────────────────────────────────────────────


def test_fd_bin_count_flat_array_no_division_by_zero():
    flat = np.full(100, 5.0)
    assert _freedman_diaconis_bin_count(flat) == 20


def test_fd_bin_count_single_element():
    assert _freedman_diaconis_bin_count(np.array([42.0])) == 20


def test_fd_bin_count_returns_int_within_clamp_range():
    rng = np.random.RandomState(3)
    data = rng.normal(0, 1, 5000)
    bins = _freedman_diaconis_bin_count(data)
    assert isinstance(bins, int)
    assert 10 <= bins <= 100


def test_fd_bin_count_sane_with_outliers():
    rng = np.random.RandomState(11)
    clean = rng.normal(0, 1, 500)
    with_outliers = np.concatenate([clean, rng.normal(20, 1, 50)])
    assert 10 <= _freedman_diaconis_bin_count(with_outliers) <= 100


def test_jensen_shannon_distributions_handles_constant_projection():
    """Constant (zero-variance) projection must not crash and yields uniform,
    normalized, matching distributions."""
    monitor = DriftMonitor()
    constant = np.full((10, 4), 0.5)
    p, q = monitor._jensen_shannon_distributions(constant, constant)
    assert p.shape == q.shape
    assert np.allclose(p, q)
    assert np.isclose(p.sum(), 1.0)


# ── JSD bounds ─────────────────────────────────────────────────────────────


def test_jensen_shannon_drift_strictly_bounded_in_unit_interval():
    rng = np.random.RandomState(7)
    base = rng.randn(80, 16)
    curr = rng.normal(2.0, 1.0, (80, 16))

    monitor = DriftMonitor()
    base_df = pl.DataFrame({"embedding": base.tolist()})
    curr_df = pl.DataFrame({"embedding": curr.tolist()})

    js_identical, _ = monitor.compute_jensen_shannon_drift(base_df, base_df)
    js_shifted, _ = monitor.compute_jensen_shannon_drift(base_df, curr_df)

    assert 0.0 <= js_identical <= 1.0
    assert 0.0 <= js_shifted <= 1.0
    assert js_identical < js_shifted


def test_jensen_shannon_drift_handles_empty_baseline():
    monitor = DriftMonitor()
    empty = pl.DataFrame({"embedding": []})
    current = pl.DataFrame({"embedding": np.random.randn(10, 8).tolist()})
    with pytest.raises(Exception):
        monitor.compute_jensen_shannon_drift(empty, current)
