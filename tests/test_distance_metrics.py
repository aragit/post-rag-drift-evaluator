from __future__ import annotations

import numpy as np
import pytest

from evaluator.latent_drift.distance import compute_mmd, compute_swd


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def baseline_vectors(rng):
    return rng.normal(0, 1, size=(200, 10))


@pytest.fixture
def current_vectors(rng):
    return rng.normal(0, 1, size=(200, 10))


@pytest.fixture
def shifted_vectors(rng):
    return rng.normal(3.0, 1, size=(200, 10))


# ── MMD ──────────────────────────────────────────────────────────────────


def test_mmd_identical_distributions(baseline_vectors):
    """MMD between identical distributions should be ~0."""
    score = compute_mmd(baseline_vectors, baseline_vectors)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_mmd_similar_distributions(baseline_vectors, current_vectors):
    """MMD between two samples from the same distribution should be low."""
    score = compute_mmd(baseline_vectors, current_vectors)
    assert score < 0.3


def test_mmd_shifted_distributions(baseline_vectors, shifted_vectors):
    """MMD between distinctly shifted distributions should be high."""
    score = compute_mmd(baseline_vectors, shifted_vectors)
    assert score > 0.2


def test_mmd_bounds(baseline_vectors, shifted_vectors):
    """MMD should be in [0, 1]."""
    score = compute_mmd(baseline_vectors, shifted_vectors)
    assert 0.0 <= score <= 1.0


def test_mmd_deterministic(baseline_vectors, current_vectors):
    """MMD should be deterministic for the same inputs."""
    s1 = compute_mmd(baseline_vectors, current_vectors)
    s2 = compute_mmd(baseline_vectors, current_vectors)
    assert s1 == pytest.approx(s2)


def test_mmd_with_explicit_gamma(baseline_vectors, current_vectors):
    """MMD should accept an explicit gamma parameter."""
    score = compute_mmd(baseline_vectors, current_vectors, gamma=0.5)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_mmd_dimension_mismatch(baseline_vectors):
    """MMD should raise ValueError on dimension mismatch."""
    wrong = np.random.RandomState(1).normal(0, 1, (50, 5))
    try:
        compute_mmd(baseline_vectors, wrong)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ── SWD ──────────────────────────────────────────────────────────────────


def test_swd_identical_distributions(baseline_vectors):
    """SWD between identical distributions should be ~0."""
    score = compute_swd(baseline_vectors, baseline_vectors)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_swd_similar_distributions(baseline_vectors, current_vectors):
    """SWD between two samples from the same distribution should be low."""
    score = compute_swd(baseline_vectors, current_vectors)
    assert score < 1.0


def test_swd_shifted_distributions(baseline_vectors, shifted_vectors):
    """SWD between distinctly shifted distributions should be high."""
    score = compute_swd(baseline_vectors, shifted_vectors)
    assert score > 0.5


def test_swd_returns_float(baseline_vectors, current_vectors):
    """SWD should return a Python float."""
    score = compute_swd(baseline_vectors, current_vectors)
    assert isinstance(score, float)


def test_swd_deterministic(baseline_vectors, current_vectors):
    """SWD should be deterministic (fixed seed)."""
    s1 = compute_swd(baseline_vectors, current_vectors)
    s2 = compute_swd(baseline_vectors, current_vectors)
    assert s1 == pytest.approx(s2)


def test_swd_custom_n_projections(baseline_vectors, current_vectors):
    """SWD should accept custom n_projections."""
    score = compute_swd(baseline_vectors, current_vectors, n_projections=50)
    assert isinstance(score, float)
    assert score >= 0.0


def test_swd_dimension_mismatch(baseline_vectors):
    """SWD should raise ValueError on dimension mismatch."""
    wrong = np.random.RandomState(1).normal(0, 1, (50, 5))
    try:
        compute_swd(baseline_vectors, wrong)
        assert False, "Expected ValueError"
    except ValueError:
        pass
