from __future__ import annotations

import numpy as np
import pytest

from evaluator.latent_drift import (
    EmbeddingBatch,
    LatentDriftEngine,
    LatentDriftResult,
    compute_jsd,
    compute_latent_drift,
    detect_latent_drift_events,
    evaluate_density,
    fit_kde,
    fit_pca,
    project_vectors,
)
from evaluator.temporal.models import DriftEvent

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def baseline_vectors():
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, size=(200, 20))


@pytest.fixture
def current_vectors():
    rng = np.random.RandomState(99)
    return rng.normal(0, 1, size=(200, 20))


@pytest.fixture
def shifted_vectors():
    rng = np.random.RandomState(77)
    return rng.normal(2.0, 1, size=(200, 20))


# ── PCA ──────────────────────────────────────────────────────────────────


def test_fit_pca_returns_fitted_model(baseline_vectors):
    pca = fit_pca(baseline_vectors, n_components=10)
    assert pca.n_components_ > 0
    assert pca.components_.shape[0] == pca.n_components_


def test_fit_pca_caps_at_rank(baseline_vectors):
    """When n_components exceeds the input rank, it is capped."""
    pca = fit_pca(baseline_vectors, n_components=100)
    assert pca.n_components_ <= min(baseline_vectors.shape)


def test_project_vectors_transforms(baseline_vectors):
    pca = fit_pca(baseline_vectors, n_components=5)
    projected = project_vectors(pca, baseline_vectors)
    assert projected.shape == (len(baseline_vectors), pca.n_components_)


def test_project_vectors_shape_mismatch():
    """Projecting with mismatched dimensions is handled by sklearn."""
    pca = fit_pca(np.random.RandomState(1).normal(0, 1, (50, 10)), n_components=5)
    wrong_dim = np.random.RandomState(2).normal(0, 1, (10, 5))
    # Should raise — dimensionality mismatch
    try:
        project_vectors(pca, wrong_dim)
        assert False, "Expected ValueError for dimension mismatch"
    except ValueError:
        pass


# ── KDE ──────────────────────────────────────────────────────────────────


def test_fit_kde_returns_gaussian_kde(baseline_vectors):
    pca = fit_pca(baseline_vectors, n_components=5)
    projected = project_vectors(pca, baseline_vectors)
    kde = fit_kde(projected)
    assert kde is not None


def test_kde_requires_minimum_samples():
    """KDE on a single sample should raise ValueError."""
    try:
        fit_kde(np.array([[0.1, 0.2, 0.3]]))
        assert False, "Expected ValueError for < 2 samples"
    except ValueError:
        pass


def test_evaluate_density_returns_correct_shape(baseline_vectors):
    pca = fit_pca(baseline_vectors, n_components=5)
    projected = project_vectors(pca, baseline_vectors)
    kde = fit_kde(projected)

    grid = np.linspace(-3, 3, 100).reshape(-1, 1)
    if projected.shape[1] == 1:
        densities = evaluate_density(kde, grid)
        assert len(densities) == 100
    else:
        grid_multi = np.random.RandomState(0).normal(0, 1, (100, projected.shape[1]))
        densities = evaluate_density(kde, grid_multi)
        assert len(densities) == 100


# ── JSD ──────────────────────────────────────────────────────────────────


def test_compute_jsd_identical_distributions():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    jsd = compute_jsd(p, p)
    assert jsd == pytest.approx(0.0, abs=1e-6)


def test_compute_jsd_identical_dense_distributions():
    rng = np.random.RandomState(42)
    p = np.abs(rng.normal(0, 1, 1000)) + 1e-6
    p = p / p.sum()
    q = p.copy()
    jsd = compute_jsd(p, q)
    assert jsd == pytest.approx(0.0, abs=1e-6)


def test_compute_jsd_different_distributions():
    p = np.array([0.9, 0.05, 0.03, 0.02])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    jsd = compute_jsd(p, q)
    assert jsd > 0.0


def test_compute_jsd_bounds():
    p = np.array([0.99, 0.01])
    q = np.array([0.01, 0.99])
    jsd = compute_jsd(p, q)
    assert 0.0 <= jsd <= 1.0


def test_compute_jsd_handles_zeros():
    p = np.array([0.0, 0.5, 0.5])
    q = np.array([0.5, 0.5, 0.0])
    jsd = compute_jsd(p, q)
    assert 0.0 <= jsd <= 1.0


# ── Engine ───────────────────────────────────────────────────────────────


def test_engine_identical_distributions_no_drift(baseline_vectors):
    engine = LatentDriftEngine(threshold=0.15)
    result = engine.fit_compute(baseline_vectors, baseline_vectors)
    assert isinstance(result, LatentDriftResult)
    assert result.drift_score == pytest.approx(0.0, abs=1e-6)
    assert result.drift_detected is False


def test_engine_detects_shifted_distribution(baseline_vectors, shifted_vectors):
    engine = LatentDriftEngine(threshold=0.15)
    result = engine.fit_compute(baseline_vectors, shifted_vectors)
    assert result.drift_score > result.threshold
    assert result.drift_detected is True


def test_engine_similar_distributions_no_drift(baseline_vectors, current_vectors):
    engine = LatentDriftEngine(threshold=0.15)
    result = engine.fit_compute(baseline_vectors, current_vectors)
    assert not result.drift_detected or result.drift_score < 0.2


def test_engine_returns_metadata(baseline_vectors, current_vectors):
    engine = LatentDriftEngine(threshold=0.15)
    result = engine.fit_compute(baseline_vectors, current_vectors)
    assert "pca_components" in result.metadata
    assert "explained_variance_ratio" in result.metadata
    assert result.n_samples_baseline == len(baseline_vectors)
    assert result.n_samples_current == len(current_vectors)


def test_engine_deterministic(baseline_vectors, shifted_vectors):
    engine1 = LatentDriftEngine(threshold=0.15)
    result1 = engine1.fit_compute(baseline_vectors, shifted_vectors)

    engine2 = LatentDriftEngine(threshold=0.15)
    result2 = engine2.fit_compute(baseline_vectors, shifted_vectors)

    assert result1.drift_score == result2.drift_score
    assert result1.drift_detected == result2.drift_detected


def test_engine_compute_drift_requires_fit(baseline_vectors, shifted_vectors):
    engine = LatentDriftEngine()
    try:
        engine.compute_drift(shifted_vectors)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_engine_to_drift_event(baseline_vectors, shifted_vectors):
    engine = LatentDriftEngine(threshold=0.15, metric="jsd")
    result = engine.fit_compute(baseline_vectors, shifted_vectors)
    event = engine.to_drift_event(result)

    assert isinstance(event, DriftEvent)
    assert event.metric_name == "latent_jsd"
    assert event.magnitude == pytest.approx(result.drift_score, abs=1e-6)
    assert event.metadata["latent_drift"] is True


# ── Integration ──────────────────────────────────────────────────────────


def test_compute_latent_drift_function(baseline_vectors, shifted_vectors):
    baseline = EmbeddingBatch(vectors=baseline_vectors)
    current = EmbeddingBatch(vectors=shifted_vectors)
    result = compute_latent_drift(baseline, current, threshold=0.15)
    assert result.drift_detected is True


def test_detect_latent_drift_events_returns_drift_event(
    baseline_vectors, shifted_vectors
):
    baseline = EmbeddingBatch(vectors=baseline_vectors)
    current = EmbeddingBatch(vectors=shifted_vectors)
    events = detect_latent_drift_events(baseline, current, threshold=0.15, metric="jsd")
    assert len(events) == 1
    assert isinstance(events[0], DriftEvent)
    assert events[0].metric_name == "latent_jsd"
    assert events[0].magnitude > 0.15


def test_detect_latent_drift_events_no_drift(baseline_vectors, current_vectors):
    baseline = EmbeddingBatch(vectors=baseline_vectors)
    current = EmbeddingBatch(vectors=current_vectors)
    events = detect_latent_drift_events(baseline, current, threshold=0.99)
    assert len(events) == 0


def test_detect_latent_drift_events_with_timestamp(baseline_vectors, shifted_vectors):
    from datetime import datetime, timezone

    baseline = EmbeddingBatch(vectors=baseline_vectors)
    current = EmbeddingBatch(
        vectors=shifted_vectors,
        timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    events = detect_latent_drift_events(baseline, current, threshold=0.15)
    assert len(events) == 1


# ── Edge Cases ──────────────────────────────────────────────────────────


def test_engine_small_sample_current():
    """Current batch with fewer samples than dimensions."""
    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, size=(50, 20))
    current = rng.normal(3, 1, size=(5, 20))  # only 5 samples

    engine = LatentDriftEngine(threshold=0.15)
    result = engine.fit_compute(baseline, current)
    assert isinstance(result, LatentDriftResult)
    assert result.n_samples_current == 5


def test_engine_high_threshold_no_drift(baseline_vectors, shifted_vectors):
    engine = LatentDriftEngine(threshold=0.99)
    result = engine.fit_compute(baseline_vectors, shifted_vectors)
    assert result.drift_detected is False


def test_engine_low_threshold_detects_drift(baseline_vectors, current_vectors):
    engine = LatentDriftEngine(threshold=0.001)
    result = engine.fit_compute(baseline_vectors, current_vectors)
    assert result.drift_detected is True or result.drift_score > 0.0


def test_jsd_deterministic_on_same_input():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.4, 0.3])
    r1 = compute_jsd(p, q)
    r2 = compute_jsd(p, q)
    assert r1 == r2
