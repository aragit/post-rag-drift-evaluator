import numpy as np
import polars as pl

from evaluator.drift_monitor import DriftMonitor


def test_jsd_is_symmetric():
    rng = np.random.RandomState(42)
    baseline = rng.randn(50, 32)
    current = rng.randn(50, 32)

    monitor = DriftMonitor()
    js1, _ = monitor.compute_jensen_shannon_drift(
        pl.DataFrame({"embedding": baseline.tolist()}),
        pl.DataFrame({"embedding": current.tolist()}),
    )
    js2, _ = monitor.compute_jensen_shannon_drift(
        pl.DataFrame({"embedding": current.tolist()}),
        pl.DataFrame({"embedding": baseline.tolist()}),
    )

    assert abs(js1 - js2) < 1e-10


def test_jsd_range_bounded():
    rng = np.random.RandomState(42)
    baseline = rng.randn(50, 32)
    current = rng.randn(50, 32)

    monitor = DriftMonitor()
    js_score, _ = monitor.compute_jensen_shannon_drift(
        pl.DataFrame({"embedding": baseline.tolist()}),
        pl.DataFrame({"embedding": current.tolist()}),
    )

    assert 0.0 <= js_score <= 1.0


def test_drift_detects_known_shift():
    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, (100, 64))
    current = rng.normal(3.0, 1, (100, 64))

    monitor = DriftMonitor()
    _, is_drifted = monitor.compute_jensen_shannon_drift(
        pl.DataFrame({"embedding": baseline.tolist()}),
        pl.DataFrame({"embedding": current.tolist()}),
    )

    assert is_drifted


def test_drift_false_on_identical():
    rng = np.random.RandomState(42)
    dist = rng.randn(100, 32)

    monitor = DriftMonitor()
    js_score, is_drifted = monitor.compute_jensen_shannon_drift(
        pl.DataFrame({"embedding": dist.tolist()}),
        pl.DataFrame({"embedding": dist.tolist()}),
    )

    assert js_score < 0.1
    assert not is_drifted


def test_pca_stability():
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(42)
    matrix = rng.randn(50, 16)

    pca1 = PCA(n_components=2)
    pca1.fit(matrix)
    result1 = pca1.transform(matrix)

    pca2 = PCA(n_components=2)
    pca2.fit(matrix)
    result2 = pca2.transform(matrix)

    np.testing.assert_array_almost_equal(result1, result2, decimal=6)
