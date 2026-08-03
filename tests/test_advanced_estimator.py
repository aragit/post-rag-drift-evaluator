from __future__ import annotations

import os
import tempfile

import pytest

from evaluator.counterfactual.estimator import (
    estimate_metric_after_intervention,
)
from evaluator.metrics.results import DriftResult
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal.models import DriftEvent


def make_drift_record(
    run_id: str,
    timestamp: float,
    value: float,
    system_version: str = "0.1.0",
    metadata: dict | None = None,
) -> EvaluationRecord:
    if metadata is None:
        metadata = {}
    return EvaluationRecord(
        run_id=run_id,
        timestamp=timestamp,
        system_version=system_version,
        metadata=metadata,
        metrics=[
            DriftResult(metric_name="js_divergence", value=value, current_run_id=run_id)
        ],
    )


# ── "mean" method tests ───────────────────────────────────────────────────


def test_estimate_mean_method():
    """Mean method returns arithmetic mean of pre-window values."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i in range(3):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.10)
        )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="mean"
    )
    assert estimate == pytest.approx(0.10)


# ── "ewma" method tests ───────────────────────────────────────────────────


def test_estimate_ewma_method():
    """EWMA gives higher weight to more recent observations."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Rising baseline: 0.05, 0.10, 0.15, 0.20
    for i, v in enumerate([0.05, 0.10, 0.15, 0.20]):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=v)
        )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=4.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="ewma", alpha=0.3
    )

    # EWMA with alpha=0.3, forward iteration:
    # EWMA_0 = 0.05
    # EWMA_1 = 0.3*0.10 + 0.7*0.05 = 0.065
    # EWMA_2 = 0.3*0.15 + 0.7*0.065 = 0.0905
    # EWMA_3 = 0.3*0.20 + 0.7*0.0905 = 0.12335
    expected = 0.3 * 0.20 + 0.7 * (0.3 * 0.15 + 0.7 * (0.3 * 0.10 + 0.7 * 0.05))
    assert estimate == pytest.approx(expected)


def test_estimate_ewma_weight_recency():
    """EWMA with high alpha should be closer to the most recent value."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i, v in enumerate([0.05, 0.10, 0.15, 0.20]):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=v)
        )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=4.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    # With alpha=1.0, EWMA should return the last value
    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="ewma", alpha=1.0
    )
    assert estimate == pytest.approx(0.20)


# ── "trend_adjusted" method tests ─────────────────────────────────────────


def test_estimate_trend_adjusted_uses_extrapolation():
    """Trend-adjusted estimator extrapolates linear trend to drift start."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Linearly increasing values at timestamps 0, 1, 2
    for i, v in enumerate([0.10, 0.15, 0.20]):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=v)
        )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="trend_adjusted"
    )

    # Linear regression: slope = 0.05, intercept = 0.10
    # At t=3: 0.05 * 3 + 0.10 = 0.25
    assert estimate == 0.25


def test_estimate_trend_adjusted_falls_back_to_ewma_for_few_points():
    """With <3 pre-window points, falls back to EWMA."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Only 2 points
    store.save(make_drift_record("r0", timestamp=0.0, value=0.10))
    store.save(make_drift_record("r1", timestamp=1.0, value=0.20))

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=2.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    # Should not raise — falls back to ewma
    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="trend_adjusted"
    )

    # EWMA of [0.10, 0.20] with alpha=0.3, forward:
    # EWMA_0 = 0.10
    # EWMA_1 = 0.3*0.20 + 0.7*0.10 = 0.13
    assert estimate == pytest.approx(0.13)


def test_estimate_trend_adjusted_more_accurate_than_mean_for_rising_series():
    """Trend-adjusted is closer to true value than simple mean for trend."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Linearly increasing: 0.10, 0.20, 0.30, 0.40, 0.50
    for i, v in enumerate([0.10, 0.20, 0.30, 0.40, 0.50]):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=v)
        )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,
        end_timestamp=7.0,
        magnitude=0.60,
    )

    mean_estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="mean"
    )
    trend_estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="trend_adjusted"
    )

    # True trend at t=5 is 0.10 + 0.10*5 = 0.60
    true_value = 0.60

    mean_error = abs(mean_estimate - true_value)
    trend_error = abs(trend_estimate - true_value)

    assert trend_error < mean_error


# ── Fallback / edge case tests ────────────────────────────────────────────


def test_estimate_fallback_to_all_values():
    """No pre-window records: falls back to mean of all values."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(make_drift_record("r1", timestamp=10.0, value=0.3))
    store.save(make_drift_record("r2", timestamp=11.0, value=0.5))

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,  # all records are after this
        end_timestamp=15.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="trend_adjusted"
    )
    assert estimate == 0.4


def test_estimate_empty_store():
    """Empty store returns 0.0."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "empty.jsonl"))

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=15.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(
        drift, store, "js_divergence", method="mean"
    )
    assert estimate == 0.0


def test_estimate_deterministic():
    """Same inputs always produce the same estimate."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i, v in enumerate([0.10, 0.20, 0.30]):
        store.save(make_drift_record(f"r{i}", timestamp=i, value=v))

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.5,
    )

    e1 = estimate_metric_after_intervention(drift, store, "js_divergence", method="trend_adjusted")
    e2 = estimate_metric_after_intervention(drift, store, "js_divergence", method="trend_adjusted")
    assert e1 == e2
