from __future__ import annotations

import os
import tempfile

from evaluator.metrics.results import DriftResult, QualityResult
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal import (
    DriftEvent,
    detect_drift_events,
    detect_drift_from_store,
    get_metric_series,
    get_metric_series_with_runs,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def make_record(
    metric_name: str,
    value: float,
    run_id: str | None = None,
    timestamp: float = 0.0,
) -> EvaluationRecord:
    """Build a minimal EvaluationRecord with a single metric."""
    if run_id is None:
        run_id = f"run-{value}"
    if metric_name == "js_divergence":
        metrics = [
            DriftResult(metric_name=metric_name, value=value, current_run_id=run_id)
        ]
    else:
        metrics = [QualityResult(metric_name=metric_name, value=value, run_id=run_id)]
    return EvaluationRecord(
        run_id=run_id,
        metrics=metrics,
        timestamp=timestamp,
    )


# ── Step 6 — Tests ───────────────────────────────────────────────────────


def test_series_extraction():
    """Test 1 — Series extraction returns correct ordered values."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "history.jsonl"))

    values = [0.05, 0.06, 0.07]
    for i, v in enumerate(values):
        store.save(make_record("js_divergence", v, timestamp=float(i)))

    series = get_metric_series(store, "js_divergence")
    assert len(series) == 3
    assert [v for _, v in series] == values
    # Sorted by timestamp
    timestamps = [ts for ts, _ in series]
    assert timestamps == [0.0, 1.0, 2.0]


def test_series_with_run_ids():
    """Test 2 — Series with run_ids align timestamps."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "history.jsonl"))

    for i, v in enumerate([0.05, 0.06, 0.07]):
        store.save(
            make_record("js_divergence", v, run_id=f"run-{i}", timestamp=float(i))
        )

    series = get_metric_series_with_runs(store, "js_divergence")
    assert len(series) == 3
    for ts, val, run_id in series:
        assert run_id.startswith("run-")
        assert isinstance(ts, float)
        assert isinstance(val, float)


def test_no_drift_case():
    """Test 3 — No drift detected when values stay within threshold."""
    series = [
        (1.0, 0.05, "r1"),
        (2.0, 0.06, "r2"),
        (3.0, 0.07, "r3"),
        (4.0, 0.06, "r4"),
        (5.0, 0.05, "r5"),
    ]
    events = detect_drift_events(series, window_size=3, threshold=0.15)
    assert len(events) == 0


def test_drift_detected():
    """Test 4 — Drift detected when values shift significantly."""
    series = [
        (1.0, 0.05, "r1"),
        (2.0, 0.06, "r2"),
        (3.0, 0.07, "r3"),
        (4.0, 0.40, "r4"),
        (5.0, 0.42, "r5"),
        (6.0, 0.45, "r6"),
    ]
    events = detect_drift_events(series, window_size=3, threshold=0.15)
    assert len(events) >= 1


def test_drift_event_correctness():
    """Test 5 — DriftEvent fields are correctly populated."""
    series = [
        (1.0, 0.05, "r1"),
        (2.0, 0.06, "r2"),
        (3.0, 0.07, "r3"),
        (4.0, 0.40, "r4"),
        (5.0, 0.42, "r5"),
        (6.0, 0.45, "r6"),
    ]
    events = detect_drift_events(series, window_size=3, threshold=0.01)
    assert len(events) >= 1

    event = events[0]
    assert event.event_id is not None
    assert event.metric_name == "unknown"  # default since not specified
    assert event.start_timestamp == 4.0
    assert event.end_timestamp == 6.0
    assert event.magnitude > 0.01
    assert "r4" in event.involved_run_ids
    assert event.metadata["method"] == "mean_shift"


def test_drift_event_serialization():
    """DriftEvent must serialize and deserialize without loss."""
    event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=1.0,
        end_timestamp=6.0,
        magnitude=0.35,
        involved_run_ids=["r1", "r2", "r3"],
        metadata={"method": "mean_shift"},
    )
    d = event.to_dict()
    assert d["event_id"] is not None
    assert d["metric_name"] == "js_divergence"

    restored = DriftEvent.from_dict(d)
    assert restored.event_id == event.event_id
    assert restored.metric_name == event.metric_name
    assert restored.start_timestamp == 1.0
    assert restored.end_timestamp == 6.0
    assert restored.magnitude == 0.35
    assert restored.involved_run_ids == ["r1", "r2", "r3"]
    assert restored.metadata == {"method": "mean_shift"}


def test_drift_event_auto_id():
    """DriftEvent must auto-generate event_id."""
    event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=1.0,
        end_timestamp=2.0,
        magnitude=0.1,
    )
    assert event.event_id is not None
    # Should be a valid UUID string
    import uuid

    uuid.UUID(event.event_id)


def test_drift_from_store_integration():
    """Test 6 — Full integration with JSONHistoryStore."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)

    # Save records with a clear drift pattern
    values = [0.05, 0.06, 0.07, 0.40, 0.42, 0.45]
    for i, v in enumerate(values):
        store.save(
            make_record("js_divergence", v, run_id=f"run-{i}", timestamp=float(i))
        )

    events = detect_drift_from_store(
        store, metric_name="js_divergence", window_size=3, threshold=0.15
    )
    assert len(events) >= 1

    event = events[0]
    assert event.metric_name == "js_divergence"
    assert event.start_timestamp >= 0.0
    assert event.magnitude > 0.15


def test_empty_series_no_drift():
    """No series → no events."""
    events = detect_drift_events([], window_size=3, threshold=0.15)
    assert len(events) == 0


def test_short_series_no_drift():
    """Series too short for windows → no events."""
    series = [(1.0, 0.05, "r1"), (2.0, 0.06, "r2")]
    events = detect_drift_events(series, window_size=3, threshold=0.15)
    assert len(events) == 0
