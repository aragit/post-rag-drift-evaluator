from __future__ import annotations

import os
import tempfile

from evaluator.causal import (
    CausalAttribution,
    CausalFactor,
    ChangeEvent,
    attribute_drift,
    build_drift_features,
    extract_change_events,
    score_causal_impact,
)
from evaluator.metrics.results import DriftResult
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal.models import DriftEvent

# ── Helpers ──────────────────────────────────────────────────────────────


def make_record(
    run_id: str,
    timestamp: float,
    system_version: str | None = None,
    metadata: dict | None = None,
    metrics: list | None = None,
) -> EvaluationRecord:
    if metrics is None:
        metrics = [
            DriftResult(metric_name="js_divergence", value=0.01, current_run_id=run_id)
        ]
    if metadata is None:
        metadata = {}
    return EvaluationRecord(
        run_id=run_id,
        metrics=metrics,
        metadata=metadata,
        system_version=system_version,
        timestamp=timestamp,
    )


def make_drift_record(
    run_id: str,
    timestamp: float,
    value: float,
    system_version: str = "0.1.0",
) -> EvaluationRecord:
    return make_record(
        run_id=run_id,
        timestamp=timestamp,
        system_version=system_version,
        metrics=[
            DriftResult(metric_name="js_divergence", value=value, current_run_id=run_id)
        ],
    )


# ── Step 1: Models ───────────────────────────────────────────────────────


def test_change_event_auto_id():
    ev = ChangeEvent(timestamp=1.0, run_id="r1", change_type="model_update")
    assert ev.change_id is not None
    assert ev.run_id == "r1"
    assert ev.change_type == "model_update"

    restored = ChangeEvent.from_dict(ev.to_dict())
    assert restored.change_id == ev.change_id
    assert restored.run_id == ev.run_id


def test_causal_factor_serialization():
    factor = CausalFactor(
        factor_name="model_update",
        score=0.87,
        related_run_ids=["run_45"],
        metadata={},
    )
    d = factor.to_dict()
    assert d["factor_name"] == "model_update"
    assert d["score"] == 0.87

    restored = CausalFactor.from_dict(d)
    assert restored.factor_name == factor.factor_name
    assert restored.score == factor.score
    assert restored.related_run_ids == ["run_45"]


def test_causal_attribution_serialization():
    attr = CausalAttribution(
        drift_event_id="abc123",
        metric_name="js_divergence",
        factors=[
            CausalFactor(factor_name="model_update", score=0.87),
            CausalFactor(factor_name="config_change", score=0.42),
        ],
        confidence=0.68,
        metadata={"method": "heuristic"},
    )
    d = attr.to_dict()
    assert d["attribution_id"] is not None
    assert d["drift_event_id"] == "abc123"
    assert len(d["factors"]) == 2

    restored = CausalAttribution.from_dict(d)
    assert restored.attribution_id == attr.attribution_id
    assert restored.drift_event_id == "abc123"
    assert restored.metric_name == "js_divergence"
    assert len(restored.factors) == 2
    assert restored.factors[0].factor_name == "model_update"


# ── Step 2: Change Extraction ────────────────────────────────────────────


def test_change_extraction_detects_model_change():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(
        make_record(
            "r1",
            timestamp=1.0,
            system_version="0.1.0",
            metadata={"pipeline_name": "NaiveRAG"},
        )
    )
    store.save(
        make_record(
            "r2",
            timestamp=2.0,
            system_version="0.2.0",
            metadata={"pipeline_name": "NaiveRAG"},
        )
    )

    changes = extract_change_events(store)
    assert len(changes) == 1
    assert changes[0].change_type == "version_change"
    assert changes[0].run_id == "r2"
    assert changes[0].timestamp == 2.0


def test_change_extraction_detects_config_change():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(make_record("r1", timestamp=1.0, metadata={"temperature": 0.7}))
    store.save(make_record("r2", timestamp=2.0, metadata={"temperature": 0.9}))

    changes = extract_change_events(store)
    assert len(changes) == 1
    assert changes[0].change_type == "config_change"


def test_change_extraction_no_changes():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(
        make_record("r1", timestamp=1.0, system_version="1.0", metadata={"k": "v"})
    )
    store.save(
        make_record("r2", timestamp=2.0, system_version="1.0", metadata={"k": "v"})
    )

    changes = extract_change_events(store)
    assert len(changes) == 0


def test_change_extraction_sorted_chronologically():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Save in non-sorted order
    store.save(make_record("r3", timestamp=3.0, system_version="0.3.0"))
    store.save(make_record("r1", timestamp=1.0, system_version="0.1.0"))
    store.save(make_record("r2", timestamp=2.0, system_version="0.2.0"))

    changes = extract_change_events(store)
    assert len(changes) == 2
    assert changes[0].timestamp <= changes[1].timestamp


def test_change_extraction_empty_store():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "missing.jsonl"))
    changes = extract_change_events(store)
    assert len(changes) == 0


# ── Step 3: Feature Builder ────────────────────────────────────────────


def test_feature_builder_in_window():
    drift = DriftEvent(
        event_id="ev1",
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=20.0,
        magnitude=0.35,
        involved_run_ids=["r3"],
    )
    change = ChangeEvent(
        timestamp=15.0,  # inside the window
        run_id="r3",
        change_type="model_update",
    )
    features = build_drift_features(drift, [change])
    assert len(features) == 1
    assert features[0]["in_window"] is True
    assert features[0]["time_delta"] == 0.0
    assert features[0]["change_type"] == "model_update"
    assert features[0]["drift_magnitude"] == 0.35


def test_feature_builder_outside_window():
    drift = DriftEvent(
        event_id="ev1",
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=20.0,
        magnitude=0.35,
        involved_run_ids=["r3"],
    )
    change = ChangeEvent(
        timestamp=5.0,  # before the window
        run_id="r1",
        change_type="config_change",
    )
    features = build_drift_features(drift, [change])
    assert len(features) == 1
    assert features[0]["in_window"] is False
    assert features[0]["time_delta"] == 5.0  # 10.0 - 5.0


def test_feature_builder_empty_changes():
    drift = DriftEvent(
        event_id="ev1",
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=20.0,
        magnitude=0.35,
        involved_run_ids=["r3"],
    )
    features = build_drift_features(drift, [])
    assert len(features) == 0


def test_feature_builder_multiple_changes():
    drift = DriftEvent(
        event_id="ev1",
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=20.0,
        magnitude=0.35,
        involved_run_ids=["r2", "r3"],
    )
    changes = [
        ChangeEvent(timestamp=5.0, run_id="r1", change_type="config_change"),
        ChangeEvent(timestamp=15.0, run_id="r2", change_type="model_update"),
        ChangeEvent(timestamp=25.0, run_id="r4", change_type="version_change"),
    ]
    features = build_drift_features(drift, changes)
    assert len(features) == 3
    # Should be sorted by timestamp
    assert features[0]["run_id"] == "r1"
    assert features[1]["run_id"] == "r2"
    assert features[2]["run_id"] == "r4"
    assert features[1]["in_window"] is True
    assert features[0]["in_window"] is False


# ── Step 4: Attribution Scoring ────────────────────────────────────────


def test_score_causal_impact_ranks_descending():
    features = [
        {
            "change_id": "c1",
            "run_id": "r1",
            "change_type": "config_change",
            "change_type_weight": 0.6,
            "time_delta": 10.0,
            "in_window": False,
            "drift_magnitude": 0.3,
            "details": {},
        },
        {
            "change_id": "c2",
            "run_id": "r2",
            "change_type": "model_update",
            "change_type_weight": 1.0,
            "time_delta": 0.0,
            "in_window": True,
            "drift_magnitude": 0.3,
            "details": {},
        },
    ]
    factors = score_causal_impact(features)
    assert len(factors) == 2
    assert factors[0].score >= factors[1].score
    assert factors[0].factor_name == "model_update"


def test_score_causal_impact_in_window_scores_higher():
    features_in = [
        {
            "change_id": "c1",
            "run_id": "r1",
            "change_type": "model_update",
            "change_type_weight": 1.0,
            "time_delta": 0.0,
            "in_window": True,
            "drift_magnitude": 0.5,
            "details": {},
        },
    ]
    features_out = [
        {
            "change_id": "c1",
            "run_id": "r1",
            "change_type": "model_update",
            "change_type_weight": 1.0,
            "time_delta": 100.0,
            "in_window": False,
            "drift_magnitude": 0.5,
            "details": {},
        },
    ]
    score_in = score_causal_impact(features_in)[0].score
    score_out = score_causal_impact(features_out)[0].score
    assert score_in > score_out


def test_score_causal_impact_empty():
    factors = score_causal_impact([])
    assert len(factors) == 0


def test_score_causal_impact_deterministic():
    features = [
        {
            "change_id": "c1",
            "run_id": "r1",
            "change_type": "model_update",
            "change_type_weight": 1.0,
            "time_delta": 0.0,
            "in_window": True,
            "drift_magnitude": 0.3,
            "details": {},
        },
        {
            "change_id": "c2",
            "run_id": "r2",
            "change_type": "config_change",
            "change_type_weight": 0.6,
            "time_delta": 5.0,
            "in_window": False,
            "drift_magnitude": 0.3,
            "details": {},
        },
    ]
    f1 = score_causal_impact(features)
    f2 = score_causal_impact(features)
    assert [f.score for f in f1] == [f.score for f in f2]


# ── Step 5: Full Attribution ───────────────────────────────────────────


def test_attribute_drift_with_changes():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # 3 stable runs
    for i in range(3):
        store.save(make_drift_record(f"r{i}", timestamp=i, value=0.05))
    # 3 drifted runs after a model update
    for i in range(3, 6):
        store.save(
            make_drift_record(
                f"r{i}",
                timestamp=i,
                value=0.45,
                system_version="0.2.0",
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.4,
        involved_run_ids=["r3", "r4", "r5"],
    )

    attribution = attribute_drift(drift_event, store)
    assert isinstance(attribution, CausalAttribution)
    assert attribution.drift_event_id == drift_event.event_id
    assert attribution.metric_name == "js_divergence"
    assert len(attribution.factors) > 0
    assert 0.0 <= attribution.confidence <= 1.0
    assert attribution.factors[0].score >= attribution.factors[-1].score


def test_attribute_drift_no_changes():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i in range(6):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.05, system_version="1.0")
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.4,
        involved_run_ids=["r3", "r4", "r5"],
    )

    attribution = attribute_drift(drift_event, store)
    assert len(attribution.factors) == 0
    assert attribution.confidence == 0.0


def test_attribute_drift_serialization_round_trip():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(make_drift_record("r0", timestamp=0.0, value=0.05))
    store.save(
        make_drift_record("r1", timestamp=1.0, value=0.45, system_version="0.2.0")
    )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=1.0,
        end_timestamp=1.0,
        magnitude=0.4,
        involved_run_ids=["r1"],
    )

    attribution = attribute_drift(drift_event, store)
    d = attribution.to_dict()
    restored = CausalAttribution.from_dict(d)
    assert restored.drift_event_id == attribution.drift_event_id
    assert restored.metric_name == attribution.metric_name
    assert restored.confidence == attribution.confidence
    assert len(restored.factors) == len(attribution.factors)
