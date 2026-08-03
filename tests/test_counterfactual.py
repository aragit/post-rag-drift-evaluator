from __future__ import annotations

import os
import tempfile

from evaluator.causal import (
    CausalAttribution,
    attribute_drift,
)
from evaluator.causal.change_extractor import extract_change_events
from evaluator.counterfactual import (
    CounterfactualResult,
    CounterfactualScenario,
    Intervention,
    apply_intervention,
    apply_scenario,
    build_counterfactual_scenarios,
    estimate_metric_after_intervention,
    run_counterfactual_analysis,
)
from evaluator.metrics.results import DriftResult
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal.models import DriftEvent

# ── Helpers ──────────────────────────────────────────────────────────────


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


def _make_store_with_drift() -> tuple[JSONHistoryStore, DriftEvent]:
    """Build a store with a model-change drift pattern.

    Returns:
        (store, drift_event)
    """
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i in range(3):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.05, system_version="0.1.0")
        )

    store.save(
        make_drift_record("r3", timestamp=3.0, value=0.30, system_version="0.2.0")
    )
    store.save(
        make_drift_record("r4", timestamp=4.0, value=0.45, system_version="0.2.0")
    )
    store.save(
        make_drift_record("r5", timestamp=5.0, value=0.50, system_version="0.2.0")
    )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
        involved_run_ids=["r3", "r4", "r5"],
    )
    return store, drift_event


def _get_first_change_id(store: JSONHistoryStore) -> str:
    changes = extract_change_events(store)
    return changes[0].change_id if changes else ""


# ── Step 1: Models ───────────────────────────────────────────────────────


def test_intervention_auto_id():
    iv = Intervention(action="remove", change_id="c1")
    assert iv.intervention_id is not None
    assert iv.action == "remove"
    assert iv.change_id == "c1"


def test_intervention_serialization():
    iv = Intervention(
        action="modify",
        change_id="c42",
        override_metadata={"temperature": 0.5},
    )
    d = iv.to_dict()
    assert d["action"] == "modify"
    assert d["change_id"] == "c42"
    assert d["override_metadata"]["temperature"] == 0.5
    assert d["intervention_id"] is not None

    restored = Intervention.from_dict(d)
    assert restored.intervention_id == iv.intervention_id
    assert restored.change_id == "c42"
    assert restored.action == "modify"


def test_counterfactual_scenario_serialization():
    iv = Intervention(action="remove", change_id="c1")
    scenario = CounterfactualScenario(
        drift_event_id="ev1",
        interventions=[iv],
        description="test scenario",
    )
    d = scenario.to_dict()
    assert d["drift_event_id"] == "ev1"
    assert len(d["interventions"]) == 1
    assert d["description"] == "test scenario"
    assert d["scenario_id"] is not None

    restored = CounterfactualScenario.from_dict(d)
    assert restored.scenario_id == scenario.scenario_id
    assert len(restored.interventions) == 1


def test_counterfactual_result_serialization():
    result = CounterfactualResult(
        scenario_id="s1",
        original_metric=0.45,
        counterfactual_metric=0.05,
        delta=0.40,
        confidence=0.87,
        metadata={"metric_name": "js_divergence"},
    )
    d = result.to_dict()
    assert d["original_metric"] == 0.45
    assert d["counterfactual_metric"] == 0.05
    assert d["delta"] == 0.40
    assert d["confidence"] == 0.87
    assert d["result_id"] is not None

    restored = CounterfactualResult.from_dict(d)
    assert restored.result_id == result.result_id
    assert restored.scenario_id == "s1"
    assert restored.delta == 0.40


# ── Step 2: Scenario Builder ─────────────────────────────────────────────


def test_build_scenarios_from_attribution():
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)
    scenarios = build_counterfactual_scenarios(attribution, top_k=3)

    assert len(scenarios) >= 1
    for scenario in scenarios:
        assert scenario.drift_event_id == drift_event.event_id
        assert len(scenario.interventions) >= 1
        for iv in scenario.interventions:
            assert iv.action == "remove"
            # change_id must correspond to one of the attribution factors
            factor_change_ids = {
                f.metadata.get("change_id", "") for f in attribution.factors
            }
            assert iv.change_id in factor_change_ids


def test_build_scenarios_includes_combined():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Create two changes to get 2 factors
    for i in range(2):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.05, system_version="0.1.0")
        )
    store.save(
        make_drift_record("r2", timestamp=2.0, value=0.30, system_version="0.2.0")
    )
    store.save(
        make_drift_record("r3", timestamp=3.0, value=0.45, system_version="0.3.0")
    )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=2.0,
        end_timestamp=3.0,
        magnitude=0.40,
        involved_run_ids=["r2", "r3"],
    )

    attribution = attribute_drift(drift_event, store)
    scenarios = build_counterfactual_scenarios(attribution, top_k=3)

    has_combined = any(
        "all" in s.description.lower() and len(s.interventions) > 1 for s in scenarios
    )
    assert has_combined


def test_build_scenarios_empty_attribution():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[],
    )
    scenarios = build_counterfactual_scenarios(attr)
    assert len(scenarios) == 0


def test_build_scenarios_top_k_respected():
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)
    scenarios = build_counterfactual_scenarios(attribution, top_k=1)

    indiv_scenarios = [s for s in scenarios if len(s.interventions) == 1]
    assert len(indiv_scenarios) == 1


# ── Step 3: Intervention Engine ──────────────────────────────────────────


def test_apply_intervention_does_not_mutate_original():
    store, drift_event = _make_store_with_drift()
    change_id = _get_first_change_id(store)
    original_records = sorted(store.load_all(), key=lambda r: r.timestamp or 0.0)
    original_system_version = original_records[3].system_version  # r3 = 0.2.0

    iv = Intervention(action="remove", change_id=change_id)
    new_store = apply_intervention(store, iv)

    # Original store unchanged
    same_records = sorted(store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert same_records[3].system_version == original_system_version

    # New store has reverted version
    new_records = sorted(new_store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert new_records[3].system_version == "0.1.0"


def test_apply_intervention_remove_reverts_metadata():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(
        make_drift_record(
            "r1", timestamp=1.0, value=0.05, metadata={"temperature": 0.7}
        )
    )
    store.save(
        make_drift_record(
            "r2", timestamp=2.0, value=0.30, metadata={"temperature": 0.9}
        )
    )

    change_id = _get_first_change_id(store)
    iv = Intervention(action="remove", change_id=change_id)
    new_store = apply_intervention(store, iv)

    new_records = sorted(new_store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert new_records[1].metadata.get("temperature") == 0.7


def test_apply_intervention_modify_overrides_metadata():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    store.save(
        make_drift_record(
            "r1", timestamp=1.0, value=0.05, metadata={"temperature": 0.7}
        )
    )
    store.save(
        make_drift_record(
            "r2", timestamp=2.0, value=0.30, metadata={"temperature": 0.9}
        )
    )

    change_id = _get_first_change_id(store)
    iv = Intervention(
        action="modify",
        change_id=change_id,
        override_metadata={"temperature": 0.5},
    )
    new_store = apply_intervention(store, iv)

    new_records = sorted(new_store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert new_records[1].metadata.get("temperature") == 0.5


def test_apply_intervention_unknown_change_id():
    store, drift_event = _make_store_with_drift()
    iv = Intervention(action="remove", change_id="nonexistent-change")
    new_store = apply_intervention(store, iv)

    original = sorted(store.load_all(), key=lambda r: r.timestamp or 0.0)
    modified = sorted(new_store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert len(original) == len(modified)
    for o, m in zip(original, modified):
        assert o.system_version == m.system_version


def test_apply_scenario_multiple_interventions():
    store, drift_event = _make_store_with_drift()
    change_id = _get_first_change_id(store)

    scenario = CounterfactualScenario(
        drift_event_id=drift_event.event_id,
        interventions=[
            Intervention(action="remove", change_id=change_id),
        ],
    )
    new_store = apply_scenario(store, scenario)
    assert new_store is not store
    assert len(new_store.load_all()) == len(store.load_all())


# ── Step 4: Metric Estimator ─────────────────────────────────────────────


def test_estimate_metric_pre_window_mean():
    store, drift_event = _make_store_with_drift()
    change_id = _get_first_change_id(store)
    new_store = apply_intervention(
        store, Intervention(action="remove", change_id=change_id)
    )

    estimate = estimate_metric_after_intervention(
        drift_event, new_store, "js_divergence"
    )

    # Pre-window records (r0, r1, r2) all have value 0.05
    assert round(estimate, 6) == 0.05


def test_estimate_metric_fallback_to_all_values():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # All records in the drift window — no pre-window records
    store.save(
        make_drift_record("r1", timestamp=10.0, value=0.3, system_version="0.2.0")
    )
    store.save(
        make_drift_record("r2", timestamp=11.0, value=0.5, system_version="0.2.0")
    )

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=15.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(drift, store, "js_divergence")
    assert estimate == 0.4


def test_estimate_metric_empty_store():
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "empty.jsonl"))

    drift = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=10.0,
        end_timestamp=15.0,
        magnitude=0.5,
    )

    estimate = estimate_metric_after_intervention(drift, store, "js_divergence")
    assert estimate == 0.0


def test_estimate_metric_deterministic():
    store, drift_event = _make_store_with_drift()
    change_id = _get_first_change_id(store)
    new_store = apply_intervention(
        store, Intervention(action="remove", change_id=change_id)
    )

    e1 = estimate_metric_after_intervention(drift_event, new_store, "js_divergence")
    e2 = estimate_metric_after_intervention(drift_event, new_store, "js_divergence")
    assert e1 == e2


# ── Step 5: Full Counterfactual Analysis ─────────────────────────────────


def test_run_counterfactual_analysis_returns_results():
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)
    results = run_counterfactual_analysis(drift_event, attribution, store)

    assert len(results) == len(build_counterfactual_scenarios(attribution))
    for result in results:
        assert isinstance(result, CounterfactualResult)
        assert result.scenario_id is not None
        assert result.original_metric == drift_event.magnitude
        assert result.counterfactual_metric >= 0.0
        assert result.delta == result.original_metric - result.counterfactual_metric
        assert 0.0 <= result.confidence <= 1.0


def test_run_counterfactual_analysis_shows_improvement():
    """When the change is removed, the counterfactual metric should be lower
    than the original (i.e. delta > 0)."""
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)
    results = run_counterfactual_analysis(drift_event, attribution, store)

    for result in results:
        assert result.delta > 0


def test_run_counterfactual_analysis_deterministic():
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)

    r1 = run_counterfactual_analysis(drift_event, attribution, store)
    r2 = run_counterfactual_analysis(drift_event, attribution, store)

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.original_metric == b.original_metric
        assert a.counterfactual_metric == b.counterfactual_metric
        assert a.delta == b.delta
        assert a.confidence == b.confidence


def test_run_counterfactual_analysis_uses_attribution_metric_name():
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)
    results = run_counterfactual_analysis(drift_event, attribution, store)

    for result in results:
        assert result.metadata["metric_name"] == "js_divergence"


def test_full_pipeline_integration():
    """End-to-end: store → attribution → counterfactual → results."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    # Stable baseline
    for i in range(3):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.02, system_version="1.0.0")
        )

    # Change + drift
    for i in range(3, 7):
        store.save(
            make_drift_record(
                f"r{i}",
                timestamp=i,
                value=0.40 + i * 0.05,
                system_version="2.0.0",
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=6.0,
        magnitude=0.42,
        involved_run_ids=["r3", "r4", "r5", "r6"],
    )

    attribution = attribute_drift(drift_event, store)
    assert len(attribution.factors) > 0

    results = run_counterfactual_analysis(drift_event, attribution, store)
    assert len(results) > 0

    top_result = results[0]
    assert top_result.delta > 0
    assert top_result.confidence > 0.0
    assert top_result.metadata["num_interventions"] >= 1
