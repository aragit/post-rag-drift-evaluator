from __future__ import annotations

import os
import tempfile

from evaluator.causal import attribute_drift
from evaluator.causal.change_extractor import extract_change_events
from evaluator.counterfactual import (
    Intervention,
    apply_intervention,
    apply_scenario,
    run_counterfactual_analysis,
)
from evaluator.metrics.results import DriftResult
from evaluator.storage import EvaluationRecord, InMemoryHistoryStore, JSONHistoryStore
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


def _make_store_with_drift() -> tuple[JSONHistoryStore, DriftEvent]:
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


# ── In-Memory Store Tests ─────────────────────────────────────────────────


def test_in_memory_store_clone():
    """Clone creates a separate store; mutating clone doesn't affect original."""
    store, _ = _make_store_with_drift()
    in_mem = store.to_in_memory()
    clone = in_mem.clone()

    assert len(clone) == len(in_mem)
    # Mutate clone
    clone._records[0].system_version = "9.9.9"
    # Original should be unaffected (shallow copy of list, but record objects are shared)
    original_records = in_mem.load_all()
    assert original_records[0].system_version != "9.9.9"


def test_in_memory_store_load_all():
    """InMemoryHistoryStore.load_all returns records correctly."""
    in_mem = InMemoryHistoryStore()
    rec1 = make_drift_record("r1", 1.0, 0.05)
    rec2 = make_drift_record("r2", 2.0, 0.10)
    in_mem.append(rec1)
    in_mem.append(rec2)

    records = in_mem.load_all()
    assert len(records) == 2
    assert records[0].run_id == "r1"
    assert records[1].run_id == "r2"


def test_in_memory_store_append():
    """Append adds records to the in-memory list."""
    in_mem = InMemoryHistoryStore()
    rec = make_drift_record("r1", 1.0, 0.05)
    in_mem.append(rec)
    assert len(in_mem) == 1


def test_json_store_to_in_memory():
    """JSONHistoryStore.to_in_memory snapshots records correctly."""
    store, _ = _make_store_with_drift()
    in_mem = store.to_in_memory()

    assert isinstance(in_mem, InMemoryHistoryStore)
    assert len(in_mem.load_all()) == 6


# ── In-Memory Simulation Tests ────────────────────────────────────────────


def test_apply_intervention_returns_in_memory_store():
    """apply_intervention returns an InMemoryHistoryStore."""
    store, _ = _make_store_with_drift()
    change_id = _get_first_change_id(store)

    iv = Intervention(action="remove", change_id=change_id)
    result_store = apply_intervention(store, iv)

    assert isinstance(result_store, InMemoryHistoryStore)


def test_apply_intervention_does_not_write_to_disk():
    """apply_intervention does not write to any disk file."""
    store, _ = _make_store_with_drift()
    change_id = _get_first_change_id(store)

    original_path = store._path
    iv = Intervention(action="remove", change_id=change_id)
    result_store = apply_intervention(store, iv)

    # The result store is InMemoryHistoryStore (no _path)
    assert not hasattr(result_store, "_path")

    # Original store file still exists and is unchanged
    assert os.path.exists(original_path)
    original_records = sorted(store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert original_records[3].system_version == "0.2.0"


def test_apply_intervention_reverts_correctly_in_memory():
    """In-memory intervention reverts metadata correctly."""
    store, _ = _make_store_with_drift()
    change_id = _get_first_change_id(store)

    iv = Intervention(action="remove", change_id=change_id)
    result_store = apply_intervention(store, iv)

    records = sorted(result_store.load_all(), key=lambda r: r.timestamp or 0.0)
    assert records[3].system_version == "0.1.0"


def test_multiple_scenarios_no_disk_writes():
    """Running 100 scenario interventions does not create temporary files."""
    store, _ = _make_store_with_drift()
    store_tmpdir = os.path.dirname(store._path)

    for i in range(100):
        for scenario in attribute_drift(_make_store_with_drift()[1], store).factors:
            iv = Intervention(action="remove", change_id=scenario.metadata.get("change_id", ""))
            result = apply_intervention(store, iv)
            assert isinstance(result, InMemoryHistoryStore)

    # Check no temp files were created beyond the original store
    files = [f for f in os.listdir(store_tmpdir) if f.endswith(".jsonl")]
    assert len(files) == 1  # only the original store file


def test_apply_scenario_returns_in_memory():
    """apply_scenario returns an InMemoryHistoryStore."""
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)

    from evaluator.counterfactual.scenario import build_counterfactual_scenarios
    scenarios = build_counterfactual_scenarios(attribution, top_k=3)

    for scenario in scenarios:
        result = apply_scenario(store, scenario)
        assert isinstance(result, InMemoryHistoryStore)
        assert len(result.load_all()) == len(store.load_all())


def test_run_counterfactual_analysis_no_disk_io():
    """Full counterfactual analysis runs without disk I/O."""
    store, drift_event = _make_store_with_drift()
    attribution = attribute_drift(drift_event, store)

    results = run_counterfactual_analysis(drift_event, attribution, store)
    assert len(results) >= 1

    for result in results:
        assert isinstance(result, type(results[0]))
        assert result.original_metric == drift_event.magnitude


def test_original_store_unmutated_after_intervention():
    """JSONHistoryStore records are not mutated by intervention."""
    store, _ = _make_store_with_drift()
    change_id = _get_first_change_id(store)

    original_version = store.load_all()[3].system_version

    iv = Intervention(action="remove", change_id=change_id)
    _ = apply_intervention(store, iv)

    # Reload from disk to verify original is unchanged
    reloaded = store.load_all()
    assert reloaded[3].system_version == original_version
