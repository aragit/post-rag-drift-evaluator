from __future__ import annotations

import os
import tempfile

from evaluator.causal import attribute_drift
from evaluator.counterfactual.models import (
    CounterfactualResult,
)
from evaluator.guardrails.policy import PolicyEvaluator
from evaluator.metrics.results import DriftResult
from evaluator.optimization.engine import OptimizationEngine
from evaluator.optimization.runner import (
    STATUS_APPROVED,
    STATUS_BLOCKED_BY_GUARDRAIL,
    STATUS_NO_ACTION_NEEDED,
    OptimizationResult,
    OptimizationRunner,
)
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


def _make_store_with_drift(
    track: str = "unified",
) -> tuple[JSONHistoryStore, DriftEvent]:
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
        metadata={"track": track},
    )
    return store, drift_event


# ── Engine Selection Tests ────────────────────────────────────────────────


def test_engine_selects_retrieval_action_for_retrieval_track():
    """Retrieval track drift should select a retrieval-specific action."""
    engine = OptimizationEngine(min_confidence=0.0)

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )

    cf_results = [
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.90,
            metadata={"change_ids": ["c1"], "run_id": "r3"},
        ),
    ]

    action = engine.select_action(drift_event, cf_results)
    assert action is not None
    assert action.action_type == "adjust_top_k"
    assert action.metadata["track"] == "retrieval"


def test_engine_selects_generation_action_for_generation_track():
    """Generation track drift should select a generation-specific action."""
    engine = OptimizationEngine(min_confidence=0.0)

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
        metadata={"track": "generation"},
    )

    cf_results = [
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.90,
            metadata={"change_ids": ["c1"], "run_id": "r3"},
        ),
    ]

    action = engine.select_action(drift_event, cf_results)
    assert action is not None
    assert action.action_type == "adjust_temperature"
    assert action.metadata["track"] == "generation"


def test_engine_fallback_for_unified_track():
    """Unified track should fall back to default action type."""
    engine = OptimizationEngine(min_confidence=0.0)

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
        metadata={"track": "unified"},
    )

    cf_results = [
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.90,
            metadata={"change_ids": ["c1"], "run_id": "r3"},
        ),
    ]

    action = engine.select_action(drift_event, cf_results)
    assert action is not None
    assert action.action_type == "rollback_config"


def test_engine_rejects_low_confidence():
    """Actions below min_confidence should not be selected."""
    engine = OptimizationEngine(min_confidence=0.70)

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )

    cf_results = [
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.50,
            metadata={"change_ids": ["c1"], "run_id": "r3"},
        ),
    ]

    action = engine.select_action(drift_event, cf_results)
    assert action is None


def test_engine_no_results_returns_none():
    """Empty counterfactual results should return None."""
    engine = OptimizationEngine()
    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=3.0,
        end_timestamp=5.0,
        magnitude=0.45,
    )
    action = engine.select_action(drift_event, [])
    assert action is None


# ── Closed-Loop Runner Tests ──────────────────────────────────────────────


def test_runner_retrieval_track_returns_approved_action():
    """Full cycle: retrieval drift -> approved retrieval action."""
    store, drift_event = _make_store_with_drift(track="retrieval")
    in_memory = store.to_in_memory()
    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert isinstance(result, OptimizationResult)
    assert result.status == STATUS_APPROVED
    assert result.action is not None
    assert result.action.action_type == "adjust_top_k"
    assert result.policy_decision is not None
    assert result.policy_decision.allowed


def test_runner_generation_track_returns_approved_action():
    """Full cycle: generation drift -> approved generation action."""
    store, drift_event = _make_store_with_drift(track="generation")
    in_memory = store.to_in_memory()
    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result.status == STATUS_APPROVED
    assert result.action is not None
    assert result.action.action_type == "adjust_temperature"


def test_runner_blocks_action_violating_parameter_bounds():
    """Policy should block actions with out-of-bounds parameters."""
    store, drift_event = _make_store_with_drift(track="retrieval")
    in_memory = store.to_in_memory()
    attribution = attribute_drift(drift_event, store)

    # Use custom policy with tight bounds and zero cooldown
    policy = PolicyEvaluator(
        cooldown_period_s=0,
        custom_bounds={"temperature": (0.0, 0.5)},  # tight temperature bounds
    )
    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=policy,
    )

    # Patch the engine to produce an action with out-of-bounds params
    original_select = runner.engine.select_action

    def patched_select(drift_event, cf_results, factors=None):
        action = original_select(drift_event, cf_results, factors)
        if action is not None:
            action.metadata["params"] = {"temperature": 0.9}  # violates bounds
        return action

    runner.engine.select_action = patched_select

    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result.status == STATUS_BLOCKED_BY_GUARDRAIL
    assert result.action is not None
    assert result.action.metadata["status"] == STATUS_BLOCKED_BY_GUARDRAIL


def test_runner_no_disk_writes():
    """Optimization cycle should not create temp files for simulation."""
    store, drift_event = _make_store_with_drift(track="retrieval")
    store_tmpdir = os.path.dirname(store._path)
    in_memory = store.to_in_memory()
    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result.status == STATUS_APPROVED

    # Only the original store file should exist
    files = [f for f in os.listdir(store_tmpdir) if f.endswith(".jsonl")]
    assert len(files) == 1


def test_runner_execution_history_preserved():
    """Approved actions are added to execution history."""
    store, drift_event = _make_store_with_drift(track="unified")
    in_memory = store.to_in_memory()
    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    _ = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert len(runner.execution_history) >= 1


def test_runner_empty_attribution_returns_no_action():
    """Empty attribution should produce no action needed."""
    from evaluator.causal.models import CausalAttribution

    store, drift_event = _make_store_with_drift()
    in_memory = store.to_in_memory()

    attribution = CausalAttribution(
        drift_event_id=drift_event.event_id or "",
        metric_name="js_divergence",
        factors=[],
    )

    runner = OptimizationRunner()
    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result.status == STATUS_NO_ACTION_NEEDED
    assert result.action is None
