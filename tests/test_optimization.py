from __future__ import annotations

import os
import tempfile

from evaluator.causal import (
    CausalAttribution,
    CausalFactor,
    attribute_drift,
)
from evaluator.counterfactual import run_counterfactual_analysis
from evaluator.metrics.results import DriftResult
from evaluator.optimization import (
    OptimizationAction,
    OptimizationPlan,
    OptimizationRecommendation,
    generate_actions,
    generate_optimization_plan,
    score_actions,
)
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
            DriftResult(
                metric_name="js_divergence",
                value=value,
                current_run_id=run_id,
            )
        ],
    )


def _make_drift_with_model_change() -> tuple[JSONHistoryStore, DriftEvent]:
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i in range(3):
        store.save(
            make_drift_record(
                f"r{i}", timestamp=i, value=0.05, system_version="0.1.0"
            )
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


# ── Step 1: Models ───────────────────────────────────────────────────────


def test_optimization_action_auto_id():
    action = OptimizationAction(
        action_type="revert_model",
        target_run_id="run_45",
        change_id="c1",
        description="Revert model change",
    )
    assert action.action_id is not None
    d = action.to_dict()
    assert d["action_type"] == "revert_model"
    assert d["target_run_id"] == "run_45"
    restored = OptimizationAction.from_dict(d)
    assert restored.action_id == action.action_id
    assert restored.change_id == "c1"


def test_optimization_recommendation_serialization():
    action = OptimizationAction(
        action_type="revert_model",
        target_run_id="run_45",
        change_id="c1",
        description="Revert",
    )
    rec = OptimizationRecommendation(
        action=action,
        expected_improvement=0.19,
        confidence=0.87,
        priority=1,
    )
    d = rec.to_dict()
    assert d["expected_improvement"] == 0.19
    assert d["confidence"] == 0.87
    assert d["priority"] == 1
    assert d["recommendation_id"] is not None

    restored = OptimizationRecommendation.from_dict(d)
    assert restored.recommendation_id == rec.recommendation_id
    assert restored.action.action_type == "revert_model"


def test_optimization_plan_serialization():
    action = OptimizationAction(
        action_type="restore_version",
        target_run_id="r1",
        change_id="c1",
        description="Restore",
    )
    rec = OptimizationRecommendation(
        action=action,
        expected_improvement=0.2,
        confidence=0.9,
        priority=1,
    )
    plan = OptimizationPlan(
        drift_event_id="ev1",
        recommendations=[rec],
        summary="test summary",
    )
    d = plan.to_dict()
    assert d["drift_event_id"] == "ev1"
    assert len(d["recommendations"]) == 1
    assert d["summary"] == "test summary"
    assert d["plan_id"] is not None

    restored = OptimizationPlan.from_dict(d)
    assert restored.plan_id == plan.plan_id
    assert len(restored.recommendations) == 1


# ── Step 2: Action Generator ─────────────────────────────────────────────


def test_generate_actions_model_update():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="model_update",
                score=0.9,
                related_run_ids=["run_45"],
                metadata={"change_id": "c1"},
            )
        ],
    )
    actions = generate_actions(attr)
    assert len(actions) == 1
    assert actions[0].action_type == "revert_model"
    assert actions[0].target_run_id == "run_45"
    assert actions[0].change_id == "c1"


def test_generate_actions_config_change():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="config_change",
                score=0.8,
                related_run_ids=["run_2"],
                metadata={"change_id": "c2"},
            )
        ],
    )
    actions = generate_actions(attr)
    assert actions[0].action_type == "rollback_config"


def test_generate_actions_version_change():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="version_change",
                score=0.7,
                related_run_ids=["run_1"],
                metadata={"change_id": "c3"},
            )
        ],
    )
    actions = generate_actions(attr)
    assert actions[0].action_type == "restore_version"


def test_generate_actions_unknown_type_defaults_to_config():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="weird_change",
                score=0.5,
                related_run_ids=["run_9"],
                metadata={"change_id": "c9"},
            )
        ],
    )
    actions = generate_actions(attr)
    assert actions[0].action_type == "rollback_config"


def test_generate_actions_empty_attribution():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[],
    )
    actions = generate_actions(attr)
    assert len(actions) == 0


def test_generate_actions_ranked_by_score():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="config_change",
                score=0.5,
                related_run_ids=["run_low"],
                metadata={"change_id": "c_low"},
            ),
            CausalFactor(
                factor_name="model_update",
                score=0.9,
                related_run_ids=["run_high"],
                metadata={"change_id": "c_high"},
            ),
        ],
    )
    actions = generate_actions(attr)
    assert len(actions) == 2
    assert actions[0].metadata["factor_score"] >= actions[1].metadata["factor_score"]


# ── Step 3: Scoring ────────────────────────────────────────────────────────


def test_score_actions_matches_counterfactual():
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)

    actions = generate_actions(attribution)
    recs = score_actions(actions, counterfx)

    assert len(recs) == len(actions)
    for rec in recs:
        if rec.action.change_id in counterfx[0].metadata.get("change_ids", []):
            assert rec.expected_improvement != 0.0


def test_score_actions_computes_improvement():
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)

    actions = generate_actions(attribution)
    recs = score_actions(actions, counterfx)

    for rec in recs:
        expected = None
        for cf in counterfx:
            if rec.action.change_id in cf.metadata.get("change_ids", []):
                expected = cf.delta
                expected_conf = cf.confidence
                break
        if expected is not None:
            assert rec.expected_improvement == round(expected, 6)
            assert rec.confidence == round(expected_conf, 4)


def test_score_actions_no_counterfactuals():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="model_update",
                score=0.9,
                related_run_ids=["run_1"],
                metadata={"change_id": "c_unknown"},
            )
        ],
    )
    actions = generate_actions(attr)
    recs = score_actions(actions, [])

    assert len(recs) == 1
    assert recs[0].expected_improvement == 0.0
    assert recs[0].confidence == 0.0


def test_score_actions_sorted_by_improvement():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="config_change",
                score=0.5,
                related_run_ids=["r1"],
                metadata={"change_id": "c1"},
            ),
            CausalFactor(
                factor_name="model_update",
                score=0.9,
                related_run_ids=["r2"],
                metadata={"change_id": "c2"},
            ),
        ],
    )
    actions = generate_actions(attr)

    from evaluator.counterfactual.models import CounterfactualResult

    cfs = [
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.9,
            metadata={"change_ids": ["c1"]},
        ),
        CounterfactualResult(
            scenario_id="s2",
            original_metric=0.45,
            counterfactual_metric=0.10,
            delta=0.35,
            confidence=0.8,
            metadata={"change_ids": ["c2"]},
        ),
    ]

    recs = score_actions(actions, cfs)
    assert recs[0].expected_improvement >= recs[1].expected_improvement
    assert recs[0].priority == 1


def test_score_actions_priority_renumbered():
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[
            CausalFactor(
                factor_name="config_change",
                score=0.5,
                related_run_ids=["r1"],
                metadata={"change_id": "c1"},
            ),
            CausalFactor(
                factor_name="model_update",
                score=0.9,
                related_run_ids=["r2"],
                metadata={"change_id": "c2"},
            ),
        ],
    )
    actions = generate_actions(attr)

    from evaluator.counterfactual.models import CounterfactualResult

    cfs = [
        CounterfactualResult(
            scenario_id="s2",
            original_metric=0.45,
            counterfactual_metric=0.10,
            delta=0.35,
            confidence=0.8,
            metadata={"change_ids": ["c2"]},
        ),
        CounterfactualResult(
            scenario_id="s1",
            original_metric=0.45,
            counterfactual_metric=0.05,
            delta=0.40,
            confidence=0.9,
            metadata={"change_ids": ["c1"]},
        ),
    ]

    recs = score_actions(actions, cfs)
    assert recs[0].priority == 1
    assert recs[1].priority == 2
    assert recs[0].expected_improvement >= recs[1].expected_improvement


# ── Step 4: Optimization Engine ──────────────────────────────────────────


def test_generate_optimization_plan_returns_plan():
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)

    plan = generate_optimization_plan(drift_event, attribution, counterfx)

    assert isinstance(plan, OptimizationPlan)
    assert plan.drift_event_id == drift_event.event_id
    assert len(plan.recommendations) > 0
    assert len(plan.summary) > 0


def test_generate_optimization_plan_summary():
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)

    plan = generate_optimization_plan(drift_event, attribution, counterfx)

    assert "recommendation" in plan.summary.lower()
    assert plan.recommendations[0].priority == 1


def test_generate_optimization_plan_empty_recommendations():
    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=1.0,
        end_timestamp=2.0,
        magnitude=0.5,
    )
    attr = CausalAttribution(
        drift_event_id="ev1",
        metric_name="js_divergence",
        factors=[],
    )
    plan = generate_optimization_plan(drift_event, attr, [])

    assert len(plan.recommendations) == 0
    assert "No actionable" in plan.summary


# ── Step 5: Full Integration ─────────────────────────────────────────────


def test_full_pipeline_end_to_end():
    """End-to-end: store → drift → attribution → counterfactual → optimization."""
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    assert len(attribution.factors) > 0

    counterfx = run_counterfactual_analysis(drift_event, attribution, store)
    assert len(counterfx) > 0

    plan = generate_optimization_plan(drift_event, attribution, counterfx)
    assert isinstance(plan, OptimizationPlan)
    assert plan.drift_event_id == drift_event.event_id
    assert len(plan.recommendations) > 0

    top = plan.recommendations[0]
    assert isinstance(top, OptimizationRecommendation)
    assert top.action.action_type in (
        "revert_model",
        "rollback_config",
        "restore_version",
    )
    assert top.expected_improvement > 0.0
    assert 0.0 <= top.confidence <= 1.0
    assert top.priority == 1


def test_pipeline_recommendations_ranked():
    """Recommendations should be sorted by expected_improvement descending."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))

    for i in range(2):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.05, system_version="0.1.0")
        )
    store.save(
        make_drift_record("r2", timestamp=2.0, value=0.30, system_version="0.2.0")
    )
    store.save(
        make_drift_record("r3", timestamp=3.0, value=0.50, system_version="0.2.0")
    )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=2.0,
        end_timestamp=3.0,
        magnitude=0.45,
        involved_run_ids=["r2", "r3"],
    )

    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)
    plan = generate_optimization_plan(drift_event, attribution, counterfx)

    for i in range(len(plan.recommendations) - 1):
        assert (
            plan.recommendations[i].expected_improvement
            >= plan.recommendations[i + 1].expected_improvement
        )
        assert plan.recommendations[i].priority < plan.recommendations[i + 1].priority


def test_plan_serialization_round_trip():
    store, drift_event = _make_drift_with_model_change()
    attribution = attribute_drift(drift_event, store)
    counterfx = run_counterfactual_analysis(drift_event, attribution, store)

    plan = generate_optimization_plan(drift_event, attribution, counterfx)
    d = plan.to_dict()
    restored = OptimizationPlan.from_dict(d)

    assert restored.plan_id == plan.plan_id
    assert restored.drift_event_id == plan.drift_event_id
    assert len(restored.recommendations) == len(plan.recommendations)
    assert restored.summary == plan.summary
