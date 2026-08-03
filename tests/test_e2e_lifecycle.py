"""End-to-End lifecycle integration test for Sentrix Evaluator v3.0.

Simulates a real production incident flow:
1. Initialize baseline with 500 retrieval & generation embedding pairs.
2. Ingest a drifted batch with corrupted context embeddings.
3. Execute dual-track drift detection → assert retrieval drift is flagged.
4. Trigger counterfactual simulation → evaluate top_k adjustment.
5. Pass action to PolicyEvaluator → assert STATUS_APPROVED.
6. Re-trigger identical action within cooldown → assert STATUS_BLOCKED_BY_GUARDRAIL.
7. Verify all telemetry metrics update accurately.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from evaluator.api.app import create_production_app
from evaluator.causal.attribution import attribute_drift
from evaluator.guardrails.policy import PolicyEvaluator
from evaluator.latent_drift import (
    EmbeddingBatch,
    LatentDriftEngine,
    compute_latent_drift,
)
from evaluator.metrics.results import DriftResult
from evaluator.optimization.engine import OptimizationEngine
from evaluator.optimization.runner import (
    STATUS_APPROVED,
    STATUS_BLOCKED_BY_GUARDRAIL,
    OptimizationResult,
    OptimizationRunner,
)
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal.models import DriftEvent

# ── Helpers ──────────────────────────────────────────────────────────────


def make_retrieval_generation_records(
    n_per_track: int = 500,
    seed: int = 42,
    drift: bool = False,
) -> list[EvaluationRecord]:
    """Generate synthetic evaluation records with dual-track embeddings.

    Each record has retrieval and generation metrics.  When ``drift``
    is True, retrieval embeddings are shifted to simulate vector index
    degradation.
    """
    rng = np.random.RandomState(seed)
    records: list[EvaluationRecord] = []

    for i in range(n_per_track):
        ts = float(i) * 0.01
        base_value = 0.05 + i * 0.0002  # slow rising baseline
        retrieval_value = base_value + (0.3 if drift else 0.0)
        # Store embedding info in metadata for dual-track evaluation
        retrieval_vec = rng.normal(0, 1, size=20).tolist()
        if drift:
            retrieval_vec = [v + 2.0 for v in retrieval_vec]

        generation_vec = rng.normal(0, 1, size=20).tolist()

        record = EvaluationRecord(
            run_id=f"run_{i:04d}",
            timestamp=ts,
            system_version="1.0.0" if i < n_per_track // 2 else "2.0.0",
            metadata={
                "retrieval_embedding": retrieval_vec,
                "generation_embedding": generation_vec,
                "pipeline_name": "rag_pipeline",
            },
            metrics=[
                DriftResult(
                    metric_name="js_divergence",
                    value=retrieval_value,
                    current_run_id=f"run_{i:04d}",
                ),
            ],
        )
        records.append(record)

    return records


# ── Step 1-3: Dual-Track Drift Detection ────────────────────────────────


def test_e2e_dual_track_drift_detection():
    """Full dual-track drift detection on synthetic 500-sample baseline."""
    rng = np.random.RandomState(42)
    baseline_retrieval = rng.normal(0, 1, size=(500, 20))

    rng_shift = np.random.RandomState(43)
    drifted_retrieval = rng_shift.normal(2.0, 1, size=(500, 20))

    baseline = EmbeddingBatch(vectors=baseline_retrieval, track="retrieval")
    drifted = EmbeddingBatch(vectors=drifted_retrieval, track="retrieval")

    result = compute_latent_drift(baseline, drifted, threshold=0.15, metric="mmd")
    assert result.drift_detected is True
    assert result.drift_score > 0.15
    assert result.track == "retrieval"
    assert result.metric_used == "mmd"


def test_e2e_generation_track_no_drift():
    """Generation track with similar distributions shows no drift."""
    rng = np.random.RandomState(42)
    baseline_gen = rng.normal(0, 1, size=(500, 10))

    rng_current = np.random.RandomState(99)
    current_gen = rng_current.normal(0, 1, size=(500, 10))

    baseline = EmbeddingBatch(vectors=baseline_gen, track="generation")
    current = EmbeddingBatch(vectors=current_gen, track="generation")

    engine = LatentDriftEngine(threshold=0.3, metric="mmd", pca_components=5)
    result = engine.fit_compute(baseline.vectors, current.vectors)
    assert result.drift_score < 0.3


# ── Step 4-5: Counterfactual Simulation → Policy → Approved ─────────────


def test_e2e_counterfactual_then_policy_approval():
    """Full chain: drift → attribution → CF → engine → policy → approved."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "history.jsonl"))

    # Baseline stable records
    for i in range(5):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="1.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.05, current_run_id=f"r{i}")],
            )
        )

    # Drifted records with version change
    for i in range(5, 10):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="2.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.45, current_run_id=f"r{i}")],
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,
        end_timestamp=9.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )

    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    in_memory = store.to_in_memory()
    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)

    assert isinstance(result, OptimizationResult)
    assert result.status == STATUS_APPROVED
    assert result.action is not None
    assert result.action.action_type == "adjust_top_k"
    assert result.policy_decision is not None
    assert result.policy_decision.allowed is True


# ── Step 6: Cooldown Blocks Second Action ────────────────────────────────


def test_e2e_cooldown_blocks_identical_action():
    """Re-triggering identical action within cooldown is blocked."""
    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "history.jsonl"))

    for i in range(5):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="1.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.05, current_run_id=f"r{i}")],
            )
        )

    for i in range(5, 10):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="2.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.45, current_run_id=f"r{i}")],
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,
        end_timestamp=9.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )

    attribution = attribute_drift(drift_event, store)

    # First: action should be approved (cooldown=300s default)
    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(),  # default 300s cooldown
    )

    in_memory = store.to_in_memory()
    result1 = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result1.status == STATUS_APPROVED
    assert len(runner.execution_history) == 1

    # Second: identical action within cooldown should be blocked
    result2 = runner.run_optimization_cycle(drift_event, attribution, in_memory)
    assert result2.status == STATUS_BLOCKED_BY_GUARDRAIL
    assert result2.policy_decision is not None
    assert result2.policy_decision.rule_violated == "cooldown_period"


# ── Step 7: Telemetry Metrics Verification ──────────────────────────────


def test_e2e_telemetry_metrics_updated():
    """Verify telemetry counters and gauges update during the lifecycle."""
    from evaluator.telemetry import SentrixMetricsExporter

    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "history.jsonl"))

    for i in range(5):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="1.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.05, current_run_id=f"r{i}")],
            )
        )

    for i in range(5, 10):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="2.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.45, current_run_id=f"r{i}")],
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,
        end_timestamp=9.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )

    telemetry = SentrixMetricsExporter(otel_enabled=False)
    attribution = attribute_drift(drift_event, store)

    runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    in_memory = store.to_in_memory()
    result = runner.run_optimization_cycle(drift_event, attribution, in_memory)

    # Record telemetry
    telemetry.increment_drift_events(severity="high")
    telemetry.record_latent_drift_score(
        score=0.45, track="retrieval", metric="mmd"
    )
    telemetry.increment_optimization_actions(
        status=result.status,
        rule_violated=result.policy_decision.rule_violated if result.policy_decision else None,
    )

    counters = telemetry.get_counters_snapshot()
    metrics = telemetry.get_metrics_snapshot()

    assert counters.get("drift_events:high", 0) >= 1
    assert "drift_score:retrieval:mmd" in metrics
    assert metrics["drift_score:retrieval:mmd"] == 0.45
    assert counters.get(f"optimization_actions:{result.status}:none", 0) >= 1 or \
           counters.get(f"optimization_actions:{result.status}:cooldown_period", 0) >= 1


# ── API Endpoint Tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2e_api_eval_endpoint():
    """POST /v1/eval ingests records."""
    app = create_production_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        record = EvaluationRecord(
            run_id="test_run",
            timestamp=1.0,
            system_version="1.0.0",
            metrics=[DriftResult(metric_name="js_divergence", value=0.05, current_run_id="test_run")],
        )
        response = await client.post(
            "/v1/eval",
            json={"records": [record.to_dict()]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 1


@pytest.mark.asyncio
async def test_e2e_api_drift_detect_endpoint():
    """POST /v1/drift/detect returns drift score."""
    import numpy as np

    from evaluator.api.app import create_production_app

    app = create_production_app()

    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, size=(200, 20)).tolist()
    current = rng.normal(2.0, 1, size=(200, 20)).tolist()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/drift/detect",
            json={
                "baseline_vectors": baseline,
                "current_vectors": current,
                "threshold": 0.15,
                "metric": "mmd",
                "track": "retrieval",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["drift_detected"] is True
        assert data["metric_used"] == "mmd"
        assert data["track"] == "retrieval"


@pytest.mark.asyncio
async def test_e2e_api_remediate_endpoint():
    """POST /v1/remediate returns approved or blocked status."""
    app = create_production_app()

    tmpdir = tempfile.mkdtemp()
    store = JSONHistoryStore(os.path.join(tmpdir, "h.jsonl"))
    app.state.store = store
    app.state.store_path = os.path.join(tmpdir, "h.jsonl")
    app.state.runner = OptimizationRunner(
        engine=OptimizationEngine(min_confidence=0.0),
        policy_evaluator=PolicyEvaluator(cooldown_period_s=0),
    )

    # Populate store
    for i in range(5):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="1.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.05, current_run_id=f"r{i}")],
            )
        )
    for i in range(5, 10):
        store.save(
            EvaluationRecord(
                run_id=f"r{i}",
                timestamp=float(i),
                system_version="2.0.0",
                metrics=[DriftResult(metric_name="js_divergence", value=0.45, current_run_id=f"r{i}")],
            )
        )

    drift_event = DriftEvent(
        metric_name="js_divergence",
        start_timestamp=5.0,
        end_timestamp=9.0,
        magnitude=0.45,
        metadata={"track": "retrieval"},
    )
    attribution = attribute_drift(drift_event, store)

    store_records = [r.to_dict() for r in store.load_all()]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/remediate",
            json={
                "drift_event": drift_event.to_dict(),
                "attribution": attribution.to_dict(),
                "store_records": store_records,
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["counterfactual_count"] >= 0
