from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from evaluator.metrics.results import DriftResult, QualityResult
from evaluator.pipeline import RAGEvaluator
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from ingestion.run_schema import RAGRun

# ── Fixtures ──────────────────────────────────────────────────────────────


def make_run(
    query: str = "test query",
    docs: list[str] | None = None,
    embedding: np.ndarray | None = None,
    answer: str | None = None,
) -> RAGRun:
    if docs is None:
        docs = ["doc1", "doc2"]
    if embedding is None:
        embedding = np.array([0.1, 0.2, 0.3])
    return RAGRun(
        query=query,
        retrieved_docs=docs,
        query_embedding=embedding,
        answer=answer,
    )


def make_drift_result() -> DriftResult:
    return DriftResult(
        metric_name="js_divergence",
        value=0.5,
        metadata={"is_drifted": True, "method": "jensen_shannon"},
        baseline_run_id="run-1",
        current_run_id="run-2",
    )


def make_quality_result() -> QualityResult:
    return QualityResult(
        metric_name="faithfulness",
        value=0.85,
        run_id="run-2",
    )


# ── EvaluationRecord ──────────────────────────────────────────────────────


def test_evaluation_record_auto_id_and_timestamp():
    record = EvaluationRecord(run_id="run-1")
    assert record.record_id is not None
    assert record.timestamp is not None
    # UUID v4 format
    assert len(record.record_id) == 36
    assert record.metrics == []


def test_evaluation_record_to_dict_from_dict():
    record = EvaluationRecord(
        run_id="run-1",
        metrics=[make_drift_result(), make_quality_result()],
        metadata={"pipeline": "NaiveRAG"},
        system_version="0.1.0",
    )
    d = record.to_dict()
    assert d["run_id"] == "run-1"
    assert d["record_id"] is not None
    assert len(d["metrics"]) == 2
    assert d["metrics"][0]["_type"] == "DriftResult"
    assert d["metrics"][1]["_type"] == "QualityResult"

    restored = EvaluationRecord.from_dict(d)
    assert restored.run_id == record.run_id
    assert restored.record_id == record.record_id
    assert restored.system_version == record.system_version
    assert restored.metadata == record.metadata
    assert isinstance(restored.metrics[0], DriftResult)
    assert isinstance(restored.metrics[1], QualityResult)
    assert restored.metrics[0].baseline_run_id == "run-1"
    assert restored.metrics[1].run_id == "run-2"


def test_evaluation_record_json_serializable():
    record = EvaluationRecord(
        run_id="run-1",
        metrics=[make_drift_result()],
        metadata={"pipeline": "NaiveRAG"},
    )
    d = record.to_dict()
    j = json.dumps(d)
    restored = EvaluationRecord.from_dict(json.loads(j))
    assert restored.run_id == "run-1"


# ── JSONHistoryStore ──────────────────────────────────────────────────────


def test_json_store_save_and_load_all():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)

    # Empty store
    assert store.load_all() == []

    # Save record
    record = EvaluationRecord(
        run_id="run-1",
        metrics=[make_drift_result()],
        metadata={"pipeline": "NaiveRAG"},
    )
    store.save(record)
    assert len(store.load_all()) == 1

    # Save another record
    record2 = EvaluationRecord(
        run_id="run-2",
        metrics=[make_quality_result()],
        metadata={"pipeline": "AgenticRAG"},
    )
    store.save(record2)
    all_records = store.load_all()
    assert len(all_records) == 2
    assert all_records[0].run_id == "run-1"
    assert all_records[1].run_id == "run-2"


def test_json_store_preserves_history():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)

    for i in range(5):
        record = EvaluationRecord(
            run_id=f"run-{i}",
            metrics=[make_drift_result()],
        )
        store.save(record)

    assert len(store.load_all()) == 5


def test_json_store_query_by_run():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)

    store.save(EvaluationRecord(run_id="run-a", metrics=[make_drift_result()]))
    store.save(EvaluationRecord(run_id="run-b", metrics=[make_quality_result()]))
    store.save(EvaluationRecord(run_id="run-a", metrics=[make_quality_result()]))

    results = store.query_by_run("run-a")
    assert len(results) == 2
    for r in results:
        assert r.run_id == "run-a"

    results_b = store.query_by_run("run-b")
    assert len(results_b) == 1


def test_json_store_query_by_metric():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)

    store.save(
        EvaluationRecord(
            run_id="r1",
            metrics=[make_drift_result()],
        )
    )
    store.save(
        EvaluationRecord(
            run_id="r2",
            metrics=[make_quality_result()],
        )
    )
    store.save(
        EvaluationRecord(
            run_id="r3",
            metrics=[make_drift_result(), make_quality_result()],
        )
    )

    drift_records = store.query_by_metric("js_divergence")
    assert len(drift_records) == 2

    quality_records = store.query_by_metric("faithfulness")
    assert len(quality_records) == 2

    none_records = store.query_by_metric("nonexistent")
    assert len(none_records) == 0


def test_json_store_empty_file_returns_empty():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "missing.jsonl")
    store = JSONHistoryStore(path)
    assert store.load_all() == []
    assert store.query_by_run("any") == []
    assert store.query_by_metric("any") == []


# ── RAGEvaluator with History ─────────────────────────────────────────────


def test_ragevaluator_persists_to_history():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    store = JSONHistoryStore(path)
    evaluator = RAGEvaluator(history_store=store)

    baseline = make_run(query="baseline", embedding=np.array([0.1, 0.2, 0.3]))
    current = make_run(query="current", embedding=np.array([0.9, 0.8, 0.7]))

    result = evaluator.evaluate(baseline, current)
    assert "drift" in result
    assert "quality" in result

    records = store.load_all()
    assert len(records) == 1
    assert records[0].run_id == current.run_id
    assert len(records[0].metrics) >= 2  # at least drift + quality


def test_ragevaluator_without_history_store():
    """RAGEvaluator works with no history store (backward compatible)."""
    evaluator = RAGEvaluator()

    baseline = make_run(query="baseline", embedding=np.array([0.1, 0.2, 0.3]))
    current = make_run(query="current", embedding=np.array([0.9, 0.8, 0.7]))

    result = evaluator.evaluate(baseline, current)
    assert "drift" in result
    assert "quality" in result
    assert isinstance(result["drift"][0], DriftResult)
    assert isinstance(result["quality"][0], QualityResult)
