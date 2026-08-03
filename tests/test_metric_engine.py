from __future__ import annotations

import numpy as np

from evaluator.metrics.drift import evaluate_drift
from evaluator.metrics.quality import (
    evaluate_all_from_run,
    evaluate_context_precision,
    evaluate_faithfulness,
)
from evaluator.metrics.results import DriftResult, QualityResult
from evaluator.pipeline import RAGEvaluator
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


# ── RAGEvaluator ──────────────────────────────────────────────────────────


def test_ragevaluator_accepts_ragrun():
    evaluator = RAGEvaluator()
    baseline = make_run(query="baseline", embedding=np.array([0.1, 0.2, 0.3]))
    current = make_run(query="current", embedding=np.array([0.9, 0.8, 0.7]))

    result = evaluator.evaluate(baseline, current)

    assert "drift" in result
    assert "quality" in result
    assert len(result["drift"]) >= 1
    assert len(result["quality"]) >= 1


def test_ragevaluator_returns_structured_results():
    evaluator = RAGEvaluator()
    baseline = make_run(query="baseline", embedding=np.array([0.1, 0.2]))
    current = make_run(query="current", embedding=np.array([0.9, 0.8]))

    result = evaluator.evaluate(baseline, current)

    for item in result["drift"]:
        assert isinstance(item, DriftResult)
        assert item.metric_name
        assert isinstance(item.value, float)

    for item in result["quality"]:
        assert isinstance(item, QualityResult)
        assert item.metric_name
        assert isinstance(item.value, float)


# ── Drift Metrics ─────────────────────────────────────────────────────────


def test_drift_evaluation_returns_drift_result():
    baseline = make_run(
        query="baseline",
        embedding=np.array([0.1, 0.2, 0.3]),
    )
    current = make_run(
        query="current",
        embedding=np.array([0.9, 0.8, 0.7]),
    )

    result = evaluate_drift(baseline, current)

    assert isinstance(result, DriftResult)
    assert result.metric_name == "js_divergence"
    assert result.baseline_run_id == baseline.run_id
    assert result.current_run_id == current.run_id
    assert isinstance(result.value, float)
    assert "is_drifted" in result.metadata
    assert "method" in result.metadata


def test_drift_evaluation_with_identical_embeddings():
    emb = np.array([0.5, 0.5, 0.5])
    baseline = make_run(embedding=emb, query="same")
    current = make_run(embedding=emb, query="same")

    result = evaluate_drift(baseline, current)

    assert isinstance(result, DriftResult)
    assert result.value == 0.0


def test_drift_evaluation_preserves_run_ids():
    baseline = make_run(query="baseline")
    current = make_run(query="current")

    result = evaluate_drift(baseline, current)

    assert result.baseline_run_id == baseline.run_id
    assert result.current_run_id == current.run_id


# ── Quality Metrics ──────────────────────────────────────────────────────


def test_quality_faithfulness_returns_quality_result():
    run = make_run(answer="a1")
    result = evaluate_faithfulness(run)

    assert isinstance(result, QualityResult)
    assert result.metric_name == "faithfulness"
    assert result.run_id == run.run_id
    assert isinstance(result.value, float)


def test_quality_context_precision_returns_quality_result():
    run = make_run()
    result = evaluate_context_precision(run)

    assert isinstance(result, QualityResult)
    assert result.metric_name == "context_precision"
    assert result.run_id == run.run_id
    assert isinstance(result.value, float)


def test_quality_evaluate_all_from_run():
    run = make_run(answer="a1")
    results = evaluate_all_from_run(run)

    assert "faithfulness" in results
    assert "context_precision" in results
    assert isinstance(results["faithfulness"], QualityResult)
    assert isinstance(results["context_precision"], QualityResult)


# ── Backward Compatibility ────────────────────────────────────────────────


def test_backward_compat_evaluate_faithfulness_from_run():
    """The old evaluate_faithfulness_from_run still returns a float."""
    from evaluator.utils.metrics import evaluate_faithfulness_from_run

    run = make_run(answer="a1")
    score = evaluate_faithfulness_from_run(run)
    assert isinstance(score, float)


def test_backward_compat_benchmark_uses_ragrun():
    """Benchmark imports should resolve correctly."""
    from evaluator.benchmark import run_benchmark  # noqa: F401
    from evaluator.utils.metrics import (
        evaluate_context_precision,
        evaluate_faithfulness,
    )

    assert callable(evaluate_faithfulness)
    assert callable(evaluate_context_precision)
