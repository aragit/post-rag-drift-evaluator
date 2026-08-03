from __future__ import annotations

import time

import numpy as np
import pytest

from ingestion.run_schema import RAGRun

# ── Creation & Basic Invariants ──────────────────────────────────────────


def test_ragrun_creation():
    run = RAGRun(
        query="test query",
        retrieved_docs=["doc1", "doc2"],
    )
    run.validate()
    assert run.query == "test query"
    assert run.retrieved_docs == ["doc1", "doc2"]
    assert run.answer is None
    assert run.retrieved_doc_ids is None
    assert run.retrieved_embeddings is None
    assert run.query_embedding is None
    assert run.answer_embedding is None
    assert run.metadata == {}
    assert run.timestamp is not None
    assert run.system_version is None


def test_ragrun_timestamp_auto_set():
    before = time.time()
    run = RAGRun(query="q", retrieved_docs=["d"])
    after = time.time()
    assert before <= run.timestamp <= after


def test_ragrun_timestamp_preserved():
    run = RAGRun(
        query="q",
        retrieved_docs=["d"],
        timestamp=1234567890.0,
    )
    assert run.timestamp == 1234567890.0


def test_ragrun_with_embeddings():
    emb = np.array([0.1, 0.2, 0.3])
    run = RAGRun(
        query="q",
        retrieved_docs=["d1", "d2"],
        retrieved_embeddings=[emb, emb.copy()],
        query_embedding=emb,
        answer="ans",
        answer_embedding=emb,
    )
    run.validate()
    assert run.query_embedding is not None
    assert np.allclose(run.query_embedding, [0.1, 0.2, 0.3])


# ── Validation Errors ─────────────────────────────────────────────────────


def test_validation_empty_query():
    run = RAGRun(query="", retrieved_docs=["d"])
    with pytest.raises(ValueError, match="query"):
        run.validate()


def test_validation_whitespace_query():
    run = RAGRun(query="   ", retrieved_docs=["d"])
    with pytest.raises(ValueError, match="query"):
        run.validate()


def test_validation_empty_docs():
    run = RAGRun(query="q", retrieved_docs=[])
    with pytest.raises(ValueError, match="retrieved_docs"):
        run.validate()


def test_validation_doc_ids_length_mismatch():
    run = RAGRun(
        query="q",
        retrieved_docs=["d1", "d2"],
        retrieved_doc_ids=["id1"],
    )
    with pytest.raises(ValueError, match="doc_ids"):
        run.validate()


def test_validation_embeddings_length_mismatch():
    run = RAGRun(
        query="q",
        retrieved_docs=["d1", "d2"],
        retrieved_embeddings=[np.array([1.0])],
    )
    with pytest.raises(ValueError, match="embeddings"):
        run.validate()


# ── Serialization Round-Trip ─────────────────────────────────────────────


def test_to_dict_basic():
    run = RAGRun(
        query="q",
        retrieved_docs=["d1", "d2"],
        answer="ans",
        metadata={"key": "value"},
    )
    d = run.to_dict()
    assert d["query"] == "q"
    assert d["retrieved_docs"] == ["d1", "d2"]
    assert d["answer"] == "ans"
    assert d["metadata"] == {"key": "value"}
    assert d["retrieved_doc_ids"] is None
    assert d["query_embedding"] is None
    assert d["answer_embedding"] is None


def test_to_dict_with_embeddings():
    emb = np.array([0.5, 0.6])
    run = RAGRun(
        query="q",
        retrieved_docs=["d1"],
        retrieved_embeddings=[emb],
        query_embedding=emb,
        answer_embedding=emb,
    )
    d = run.to_dict()
    assert d["query_embedding"] == [0.5, 0.6]
    assert d["answer_embedding"] == [0.5, 0.6]
    assert d["retrieved_embeddings"] == [[0.5, 0.6]]


def test_from_dict_basic():
    data = {
        "query": "q",
        "retrieved_docs": ["d1", "d2"],
        "answer": "ans",
        "metadata": {"key": "value"},
        "retrieved_doc_ids": ["id1", "id2"],
        "timestamp": 1000.0,
        "system_version": "v1",
    }
    run = RAGRun.from_dict(data)
    assert run.query == "q"
    assert run.retrieved_docs == ["d1", "d2"]
    assert run.answer == "ans"
    assert run.metadata == {"key": "value"}
    assert run.retrieved_doc_ids == ["id1", "id2"]
    assert run.timestamp == 1000.0
    assert run.system_version == "v1"


def test_from_dict_with_embeddings():
    data = {
        "query": "q",
        "retrieved_docs": ["d1", "d2"],
        "retrieved_embeddings": [[0.1, 0.2], [0.3, 0.4]],
        "query_embedding": [0.5, 0.6],
        "answer_embedding": [0.7, 0.8],
    }
    run = RAGRun.from_dict(data)
    assert run.retrieved_embeddings is not None
    assert np.allclose(run.retrieved_embeddings[0], [0.1, 0.2])
    assert np.allclose(run.query_embedding, [0.5, 0.6])
    assert np.allclose(run.answer_embedding, [0.7, 0.8])


def test_round_trip_with_embeddings():
    emb = np.array([1.1, 2.2, 3.3])
    run = RAGRun(
        query="q",
        retrieved_docs=["d1", "d2"],
        retrieved_doc_ids=["id1", "id2"],
        retrieved_embeddings=[emb, emb.copy()],
        query_embedding=emb,
        answer="ans",
        answer_embedding=emb,
        metadata={"extra": 1},
        timestamp=42.0,
        system_version="v2",
    )
    d = run.to_dict()
    restored = RAGRun.from_dict(d)
    assert restored.query == run.query
    assert restored.retrieved_docs == run.retrieved_docs
    assert restored.retrieved_doc_ids == run.retrieved_doc_ids
    assert np.allclose(restored.query_embedding, emb)
    assert np.allclose(restored.answer_embedding, emb)
    assert np.allclose(restored.retrieved_embeddings[0], emb)
    assert np.allclose(restored.retrieved_embeddings[1], emb)
    assert restored.answer == run.answer
    assert restored.metadata == run.metadata
    assert restored.timestamp == run.timestamp
    assert restored.system_version == run.system_version


# ── RAGResponse ↔ RAGRun Integration ─────────────────────────────────────


def test_ragresponse_to_ragrun_round_trip():
    from evaluator.rag_pipelines.base import RAGResponse

    resp = RAGResponse(
        query="What is 2+2?",
        retrieved_contexts=["doc A", "doc B"],
        generated_answer="4",
        query_embedding=[0.1, 0.2],
        reflection_iterations=2,
        final_confidence=0.9,
        metadata={"token_usage": {"total_tokens": 50}},
    )
    run = resp.to_ragrun()
    run.validate()
    assert run.query == resp.query
    assert run.retrieved_docs == resp.retrieved_contexts
    assert run.answer == resp.generated_answer
    assert run.metadata["reflection_iterations"] == 2
    assert run.metadata["final_confidence"] == 0.9
    assert run.metadata["token_usage"] == {"total_tokens": 50}

    # Round-trip back
    resp2 = RAGResponse.from_ragrun(run)
    assert resp2.query == resp.query
    assert resp2.retrieved_contexts == resp.retrieved_contexts
    assert resp2.generated_answer == resp.generated_answer
    assert resp2.query_embedding == resp.query_embedding
    assert resp2.reflection_iterations == 2
    assert resp2.final_confidence == 0.9


def test_ragrun_drifts_monitor():
    from evaluator.drift_monitor import DriftMonitor

    monitor = DriftMonitor()
    baseline = [
        RAGRun(
            query="q1",
            retrieved_docs=["d"],
            query_embedding=np.array([1.0, 2.0, 3.0]),
        ),
        RAGRun(
            query="q2",
            retrieved_docs=["d"],
            query_embedding=np.array([1.5, 2.5, 3.5]),
        ),
    ]
    current = [
        RAGRun(
            query="q3",
            retrieved_docs=["d"],
            query_embedding=np.array([5.0, 6.0, 7.0]),
        ),
        RAGRun(
            query="q4",
            retrieved_docs=["d"],
            query_embedding=np.array([5.5, 6.5, 7.5]),
        ),
    ]
    result = monitor.evaluate_vector_drift_between_runs(baseline, current)
    assert "js_divergence" in result
    assert "mmd_score" in result
    assert "is_drifted" in result
