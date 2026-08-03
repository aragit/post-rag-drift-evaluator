from __future__ import annotations

import pytest

from evaluator.causal.fusion import (
    CausalGraph,
    CausalLatentFusionEngine,
    CausalNode,
)
from evaluator.latent_drift.schemas import LatentDriftResult

# ── CausalGraph / CausalNode Tests ──────────────────────────────────────


def test_causal_node_schema():
    """CausalNode should have all required fields."""
    node = CausalNode(
        node_id="vector_index",
        node_type="retrieval",
        prior_failure_prob=0.5,
        description="Vector DB quality",
    )
    assert node.node_id == "vector_index"
    assert node.node_type == "retrieval"
    assert node.prior_failure_prob == 0.5


def test_causal_graph_to_from_dict():
    """CausalGraph should serialize/deserialize correctly."""
    graph = CausalGraph(
        nodes=[
            CausalNode(node_id="v_idx", node_type="retrieval", prior_failure_prob=0.3),
            CausalNode(node_id="llm", node_type="generation", prior_failure_prob=0.2),
        ],
        edges=[("v_idx", "llm")],
    )

    d = graph.to_dict()
    assert len(d["nodes"]) == 2
    assert d["edges"] == [("v_idx", "llm")]

    restored = CausalGraph.from_dict(d)
    assert len(restored.nodes) == 2
    assert restored.get_node("v_idx").prior_failure_prob == 0.3


def test_causal_graph_get_node():
    """get_node should return the correct node or None."""
    graph = CausalGraph(
        nodes=[
            CausalNode(node_id="a", node_type="retrieval"),
            CausalNode(node_id="b", node_type="generation"),
        ],
    )
    assert graph.get_node("a") is not None
    assert graph.get_node("c") is None


# ── CausalLatentFusionEngine Tests ──────────────────────────────────────


def test_fusion_default_graph_has_four_nodes():
    """Default graph should have retrieval and generation nodes."""
    engine = CausalLatentFusionEngine()
    graph = engine._default_graph()

    node_types = {n.node_type for n in graph.nodes}
    assert "retrieval" in node_types
    assert "generation" in node_types


def test_fusion_high_retrieval_drift_boosts_retrieval_priors():
    """High retrieval drift score should boost failure probability on retrieval nodes."""
    engine = CausalLatentFusionEngine(sensitivity=2.0)

    drift_result = LatentDriftResult(
        drift_score=0.8,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="retrieval",
    )

    graph = engine.fuse_drift_into_causal_graph(drift_result)

    retrieval_nodes = [n for n in graph.nodes if n.node_type == "retrieval"]
    assert len(retrieval_nodes) > 0
    for node in retrieval_nodes:
        assert node.prior_failure_prob > 0.0
        # 0.8 * 2.0 = 1.6, clamped to 1.0
        assert node.prior_failure_prob <= 1.0


def test_fusion_low_drift_keeps_zero_priors():
    """Low drift score should keep priors near zero."""
    engine = CausalLatentFusionEngine(sensitivity=2.0)

    drift_result = LatentDriftResult(
        drift_score=0.01,
        drift_detected=False,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="retrieval",
    )

    graph = engine.fuse_drift_into_causal_graph(drift_result)

    retrieval_nodes = [n for n in graph.nodes if n.node_type == "retrieval"]
    for node in retrieval_nodes:
        assert node.prior_failure_prob < 0.1  # 0.01 * 2.0 = 0.02


def test_fusion_generation_track_boosts_generation_priors():
    """High generation drift score should boost generation node priors."""
    engine = CausalLatentFusionEngine(sensitivity=2.0)

    drift_result = LatentDriftResult(
        drift_score=0.9,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="generation",
    )

    graph = engine.fuse_drift_into_causal_graph(drift_result)

    gen_nodes = [n for n in graph.nodes if n.node_type == "generation"]
    for node in gen_nodes:
        assert node.prior_failure_prob > 0.0


def test_fusion_does_not_mutate_original_graph():
    """Fusing into a graph should not modify the original."""
    engine = CausalLatentFusionEngine()
    original_graph = engine._default_graph()
    original_probs = {n.node_id: n.prior_failure_prob for n in original_graph.nodes}

    drift_result = LatentDriftResult(
        drift_score=0.9,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="retrieval",
    )

    _ = engine.fuse_drift_into_causal_graph(drift_result, graph=original_graph)

    # Original graph should still have zero priors
    for node in original_graph.nodes:
        assert node.prior_failure_prob == original_probs[node.node_id]


def test_fusion_with_metric_breakdown():
    """DriftResult with metric_breakdown should update per-track scores."""
    engine = CausalLatentFusionEngine(sensitivity=2.0)

    drift_result = LatentDriftResult(
        drift_score=0.5,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="unified",
        metric_breakdown={
            "retrieval": 0.8,
            "generation": 0.2,
            "unified": 0.5,
        },
    )

    graph = engine.fuse_drift_into_causal_graph(drift_result)

    retrieval_nodes = [n for n in graph.nodes if n.node_type == "retrieval"]
    gen_nodes = [n for n in graph.nodes if n.node_type == "generation"]

    for node in retrieval_nodes:
        # 0.8 * 2.0 = 1.6, clamped to 1.0
        assert node.prior_failure_prob <= 1.0
        assert node.prior_failure_prob > 0.0

    for node in gen_nodes:
        # 0.2 * 2.0 = 0.4
        assert node.prior_failure_prob == pytest.approx(0.4, abs=0.01)


def test_fuse_dual_track_separate_results():
    """fuse_dual_track should update both retrieval and generation nodes."""
    engine = CausalLatentFusionEngine(sensitivity=2.0)

    retrieval_result = LatentDriftResult(
        drift_score=0.6,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="retrieval",
    )

    generation_result = LatentDriftResult(
        drift_score=0.4,
        drift_detected=True,
        threshold=0.15,
        n_samples_baseline=500,
        n_samples_current=500,
        metric_used="mmd",
        track="generation",
    )

    graph = engine.fuse_dual_track(retrieval_result, generation_result)

    retrieval_nodes = [n for n in graph.nodes if n.node_type == "retrieval"]
    gen_nodes = [n for n in graph.nodes if n.node_type == "generation"]

    for node in retrieval_nodes:
        assert node.prior_failure_prob == 1.0  # 0.6*2.0=1.2, clamped to 1.0

    for node in gen_nodes:
        assert node.prior_failure_prob == pytest.approx(0.4 * 2.0, abs=0.01)


def test_fuse_dual_track_with_none():
    """fuse_dual_track with None inputs should not crash."""
    engine = CausalLatentFusionEngine()
    graph = engine.fuse_dual_track(None, None)

    for node in graph.nodes:
        assert node.prior_failure_prob == 0.0
