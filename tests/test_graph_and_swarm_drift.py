from typing import Any
from unittest.mock import AsyncMock

from evaluator.drift import GraphDriftCalculator, SwarmDriftCalculator
from evaluator.drift_monitor import DriftMonitor
from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)

SPARSE_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 6)],
    "edges": [{"source": "1", "target": "2"}],
}

DENSE_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 6)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "5"},
        {"source": "5", "target": "1"},
        {"source": "1", "target": "3"},
        {"source": "2", "target": "4"},
        {"source": "3", "target": "5"},
        {"source": "4", "target": "1"},
        {"source": "5", "target": "2"},
    ],
}

CYCLE_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "1"},
    ],
}

PATH_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
    ],
}


def _frame(
    *,
    embedding: list[float] | None = None,
    rag_type: str = "naive",
    graph: dict[str, Any] | None = None,
    agent_hops: list[str] | None = None,
    reflection_iterations: int = 0,
) -> RAGEvaluationFrame:
    return RAGEvaluationFrame(
        query=QueryPayload(text="test query", embedding=embedding),
        context=RetrievalContextPayload(
            text_chunks=["chunk"],
            graph_topology=GraphTopologyPayload(**graph) if graph else None,
        ),
        metadata=ExecutionMetadataPayload(
            rag_type=rag_type,
            agent_hops=agent_hops,
            reflection_iterations=reflection_iterations,
        ),
        output=OutputPayload(generated_answer="answer"),
    )


def test_graph_drift_detects_density_shift():
    calculator = GraphDriftCalculator()

    result = calculator.compute_graph_drift([SPARSE_GRAPH], [DENSE_GRAPH])

    assert result["is_graph_drifted"] is True
    assert result["density_delta"] > 0.1
    assert result["node_count_delta"] == 0


def test_graph_drift_detects_spectral_shift():
    calculator = GraphDriftCalculator()

    result = calculator.compute_graph_drift([CYCLE_GRAPH], [PATH_GRAPH])

    assert result["is_graph_drifted"] is True
    assert result["spectral_distance"] > 0.5
    assert abs(result["density_delta"]) < 0.1


def test_graph_drift_identical_graphs_are_stable():
    calculator = GraphDriftCalculator()

    result = calculator.compute_graph_drift([CYCLE_GRAPH], [CYCLE_GRAPH])

    assert result["is_graph_drifted"] is False
    assert result["spectral_distance"] == 0.0
    assert result["density_delta"] == 0.0
    assert result["node_count_delta"] == 0


def test_graph_drift_accepts_topology_payloads():
    calculator = GraphDriftCalculator()
    baseline = [
        GraphTopologyPayload(**CYCLE_GRAPH),
        GraphTopologyPayload(**CYCLE_GRAPH),
    ]
    current = [
        GraphTopologyPayload(**PATH_GRAPH),
        GraphTopologyPayload(**PATH_GRAPH),
    ]

    result = calculator.compute_graph_drift(baseline, current)

    assert result["is_graph_drifted"] is True


def test_graph_drift_empty_groups_are_stable():
    calculator = GraphDriftCalculator()

    result = calculator.compute_graph_drift([], [])

    assert result["is_graph_drifted"] is False
    assert result["spectral_distance"] == 0.0
    assert result["density_delta"] == 0.0


def test_graph_drift_single_node_graphs_are_stable():
    calculator = GraphDriftCalculator()
    single = {"nodes": [{"id": "1"}], "edges": []}

    result = calculator.compute_graph_drift([single], [single])

    assert result["is_graph_drifted"] is False
    assert result["density_delta"] == 0.0


def test_swarm_drift_detects_transition_entropy_shift():
    calculator = SwarmDriftCalculator()
    baseline = [
        {"agent_hops": ["a", "b", "a", "b"], "reflection_iterations": 1},
        {"agent_hops": ["a", "b", "a", "b"], "reflection_iterations": 1},
    ]
    current = [
        {"agent_hops": ["a", "b", "c", "a", "c", "b"], "reflection_iterations": 1},
    ]

    result = calculator.compute_swarm_drift(baseline, current)

    assert result["is_swarm_drifted"] is True
    assert result["transition_entropy_delta"] > 0.5


def test_swarm_drift_detects_reflection_iteration_shift():
    calculator = SwarmDriftCalculator()
    baseline = [
        {"agent_hops": ["a", "b"], "reflection_iterations": 1},
        {"agent_hops": ["a", "b"], "reflection_iterations": 1},
        {"agent_hops": ["a", "b"], "reflection_iterations": 1},
    ]
    current = [
        {"agent_hops": ["a", "b"], "reflection_iterations": 5},
        {"agent_hops": ["a", "b"], "reflection_iterations": 5},
        {"agent_hops": ["a", "b"], "reflection_iterations": 5},
    ]

    result = calculator.compute_swarm_drift(baseline, current)

    assert result["is_swarm_drifted"] is True
    assert result["avg_reflection_iterations_delta"] == 4.0


def test_swarm_drift_identical_sequences_are_stable():
    calculator = SwarmDriftCalculator()
    baseline = [
        {"agent_hops": ["a", "b", "c"], "reflection_iterations": 2},
        {"agent_hops": ["a", "b", "c"], "reflection_iterations": 2},
    ]
    current = [
        {"agent_hops": ["a", "b", "c"], "reflection_iterations": 2},
        {"agent_hops": ["a", "b", "c"], "reflection_iterations": 2},
    ]

    result = calculator.compute_swarm_drift(baseline, current)

    assert result["is_swarm_drifted"] is False
    assert result["transition_entropy_delta"] == 0.0
    assert result["avg_reflection_iterations_delta"] == 0.0


def test_swarm_drift_empty_sequences_are_stable():
    calculator = SwarmDriftCalculator()

    result = calculator.compute_swarm_drift([], [])

    assert result["is_swarm_drifted"] is False
    assert result["transition_entropy_delta"] == 0.0


async def test_evaluate_frames_combines_vector_graph_and_swarm():
    monitor = DriftMonitor(store=AsyncMock())

    baseline = [
        _frame(embedding=[0.1, 0.2], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
        _frame(embedding=[0.11, 0.21], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
    ]
    current = [
        _frame(
            embedding=[0.1, 0.2],
            graph=CYCLE_GRAPH,
            agent_hops=["a", "b", "c", "a"],
        ),
        _frame(
            embedding=[0.11, 0.21],
            graph=CYCLE_GRAPH,
            agent_hops=["a", "c", "b", "a"],
        ),
    ]

    result = await monitor.evaluate_frames(baseline, current)

    assert set(result) == {"vector_drift", "graph_drift", "swarm_drift", "is_drifted"}
    assert result["graph_drift"]["is_graph_drifted"] is True
    assert result["swarm_drift"]["is_swarm_drifted"] is True
    assert result["is_drifted"] is True


async def test_evaluate_frames_identical_frames_are_stable():
    monitor = DriftMonitor(store=AsyncMock())
    frames = [
        _frame(embedding=[0.1, 0.2], graph=CYCLE_GRAPH, agent_hops=["a", "b"]),
        _frame(embedding=[0.11, 0.21], graph=CYCLE_GRAPH, agent_hops=["a", "b"]),
    ]

    result = await monitor.evaluate_frames(frames, frames)

    assert result["is_drifted"] is False


async def test_evaluate_frames_missing_embeddings_does_not_crash():
    monitor = DriftMonitor(store=AsyncMock())
    baseline = [_frame(graph=PATH_GRAPH, agent_hops=["a", "b"])]
    current = [_frame(graph=CYCLE_GRAPH, agent_hops=["a", "c", "b"])]

    result = await monitor.evaluate_frames(baseline, current)

    assert result["vector_drift"]["js_divergence"] == 0.0
    assert result["graph_drift"]["is_graph_drifted"] is True
    assert result["is_drifted"] is True


async def test_evaluate_frames_persists_each_current_frame():
    store = AsyncMock()
    monitor = DriftMonitor(store=store)
    baseline = [_frame(embedding=[0.1, 0.2])]
    current = [
        _frame(embedding=[0.1, 0.2]),
        _frame(embedding=[0.1, 0.2]),
    ]

    await monitor.evaluate_frames(baseline, current)

    assert store.record_evaluation.await_count == 2
    metrics = store.record_evaluation.await_args.args[1]
    assert metrics["is_drifted"] is False
    assert "spectral_distance" in metrics
    assert "transition_entropy_delta" in metrics


async def test_evaluate_frames_skips_persistence_when_current_empty():
    store = AsyncMock()
    monitor = DriftMonitor(store=store)
    baseline = [_frame(embedding=[0.1, 0.2])]

    result = await monitor.evaluate_frames(baseline, [])

    assert result["is_drifted"] is False
    store.record_evaluation.assert_not_awaited()
