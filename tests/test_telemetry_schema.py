import pytest
from datetime import datetime

from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)
from evaluator.rag_pipelines.base import RAGResponse


def _naive_frame() -> RAGEvaluationFrame:
    return RAGEvaluationFrame(
        query=QueryPayload(
            text="What is the capital of France?",
            embedding=[0.1, 0.2, 0.3],
        ),
        context=RetrievalContextPayload(
            text_chunks=["Paris is the capital of France."],
            dense_embeddings=[[0.1, 0.2, 0.3]],
        ),
        metadata=ExecutionMetadataPayload(
            rag_type="naive",
            latency_ms=120.5,
        ),
        output=OutputPayload(
            generated_answer="Paris.",
            response_embedding=[0.4, 0.5, 0.6],
            confidence_score=0.92,
        ),
    )


def test_naive_rag_serialization_roundtrip():
    frame = _naive_frame()

    dumped = frame.model_dump_json()
    reloaded = RAGEvaluationFrame.model_validate_json(dumped)

    assert isinstance(dumped, str)
    assert reloaded.query.text == frame.query.text
    assert reloaded.query.embedding == frame.query.embedding
    assert reloaded.context.text_chunks == frame.context.text_chunks
    assert reloaded.context.dense_embeddings == frame.context.dense_embeddings
    assert reloaded.metadata.rag_type == "naive"
    assert reloaded.metadata.latency_ms == 120.5
    assert reloaded.output.generated_answer == "Paris."
    assert reloaded.output.confidence_score == 0.92
    assert reloaded.trace_id == frame.trace_id
    assert reloaded.timestamp == frame.timestamp


def test_graph_rag_payload_serialization():
    frame = RAGEvaluationFrame(
        query=QueryPayload(text="Map the drug interaction network."),
        context=RetrievalContextPayload(
            text_chunks=["Entity A connects to Entity B."],
            graph_topology=GraphTopologyPayload(
                nodes=[
                    {"id": "n1", "label": "Entity A"},
                    {"id": "n2", "label": "Entity B"},
                ],
                edges=[
                    {"source": "n1", "target": "n2", "relation": "inhibits"},
                ],
                density=0.5,
                metadata={"community_count": 1},
            ),
        ),
        metadata=ExecutionMetadataPayload(
            rag_type="graph_rag",
            latency_ms=250.0,
        ),
        output=OutputPayload(generated_answer="A inhibits B."),
    )

    dumped = frame.model_dump_json()
    reloaded = RAGEvaluationFrame.model_validate_json(dumped)

    assert reloaded.context.graph_topology is not None
    assert len(reloaded.context.graph_topology.nodes) == 2
    assert reloaded.context.graph_topology.nodes[0]["label"] == "Entity A"
    assert reloaded.context.graph_topology.edges[0]["relation"] == "inhibits"
    assert reloaded.context.graph_topology.density == 0.5
    assert reloaded.context.graph_topology.metadata["community_count"] == 1
    assert reloaded.metadata.rag_type == "graph_rag"


def test_swarm_payload_serialization_with_agent_hops():
    frame = RAGEvaluationFrame(
        query=QueryPayload(text="Plan the treatment protocol."),
        context=RetrievalContextPayload(
            text_chunks=["Retrieved via planner agent."],
        ),
        metadata=ExecutionMetadataPayload(
            rag_type="swarm",
            agent_hops=["planner", "researcher", "validator"],
            reflection_iterations=3,
            latency_ms=812.3,
            extra={"swarm_size": 5},
        ),
        output=OutputPayload(
            generated_answer="Protocol drafted and validated.",
            confidence_score=0.77,
        ),
    )

    dumped = frame.model_dump_json()
    reloaded = RAGEvaluationFrame.model_validate_json(dumped)

    assert reloaded.metadata.agent_hops == [
        "planner",
        "researcher",
        "validator",
    ]
    assert reloaded.metadata.reflection_iterations == 3
    assert reloaded.metadata.latency_ms == 812.3
    assert reloaded.metadata.extra["swarm_size"] == 5
    assert reloaded.output.confidence_score == 0.77


def test_from_legacy_rag_response():
    legacy = RAGResponse(
        query="test query",
        retrieved_contexts=["ctx1", "ctx2"],
        generated_answer="test answer",
        query_embedding=[0.1, 0.2, 0.3],
        reflection_iterations=2,
        final_confidence=0.85,
        metadata={"pipeline": "agentic", "token_usage": {"total_tokens": 100}},
    )

    frame = RAGEvaluationFrame.from_legacy_rag_response(legacy, rag_type="agentic")

    assert frame.query.text == "test query"
    assert frame.query.embedding == [0.1, 0.2, 0.3]
    assert frame.context.text_chunks == ["ctx1", "ctx2"]
    assert frame.metadata.rag_type == "agentic"
    assert frame.metadata.reflection_iterations == 2
    assert frame.metadata.extra == {
        "pipeline": "agentic",
        "token_usage": {"total_tokens": 100},
    }
    assert frame.output.generated_answer == "test answer"
    assert frame.output.confidence_score == 0.85


def test_from_legacy_rag_response_default_rag_type_and_null_confidence():
    legacy = RAGResponse(
        query="q",
        retrieved_contexts=["c"],
        generated_answer="a",
        query_embedding=[0.0],
    )

    frame = RAGEvaluationFrame.from_legacy_rag_response(legacy)

    assert frame.metadata.rag_type == "agentic"
    assert frame.output.confidence_score is None
    assert frame.trace_id is not None
    assert isinstance(frame.timestamp, datetime)
    assert frame.timestamp.tzinfo is not None


def test_from_legacy_rag_response_rejects_foreign_types():
    with pytest.raises(TypeError):
        RAGEvaluationFrame.from_legacy_rag_response({"not": "a response"})


def test_timestamps_serialize_as_iso8601():
    frame = _naive_frame()
    dumped = frame.model_dump_json()

    assert '"timestamp"' in dumped
    assert "Z" in dumped or "+00:00" in dumped


def test_uuid_trace_id_serializes_as_string():
    frame = _naive_frame()
    dumped = frame.model_dump_json()

    assert '"trace_id":' in dumped
    reloaded = RAGEvaluationFrame.model_validate_json(dumped)
    assert len(reloaded.trace_id) == 36
