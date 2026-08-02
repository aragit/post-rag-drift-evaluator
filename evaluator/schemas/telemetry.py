from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from evaluator.rag_pipelines.base import RAGResponse


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GraphTopologyPayload(BaseModel):
    """Sub-graph topology emitted by GraphRAG systems."""

    model_config = ConfigDict(extra="allow")

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    density: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryPayload(BaseModel):
    """The user query as received by the evaluated RAG system."""

    model_config = ConfigDict(extra="allow")

    text: str
    embedding: list[float] | None = None


class RetrievalContextPayload(BaseModel):
    """Everything the RAG system retrieved/generated before synthesis."""

    model_config = ConfigDict(extra="allow")

    text_chunks: list[str]
    dense_embeddings: list[list[float]] | None = None
    graph_topology: GraphTopologyPayload | None = None


class ExecutionMetadataPayload(BaseModel):
    """Runtime characteristics of the RAG execution trace."""

    model_config = ConfigDict(extra="allow")

    rag_type: Literal["naive", "agentic", "graph_rag", "swarm", "custom"]
    agent_hops: list[str] | None = None
    reflection_iterations: int = 0
    latency_ms: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class OutputPayload(BaseModel):
    """The final synthesized answer and its properties."""

    model_config = ConfigDict(extra="allow")

    generated_answer: str
    response_embedding: list[float] | None = None
    confidence_score: float | None = None


class RAGEvaluationFrame(BaseModel):
    """Framework-agnostic telemetry contract for any RAG system.

    Supports dense-vector RAGs, GraphRAGs (sub-graph topologies), and
    multi-agent swarms (routing hops, reflection iterations).
    """

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=_utcnow)
    query: QueryPayload
    context: RetrievalContextPayload
    metadata: ExecutionMetadataPayload
    output: OutputPayload

    @classmethod
    def from_legacy_rag_response(
        cls, rag_response: RAGResponse, rag_type: str = "agentic"
    ) -> RAGEvaluationFrame:
        """Adapt a legacy ``RAGResponse`` into the unified telemetry frame.

        Keeps backward compatibility with the existing Naive/Agentic
        pipelines until they are migrated onto ``RAGEvaluationFrame``.
        """
        from evaluator.rag_pipelines.base import RAGResponse

        if not isinstance(rag_response, RAGResponse):
            raise TypeError(
                f"Expected a RAGResponse instance, got {type(rag_response).__name__}."
            )

        return cls(
            query=QueryPayload(
                text=rag_response.query,
                embedding=rag_response.query_embedding,
            ),
            context=RetrievalContextPayload(
                text_chunks=rag_response.retrieved_contexts,
            ),
            metadata=ExecutionMetadataPayload(
                rag_type=rag_type,
                reflection_iterations=rag_response.reflection_iterations,
                extra=dict(rag_response.metadata),
            ),
            output=OutputPayload(
                generated_answer=rag_response.generated_answer,
                confidence_score=(
                    rag_response.final_confidence
                    if rag_response.final_confidence
                    else None
                ),
            ),
        )
