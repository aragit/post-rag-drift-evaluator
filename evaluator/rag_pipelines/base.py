from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)

__all__ = [
    "BaseRAGPipeline",
    "ExecutionMetadataPayload",
    "GraphTopologyPayload",
    "OutputPayload",
    "QueryPayload",
    "RAGEvaluationFrame",
    "RAGResponse",
    "RetrievalContextPayload",
]


class RAGResponse(BaseModel):
    query: str
    retrieved_contexts: list[str]
    generated_answer: str
    query_embedding: list[float]
    reflection_iterations: int = 0
    final_confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseRAGPipeline(ABC):
    @abstractmethod
    async def execute(self, query: str) -> RAGResponse:
        """Executes the full retrieval and generation loop."""
        pass
