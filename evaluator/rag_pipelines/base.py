from abc import ABC, abstractmethod
from typing import Dict, Any, List
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
    "RAGResponse",
    "RAGEvaluationFrame",
    "GraphTopologyPayload",
    "QueryPayload",
    "RetrievalContextPayload",
    "ExecutionMetadataPayload",
    "OutputPayload",
]


class RAGResponse(BaseModel):
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    query_embedding: List[float]
    reflection_iterations: int = 0
    final_confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseRAGPipeline(ABC):
    @abstractmethod
    async def execute(self, query: str) -> RAGResponse:
        """Executes the full retrieval and generation loop."""
        pass
