from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)

if TYPE_CHECKING:
    from ingestion.run_schema import RAGRun

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

    def to_ragrun(self) -> RAGRun:
        """Convert this legacy ``RAGResponse`` into the canonical ``RAGRun``.

        ``query_embedding`` is converted from ``list[float]`` to
        ``np.ndarray`` and pipeline metadata is preserved in
        ``RAGRun.metadata``.
        """
        from ingestion.run_schema import RAGRun

        metadata = dict(self.metadata)
        metadata["reflection_iterations"] = self.reflection_iterations
        metadata["final_confidence"] = self.final_confidence

        return RAGRun(
            query=self.query,
            retrieved_docs=list(self.retrieved_contexts),
            query_embedding=(
                np.asarray(self.query_embedding, dtype=float)
                if self.query_embedding
                else None
            ),
            answer=self.generated_answer,
            metadata=metadata,
        )

    @classmethod
    def from_ragrun(cls, run: RAGRun) -> RAGResponse:
        """Construct a legacy ``RAGResponse`` from a canonical ``RAGRun``.

        Extra fields (``retrieved_doc_ids``, ``answer_embedding``,
        ``timestamp``, ``system_version``) that have no counterpart in
        ``RAGResponse`` are dropped or folded into ``metadata``.
        """
        metadata = dict(run.metadata)
        metadata.setdefault("reflection_iterations", 0)
        metadata.setdefault("final_confidence", 0.0)

        return cls(
            query=run.query,
            retrieved_contexts=list(run.retrieved_docs),
            generated_answer=run.answer or "",
            query_embedding=(
                run.query_embedding.tolist() if run.query_embedding is not None else []
            ),
            reflection_iterations=metadata.pop("reflection_iterations", 0),
            final_confidence=metadata.pop("final_confidence", 0.0),
            metadata=metadata,
        )


class BaseRAGPipeline(ABC):
    @abstractmethod
    async def execute(self, query: str) -> RAGResponse:
        """Executes the full retrieval and generation loop."""
        pass
