from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict as _asdict
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
        from ingestion.run_schema import RAGRun, RAGSystemInfo

        metadata = dict(self.metadata)
        metadata["reflection_iterations"] = self.reflection_iterations
        metadata["final_confidence"] = self.final_confidence

        system_info = None
        if "pipeline_name" in self.metadata:
            system_info = RAGSystemInfo(
                name=self.metadata.get("pipeline_name") or "",
                model=self.metadata.get("model") or "",
                embedding_model=self.metadata.get("embedding_model") or "default",
                retriever=self.metadata.get("retriever") or "default",
                version=self.metadata.get("version") or "0.0.0",
            )

        return RAGRun(
            query=self.query,
            retrieved_docs=list(self.retrieved_contexts),
            query_embedding=(
                np.asarray(self.query_embedding, dtype=float)
                if self.query_embedding
                else None
            ),
            answer=self.generated_answer,
            system_info=system_info,
            metadata=metadata,
        )

    @classmethod
    def from_ragrun(cls, run: RAGRun) -> RAGResponse:
        """Construct a legacy ``RAGResponse`` from a canonical ``RAGRun``.

        ``RAGSystemInfo`` and RAGRun-only fields (``run_id``,
        ``schema_version``, ``timestamp``) are folded into
        ``metadata`` since ``RAGResponse`` has no dedicated slots.
        """
        metadata = dict(run.metadata)
        metadata.setdefault("reflection_iterations", 0)
        metadata.setdefault("final_confidence", 0.0)

        if run.run_id is not None:
            metadata["run_id"] = run.run_id
        if run.schema_version is not None:
            metadata["schema_version"] = run.schema_version
        if run.timestamp is not None:
            metadata["timestamp"] = run.timestamp
        if run.system_info is not None:
            metadata["system_info"] = _asdict(run.system_info)

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
