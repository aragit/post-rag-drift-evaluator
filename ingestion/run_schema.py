from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from evaluator.schemas.telemetry import RAGEvaluationFrame

SCHEMA_VERSION = "1.0"

_RAG_TYPE_MAP: dict[str, str] = {
    "naive": "naive",
    "agentic": "agentic",
    "graph": "graph_rag",
    "swarm": "swarm",
}


@dataclass
class RAGSystemInfo:
    """Identity metadata for the evaluated RAG pipeline.

    Structurally captures *which* system produced a run so that
    cross-pipeline comparisons (e.g. NaiveRAG vs AgenticRAG with
    different embedding models or retrievers) are possible at
    query time rather than via ad-hoc metadata lookup.
    """

    name: str
    model: str | None = None
    embedding_model: str | None = None
    retriever: str | None = None
    version: str | None = None


@dataclass
class RAGRun:
    """Canonical data model representing one RAG evaluation run.

    Encapsulates the query, retrieved documents, embeddings, generated
    answer, and associated metadata so that downstream modules receive
    a consistent, typed object instead of loose dicts.
    """

    schema_version: str = SCHEMA_VERSION

    run_id: str | None = None

    query: str = ""

    retrieved_docs: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] | None = None

    retrieved_embeddings: list[np.ndarray] | None = None
    query_embedding: np.ndarray | None = None

    answer: str | None = None
    answer_embedding: np.ndarray | None = None

    system_info: RAGSystemInfo | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    timestamp: float | None = None
    system_version: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()

    def validate(self) -> None:
        """Validate the run's invariants.

        Raises:
            ValueError: If any invariant is violated.
        """
        if not self.query or not self.query.strip():
            raise ValueError("query must not be empty")

        if not self.retrieved_docs:
            raise ValueError("retrieved_docs must not be empty")

        if self.retrieved_doc_ids is not None:
            if len(self.retrieved_doc_ids) != len(self.retrieved_docs):
                raise ValueError(
                    "retrieved_doc_ids length must match retrieved_docs length"
                )

        if self.retrieved_embeddings is not None:
            if len(self.retrieved_embeddings) != len(self.retrieved_docs):
                raise ValueError(
                    "retrieved_embeddings length must match retrieved_docs length"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGRun:
        """Construct a :class:`RAGRun` from a plain dictionary.

        Numpy arrays provided as lists are converted back to
        ``np.ndarray`` instances.
        """
        kwargs: dict[str, Any] = {}

        for key in (
            "schema_version",
            "run_id",
            "query",
            "retrieved_docs",
            "retrieved_doc_ids",
            "answer",
            "system_info",
            "metadata",
            "timestamp",
            "system_version",
        ):
            if key in data:
                value = data[key]
                if key == "system_info" and value is not None:
                    kwargs[key] = RAGSystemInfo(**value)
                else:
                    kwargs[key] = value

        if "retrieved_embeddings" in data and data["retrieved_embeddings"] is not None:
            kwargs["retrieved_embeddings"] = [
                np.asarray(e) for e in data["retrieved_embeddings"]
            ]

        if "query_embedding" in data and data["query_embedding"] is not None:
            kwargs["query_embedding"] = np.asarray(data["query_embedding"])

        if "answer_embedding" in data and data["answer_embedding"] is not None:
            kwargs["answer_embedding"] = np.asarray(data["answer_embedding"])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run to a plain dictionary.

        Numpy arrays are converted to lists for JSON compatibility.
        ``RAGSystemInfo`` is serialized to a nested dict.
        """

        def _embed_to_list(arr: np.ndarray | None) -> list[float] | None:
            if arr is None:
                return None
            return arr.tolist()

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "query": self.query,
            "retrieved_docs": self.retrieved_docs,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "retrieved_embeddings": (
                [_embed_to_list(e) for e in self.retrieved_embeddings]
                if self.retrieved_embeddings is not None
                else None
            ),
            "query_embedding": _embed_to_list(self.query_embedding),
            "answer": self.answer,
            "answer_embedding": _embed_to_list(self.answer_embedding),
            "system_info": (
                asdict(self.system_info) if self.system_info is not None else None
            ),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "system_version": self.system_version,
        }

    def to_evaluation_frame(
        self, scores: dict[str, float] | None = None
    ) -> RAGEvaluationFrame:
        """Bridge this canonical :class:`RAGRun` into a unified telemetry frame.

        All ``np.ndarray`` attributes (``query_embedding``,
        ``retrieved_embeddings``, ``answer_embedding``) are coerced to native
        ``list[float]`` so the resulting frame is JSON / Postgres JSONB /
        Redis Streams serializable without NumPy type errors.

        Computed ``scores`` (e.g. faithfulness, context precision) are folded
        into the execution metadata's ``extra`` bag for downstream drift
        monitoring and dashboarding.
        """
        from datetime import datetime, timezone

        from evaluator.schemas.telemetry import (
            ExecutionMetadataPayload,
            OutputPayload,
            QueryPayload,
            RAGEvaluationFrame,
            RetrievalContextPayload,
        )

        scores = scores or {}

        name = ""
        if self.system_info is not None:
            name = self.system_info.name or ""
        rag_type = _RAG_TYPE_MAP.get(name.lower().removesuffix("rag"), "custom")

        timestamp = self.timestamp
        frame_ts = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if timestamp is not None
            else datetime.now(timezone.utc)
        )

        return RAGEvaluationFrame(
            trace_id=self.run_id or str(uuid.uuid4()),
            timestamp=frame_ts,
            query=QueryPayload(
                text=self.query,
                embedding=(
                    self.query_embedding.tolist()
                    if self.query_embedding is not None
                    else None
                ),
            ),
            context=RetrievalContextPayload(
                text_chunks=list(self.retrieved_docs),
                dense_embeddings=(
                    [e.tolist() for e in self.retrieved_embeddings]
                    if self.retrieved_embeddings is not None
                    else None
                ),
            ),
            metadata=ExecutionMetadataPayload(
                rag_type=rag_type,
                reflection_iterations=int(
                    self.metadata.get("reflection_iterations", 0) or 0
                ),
                agent_hops=self.metadata.get("agent_hops"),
                latency_ms=self.metadata.get("latency_ms"),
                extra={
                    **scores,
                    "token_usage": self.metadata.get("token_usage", {}),
                },
            ),
            output=OutputPayload(
                generated_answer=self.answer or "",
                response_embedding=(
                    self.answer_embedding.tolist()
                    if self.answer_embedding is not None
                    else None
                ),
                confidence_score=(
                    float(self.metadata["final_confidence"])
                    if self.metadata.get("final_confidence")
                    else None
                ),
            ),
        )
