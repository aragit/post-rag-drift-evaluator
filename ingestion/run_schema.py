from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RAGRun:
    """Canonical data model representing one RAG evaluation run.

    Encapsulates the query, retrieved documents, embeddings, generated
    answer, and associated metadata so that downstream modules receive
    a consistent, typed object instead of loose dicts.
    """

    query: str

    retrieved_docs: list[str]
    retrieved_doc_ids: list[str] | None = None

    retrieved_embeddings: list[np.ndarray] | None = None
    query_embedding: np.ndarray | None = None

    answer: str | None = None
    answer_embedding: np.ndarray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    timestamp: float | None = None
    system_version: str | None = None

    def __post_init__(self) -> None:
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
            "query",
            "retrieved_docs",
            "retrieved_doc_ids",
            "answer",
            "metadata",
            "timestamp",
            "system_version",
        ):
            if key in data:
                kwargs[key] = data[key]

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
        """

        def _embed_to_list(arr: np.ndarray | None) -> list[float] | None:
            if arr is None:
                return None
            return arr.tolist()

        return {
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
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "system_version": self.system_version,
        }
