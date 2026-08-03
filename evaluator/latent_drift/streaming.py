"""Streaming drift buffer for continuous embedding evaluation.

Provides a memory-safe ring buffer that ingests embedding vectors
continuously and exports snapshots as :class:`EmbeddingBatch` objects
for dual-track (retrieval/generation) drift evaluation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from evaluator.latent_drift.schemas import EmbeddingBatch


class StreamingDriftBuffer:
    """In-memory buffer for streaming embedding vectors with dual-track support.

    Buffers are partitioned by track (``"retrieval"`` or ``"generation"``)
    and support two overflow strategies:

    - ``"reservoir"``: When at capacity, replaces a random element (reservoir
      sampling), ensuring uniform representation across the stream.
    - ``"fifo"``: When at capacity, drops the oldest element.

    Args:
        capacity: Maximum total vectors buffered across both tracks
            (default 1000).  Individual track capacity is split evenly
            when both tracks are active.
        sample_strategy: ``"reservoir"`` or ``"fifo"`` (default "reservoir").
    """

    def __init__(
        self,
        capacity: int = 1000,
        sample_strategy: Literal["reservoir", "fifo"] = "reservoir",
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if sample_strategy not in ("reservoir", "fifo"):
            raise ValueError(
                f"sample_strategy must be 'reservoir' or 'fifo', got '{sample_strategy}'"
            )

        self.capacity = capacity
        self.sample_strategy = sample_strategy
        self._per_track_capacity = capacity // 2
        self._buffers: dict[str, list[np.ndarray]] = {
            "retrieval": [],
            "generation": [],
        }
        self._rng = np.random.RandomState(42)

    def ingest(
        self,
        vector: np.ndarray,
        track: Literal["retrieval", "generation"] = "retrieval",
    ) -> None:
        """Add a single embedding vector to the buffer for a given track.

        Args:
            vector: 1-D embedding array.
            track: Which track buffer to write to.
        """
        if track not in self._buffers:
            self._buffers[track] = []

        buf = self._buffers[track]
        vec = np.atleast_1d(np.asarray(vector, dtype=float))

        if len(buf) < self._per_track_capacity:
            buf.append(vec)
        else:
            if self.sample_strategy == "reservoir":
                # Reservoir sampling: replace a random element
                idx = self._rng.randint(0, len(buf))
                buf[idx] = vec
            else:  # fifo
                buf.pop(0)
                buf.append(vec)

    def flush_batch(self) -> EmbeddingBatch:
        """Export the current buffer snapshot as an EmbeddingBatch.

        Combines both track buffers.  Does not reset baseline statistics.
        """
        all_vectors = []
        for track in ("retrieval", "generation"):
            all_vectors.extend(self._buffers.get(track, []))

        if not all_vectors:
            return EmbeddingBatch(vectors=np.empty((0, 0)))

        return EmbeddingBatch(vectors=np.vstack(all_vectors))

    def flush_track(self, track: Literal["retrieval", "generation"] = "retrieval") -> EmbeddingBatch:
        """Export the current buffer for a specific track.

        Args:
            track: Which track to export.

        Returns:
            An :class:`EmbeddingBatch` containing only the specified track's
            vectors.
        """
        buf = self._buffers.get(track, [])
        if not buf:
            return EmbeddingBatch(vectors=np.empty((0, 0)))

        return EmbeddingBatch(vectors=np.vstack(buf))

    def is_ready(self, min_samples: int = 50) -> bool:
        """Check if sufficient vectors exist for drift evaluation.

        Args:
            min_samples: Minimum total samples needed across both tracks.

        Returns:
            ``True`` if the combined buffer has at least ``min_samples`` vectors.
        """
        total = sum(len(buf) for buf in self._buffers.values())
        return total >= min_samples

    def is_track_ready(
        self,
        track: Literal["retrieval", "generation"] = "retrieval",
        min_samples: int = 50,
    ) -> bool:
        """Check if a specific track has sufficient samples.

        Args:
            track: Which track to check.
            min_samples: Minimum samples needed.

        Returns:
            ``True`` if the track has at least ``min_samples`` vectors.
        """
        return len(self._buffers.get(track, [])) >= min_samples

    def clear(self) -> None:
        """Clear all buffer contents."""
        for buf in self._buffers.values():
            buf.clear()

    def clear_track(self, track: Literal["retrieval", "generation"] = "retrieval") -> None:
        """Clear a specific track's buffer."""
        if track in self._buffers:
            self._buffers[track].clear()

    @property
    def sizes(self) -> dict[str, int]:
        """Return current buffer sizes per track."""
        return {track: len(buf) for track, buf in self._buffers.items()}

    def total_size(self) -> int:
        """Return the total number of buffered vectors across all tracks."""
        return sum(len(buf) for buf in self._buffers.values())
