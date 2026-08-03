from __future__ import annotations

import numpy as np
import pytest

from evaluator.latent_drift.schemas import EmbeddingBatch
from evaluator.latent_drift.streaming import StreamingDriftBuffer


@pytest.fixture
def small_buffer():
    return StreamingDriftBuffer(capacity=20)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ── Basic Ingestion Tests ───────────────────────────────────────────────


def test_ingest_single_vector(rng):
    """Ingesting one vector should store it."""
    buf = StreamingDriftBuffer(capacity=100)
    vec = rng.normal(0, 1, size=10)
    buf.ingest(vec, track="retrieval")
    assert buf.total_size() == 1


def test_ingest_dual_track(rng):
    """Vectors should be stored separately per track."""
    buf = StreamingDriftBuffer(capacity=100)
    vec = rng.normal(0, 1, size=10)

    buf.ingest(vec, track="retrieval")
    buf.ingest(vec, track="generation")

    sizes = buf.sizes
    assert sizes["retrieval"] == 1
    assert sizes["generation"] == 1


def test_ingest_accepts_list(rng):
    """Ingesting a list should be converted to ndarray."""
    buf = StreamingDriftBuffer(capacity=100)
    buf.ingest([0.1, 0.2, 0.3], track="retrieval")
    batch = buf.flush_batch()
    assert batch.vectors.shape == (1, 3)


# ── Capacity Enforcement Tests ────────────────────────────────────────


def test_reservoir_capacity_enforced(rng):
    """Reservoir buffer should never exceed capacity."""
    buf = StreamingDriftBuffer(capacity=10, sample_strategy="reservoir")
    vec = rng.normal(0, 1, size=5)

    for _ in range(100):
        buf.ingest(vec, track="retrieval")

    # Per-track capacity is capacity // 2 = 5
    sizes = buf.sizes
    assert sizes["retrieval"] <= 5
    assert sizes["generation"] <= 5


def test_fifo_capacity_enforced(rng):
    """FIFO buffer should never exceed capacity."""
    buf = StreamingDriftBuffer(capacity=10, sample_strategy="fifo")
    vec = rng.normal(0, 1, size=5)

    for _ in range(100):
        buf.ingest(vec, track="retrieval")

    sizes = buf.sizes
    assert sizes["retrieval"] <= 5


def test_invalid_capacity_raises():
    """Non-positive capacity should raise ValueError."""
    try:
        StreamingDriftBuffer(capacity=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        StreamingDriftBuffer(capacity=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_strategy_raises():
    """Invalid sample strategy should raise ValueError."""
    try:
        StreamingDriftBuffer(capacity=100, sample_strategy="invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ── Flush Tests ────────────────────────────────────────────────────────


def test_flush_batch_returns_embedding_batch(rng):
    """flush_batch should return an EmbeddingBatch."""
    buf = StreamingDriftBuffer(capacity=100)
    for _ in range(10):
        buf.ingest(rng.normal(0, 1, size=5), track="retrieval")
    for _ in range(10):
        buf.ingest(rng.normal(0, 1, size=5), track="generation")

    batch = buf.flush_batch()
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors.shape == (20, 5)


def test_flush_track_returns_specific_track(rng):
    """flush_track should return only the specified track's vectors."""
    buf = StreamingDriftBuffer(capacity=100)
    for _ in range(10):
        buf.ingest(rng.normal(0, 1, size=5), track="retrieval")
    for _ in range(15):
        buf.ingest(rng.normal(0, 1, size=5), track="generation")

    ret_batch = buf.flush_track("retrieval")
    gen_batch = buf.flush_track("generation")

    assert ret_batch.vectors.shape == (10, 5)
    assert gen_batch.vectors.shape == (15, 5)


def test_flush_batch_empty_buffer():
    """Empty buffer flush should return empty EmbeddingBatch."""
    buf = StreamingDriftBuffer(capacity=100)
    batch = buf.flush_batch()
    assert isinstance(batch, EmbeddingBatch)
    assert batch.vectors.shape == (0, 0)


def test_flush_does_not_clear_buffer():
    """flush_batch should not clear the buffer."""
    buf = StreamingDriftBuffer(capacity=100)
    vec = np.array([0.1, 0.2, 0.3])
    buf.ingest(vec, track="retrieval")

    _ = buf.flush_batch()
    assert buf.total_size() == 1


# ── Readiness Tests ────────────────────────────────────────────────────


def test_is_ready_below_threshold(rng):
    """is_ready should return False when buffer is below min_samples."""
    buf = StreamingDriftBuffer(capacity=1000)
    buf.ingest(rng.normal(0, 1, size=10), track="retrieval")
    assert not buf.is_ready(min_samples=50)


def test_is_ready_above_threshold(rng):
    """is_ready should return True when buffer reaches min_samples."""
    buf = StreamingDriftBuffer(capacity=1000)
    for _ in range(50):
        buf.ingest(rng.normal(0, 1, size=10), track="retrieval")
    assert buf.is_ready(min_samples=50)


def test_is_track_ready(rng):
    """is_track_ready should check specific track size."""
    buf = StreamingDriftBuffer(capacity=1000)
    for _ in range(50):
        buf.ingest(rng.normal(0, 1, size=10), track="retrieval")
    assert buf.is_track_ready("retrieval", min_samples=50)
    assert not buf.is_track_ready("generation", min_samples=50)


# ── Clear Tests ────────────────────────────────────────────────────────


def test_clear_empties_all_buffers(rng):
    """clear should empty all track buffers."""
    buf = StreamingDriftBuffer(capacity=100)
    buf.ingest(rng.normal(0, 1, size=5), track="retrieval")
    buf.ingest(rng.normal(0, 1, size=5), track="generation")

    buf.clear()
    assert buf.total_size() == 0
    assert len(buf.sizes) == 2


def test_clear_track_empties_specific_buffer(rng):
    """clear_track should only empty the specified track."""
    buf = StreamingDriftBuffer(capacity=100)
    buf.ingest(rng.normal(0, 1, size=5), track="retrieval")
    buf.ingest(rng.normal(0, 1, size=5), track="generation")

    buf.clear_track("retrieval")
    assert buf.sizes["retrieval"] == 0
    assert buf.sizes["generation"] == 1
