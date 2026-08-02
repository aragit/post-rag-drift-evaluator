import numpy as np
import pytest
from unittest.mock import AsyncMock

from evaluator.baseline_service import DynamicBaselineService
from evaluator.config import config
from evaluator.drift_monitor import DriftMonitor
from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    QueryPayload,
    RAGEvaluationFrame,
    OutputPayload,
    RetrievalContextPayload,
)


def _make_frame(embedding=None, graph_topology=None, metadata=None):
    return RAGEvaluationFrame(
        query=QueryPayload(text="test", embedding=embedding or [0.1, 0.2]),
        context=RetrievalContextPayload(
            text_chunks=["chunk"],
            graph_topology=graph_topology,
        ),
        metadata=metadata or ExecutionMetadataPayload(rag_type="naive"),
        output=OutputPayload(generated_answer="answer"),
    )


@pytest.fixture
def baseline_frames():
    np.random.seed(42)
    frames = []
    for _ in range(50):
        embedding = np.random.normal(0, 1, 8).tolist()
        frames.append(_make_frame(embedding=embedding))
    return frames


@pytest.fixture
def current_frames():
    np.random.seed(99)
    frames = []
    for _ in range(5):
        embedding = np.random.normal(0, 1, 8).tolist()
        frames.append(_make_frame(embedding=embedding))
    return frames


# ---------------------------------------------------------------------------
# fetch_sliding_baseline_frames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_returns_frames_from_store():
    store = AsyncMock()
    store.get_frames_by_time_window = AsyncMock(return_value=["frame_a", "frame_b"])
    svc = DynamicBaselineService(store=store)
    result = await svc.fetch_sliding_baseline_frames()
    assert result == ["frame_a", "frame_b"]
    store.get_frames_by_time_window.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_empty_when_no_store():
    svc = DynamicBaselineService(store=None)
    result = await svc.fetch_sliding_baseline_frames()
    assert result == []


@pytest.mark.asyncio
async def test_fetch_uses_default_window_and_limit():
    store = AsyncMock()
    store.get_frames_by_time_window = AsyncMock(return_value=[])
    svc = DynamicBaselineService(store=store)
    await svc.fetch_sliding_baseline_frames(window_hours=48, limit=200)
    store.get_frames_by_time_window.assert_awaited_with(hours=48, limit=200)


# ---------------------------------------------------------------------------
# compute_calibrated_thresholds
# ---------------------------------------------------------------------------

def test_compute_returns_empty_for_insufficient_frames():
    svc = DynamicBaselineService(store=AsyncMock())
    result = svc.compute_calibrated_thresholds(["a"] * 5)
    assert result == {}


def test_compute_returns_empty_for_empty_list():
    svc = DynamicBaselineService(store=AsyncMock())
    result = svc.compute_calibrated_thresholds([])
    assert result == {}


def test_compute_calibrates_with_sufficient_frames(baseline_frames):
    svc = DynamicBaselineService(store=AsyncMock())
    thresholds = svc.compute_calibrated_thresholds(baseline_frames)
    assert "vector_jsd_threshold" in thresholds
    assert "vector_mmd_threshold" in thresholds
    assert "graph_spectral_threshold" in thresholds
    assert "swarm_entropy_threshold" in thresholds
    for v in thresholds.values():
        assert v >= 0


def test_compute_uses_config_min_baseline_frames(baseline_frames):
    svc = DynamicBaselineService(store=AsyncMock())
    just_below = baseline_frames[: config.MIN_BASELINE_FRAMES - 1]
    assert svc.compute_calibrated_thresholds(just_below) == {}


# ---------------------------------------------------------------------------
# DriftMonitor.evaluate_frames with dynamic baseline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_fetches_dynamic_baseline_when_none_provided(baseline_frames, current_frames):
    store = AsyncMock()
    store.get_frames_by_time_window = AsyncMock(return_value=baseline_frames)
    svc = DynamicBaselineService(store=store)
    monitor = DriftMonitor(store=store, baseline_service=svc, notifier=None)

    result = await monitor.evaluate_frames(None, current_frames)

    assert "vector_drift" in result
    assert "graph_drift" in result
    assert "swarm_drift" in result
    assert "is_drifted" in result
    store.get_frames_by_time_window.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_restores_thresholds_after_evaluation():
    store = AsyncMock()
    store.get_frames_by_time_window = AsyncMock(return_value=[])
    svc = DynamicBaselineService(store=store)
    monitor = DriftMonitor(store=store, baseline_service=svc)

    original_jsd = monitor.threshold
    original_mmd = monitor.mmd_threshold

    await monitor.evaluate_frames(None, [])

    assert monitor.threshold == original_jsd
    assert monitor.mmd_threshold == original_mmd
    assert monitor._saved_thresholds is None


@pytest.mark.asyncio
async def test_evaluate_explicit_baseline_skips_dynamic_fetch():
    store = AsyncMock()
    svc = DynamicBaselineService(store=store)
    monitor = DriftMonitor(store=store, baseline_service=svc)

    baseline = [_make_frame(embedding=[0.1, 0.2]) for _ in range(5)]
    current = [_make_frame(embedding=[0.9, 0.8]) for _ in range(3)]

    result = await monitor.evaluate_frames(baseline, current)

    store.get_frames_by_time_window.assert_not_awaited()
    assert "vector_drift" in result


@pytest.mark.asyncio
async def test_evaluate_falls_back_to_static_when_insufficient_frames():
    """When baseline fetches < MIN_BASELINE_FRAMES, static thresholds are used."""
    store = AsyncMock()
    store.get_frames_by_time_window = AsyncMock(
        return_value=[_make_frame(embedding=[0.1, 0.2]) for _ in range(5)]
    )
    svc = DynamicBaselineService(store=store)
    monitor = DriftMonitor(store=store, baseline_service=svc)

    original_threshold = monitor.threshold

    result = await monitor.evaluate_frames(None, [_make_frame(embedding=[0.1, 0.2])])

    assert monitor.threshold == original_threshold
    assert result["is_drifted"] in (True, False)


@pytest.mark.asyncio
async def test_evaluate_no_baseline_service_still_works():
    """When baseline_service is None and no baseline_frames given, empty baseline is used."""
    store = AsyncMock()
    monitor = DriftMonitor(store=store, baseline_service=None)

    result = await monitor.evaluate_frames(None, [_make_frame(embedding=[0.1, 0.2])])

    assert result["is_drifted"] is False
    store.get_frames_by_time_window.assert_not_awaited()


# ---------------------------------------------------------------------------
# Threshold save/restore
# ---------------------------------------------------------------------------

def test_apply_and_restore_thresholds():
    monitor = DriftMonitor(store=AsyncMock())
    original_jsd = monitor.threshold
    original_mmd = monitor.mmd_threshold
    original_spectral = monitor._graph_calculator.spectral_threshold
    original_entropy = monitor._swarm_calculator.entropy_threshold

    test_thresholds = {
        "vector_jsd_threshold": 0.5,
        "vector_mmd_threshold": 0.3,
        "graph_spectral_threshold": 2.0,
        "swarm_entropy_threshold": 1.5,
    }
    monitor._apply_calibrated_thresholds(test_thresholds)

    assert monitor.threshold == 0.5
    assert monitor.mmd_threshold == 0.3
    assert monitor._graph_calculator.spectral_threshold == 2.0
    assert monitor._swarm_calculator.entropy_threshold == 1.5
    assert monitor._calibrated is True

    monitor._restore_thresholds()

    assert monitor.threshold == original_jsd
    assert monitor.mmd_threshold == original_mmd
    assert monitor._graph_calculator.spectral_threshold == original_spectral
    assert monitor._swarm_calculator.entropy_threshold == original_entropy
    assert monitor._saved_thresholds is None
