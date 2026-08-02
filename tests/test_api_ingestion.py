import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import pytest

from api.app import create_app
from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)
from ingestion.queue import AsyncIngestionBuffer

PATH_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
    ],
}

CYCLE_GRAPH = {
    "nodes": [{"id": str(i)} for i in range(1, 5)],
    "edges": [
        {"source": "1", "target": "2"},
        {"source": "2", "target": "3"},
        {"source": "3", "target": "4"},
        {"source": "4", "target": "1"},
    ],
}


class FakeRepo:
    """In-memory stand-in for the DriftStore persistence contract."""

    def __init__(self, recent: Optional[List[RAGEvaluationFrame]] = None) -> None:
        self.batches: List[List[RAGEvaluationFrame]] = []
        self.recorded: List[RAGEvaluationFrame] = []
        self.recent = recent or []
        self.closed = False
        self.fail_batches = False
        self.flush_attempts = 0

    async def batch_store_frames(self, frames: List[RAGEvaluationFrame]) -> None:
        self.flush_attempts += 1
        if self.fail_batches:
            raise ConnectionError("simulated persistence outage")
        self.batches.append(list(frames))

    async def record_evaluation(
        self, frame: RAGEvaluationFrame, metrics: Dict[str, Any]
    ) -> None:
        self.recorded.append(frame)

    async def get_recent_frames(
        self, rag_type: Optional[str] = None, limit: int = 100
    ) -> List[RAGEvaluationFrame]:
        return self.recent

    async def get_frames_by_time_window(
        self, hours: int = 24, limit: int = 100
    ) -> List[RAGEvaluationFrame]:
        return self.recent

    async def close(self) -> None:
        self.closed = True


def _frame(
    *,
    embedding: Optional[List[float]] = None,
    rag_type: str = "naive",
    graph: Optional[Dict[str, Any]] = None,
    agent_hops: Optional[List[str]] = None,
    reflection_iterations: int = 0,
) -> RAGEvaluationFrame:
    return RAGEvaluationFrame(
        query=QueryPayload(text="test query", embedding=embedding),
        context=RetrievalContextPayload(
            text_chunks=["chunk"],
            graph_topology=GraphTopologyPayload(**graph) if graph else None,
        ),
        metadata=ExecutionMetadataPayload(
            rag_type=rag_type,
            agent_hops=agent_hops,
            reflection_iterations=reflection_iterations,
        ),
        output=OutputPayload(generated_answer="answer"),
    )


def _frame_json(frame: RAGEvaluationFrame) -> Dict[str, Any]:
    return frame.model_dump(mode="json")


@asynccontextmanager
async def api_client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


async def test_ingest_single_frame_returns_202():
    repo = FakeRepo()
    app = create_app(store=repo)

    async with api_client(app) as client:
        response = await client.post("/v1/telemetry/frames", json=_frame_json(_frame()))

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "count": 1}


async def test_ingest_batch_returns_202():
    repo = FakeRepo()
    app = create_app(store=repo)
    frames = [_frame_json(_frame(embedding=[0.1, i])) for i in range(4)]

    async with api_client(app) as client:
        response = await client.post("/v1/telemetry/frames", json={"frames": frames})

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "count": 4}


async def test_ingested_frames_persist_via_background_worker():
    repo = FakeRepo()
    app = create_app(store=repo)
    frames = [_frame(embedding=[0.1, i]) for i in range(5)]

    async with api_client(app) as client:
        response = await client.post(
            "/v1/telemetry/frames",
            json={"frames": [_frame_json(f) for f in frames]},
        )
        assert response.status_code == 202

    stored = [frame for batch in repo.batches for frame in batch]
    assert len(stored) == 5
    assert {f.trace_id for f in stored} == {f.trace_id for f in frames}
    assert repo.closed is True


async def test_evaluate_returns_multi_modal_drift():
    repo = FakeRepo()
    app = create_app(store=repo)
    baseline = [
        _frame(embedding=[0.1, 0.2], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
        _frame(embedding=[0.11, 0.21], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
    ]
    current = [
        _frame(
            embedding=[0.1, 0.2],
            graph=CYCLE_GRAPH,
            agent_hops=["a", "b", "c", "a"],
        ),
        _frame(
            embedding=[0.11, 0.21],
            graph=CYCLE_GRAPH,
            agent_hops=["a", "c", "b", "a"],
        ),
    ]

    async with api_client(app) as client:
        response = await client.post(
            "/v1/telemetry/evaluate",
            json={
                "baseline_frames": [_frame_json(f) for f in baseline],
                "current_frames": [_frame_json(f) for f in current],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "vector_drift",
        "graph_drift",
        "swarm_drift",
        "is_drifted",
    }
    assert payload["graph_drift"]["is_graph_drifted"] is True
    assert payload["swarm_drift"]["is_swarm_drifted"] is True
    assert payload["is_drifted"] is True
    assert len(repo.recorded) == len(current)


async def test_evaluate_uses_dynamic_baseline_fallback():
    """When no baseline source is provided, dynamic baseline is fetched."""
    repo = FakeRepo()
    app = create_app(store=repo)

    async with api_client(app) as client:
        response = await client.post(
            "/v1/telemetry/evaluate",
            json={"current_frames": [_frame_json(_frame())]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "vector_drift",
        "graph_drift",
        "swarm_drift",
        "is_drifted",
    }
    assert payload["is_drifted"] is False


async def test_evaluate_uses_baseline_batch_id():
    baseline = [_frame(embedding=[0.1, 0.2])]
    repo = FakeRepo(recent=baseline)
    app = create_app(store=repo)

    async with api_client(app) as client:
        response = await client.post(
            "/v1/telemetry/evaluate",
            json={
                "baseline_batch_id": "batch-123",
                "current_frames": [_frame_json(_frame(embedding=[0.1, 0.2]))],
            },
        )

    assert response.status_code == 200
    assert response.json()["is_drifted"] is False


async def test_high_concurrency_ingestion_all_accepted_and_persisted():
    repo = FakeRepo()
    app = create_app(store=repo, buffer=AsyncIngestionBuffer(batch_size=50))
    frame_json = _frame_json(_frame())

    async def _send(count: int) -> int:
        response = await client.post(
            "/v1/telemetry/frames", json={"frames": [frame_json] * count}
        )
        return response.status_code

    async with api_client(app) as client:
        results = await asyncio.gather(*[_send(10) for _ in range(20)])

    assert all(status == 202 for status in results)
    stored = [frame for batch in repo.batches for frame in batch]
    assert len(stored) == 200


async def test_worker_survives_persistence_failure():
    repo = FakeRepo()
    app = create_app(store=repo)
    repo.fail_batches = True

    async with api_client(app) as client:
        first = await client.post(
            "/v1/telemetry/frames", json={"frames": [_frame_json(_frame())]}
        )
        assert first.status_code == 202
        for _ in range(100):
            if repo.flush_attempts >= 1:
                break
            await asyncio.sleep(0.01)
        assert repo.flush_attempts >= 1
        repo.fail_batches = False
        second = await client.post(
            "/v1/telemetry/frames", json={"frames": [_frame_json(_frame())]}
        )
        assert second.status_code == 202

    stored = [frame for batch in repo.batches for frame in batch]
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_buffer_flushes_remaining_frames_on_stop():
    repo = FakeRepo()
    buffer = AsyncIngestionBuffer(batch_size=2)
    await buffer.start_worker(repo)
    frames = [_frame(embedding=[0.1, i]) for i in range(5)]

    await buffer.enqueue(frames)
    await buffer.stop_worker()

    stored = [frame for batch in repo.batches for frame in batch]
    assert len(stored) == 5


@pytest.mark.asyncio
async def test_buffer_enqueue_is_non_blocking():
    repo = FakeRepo()
    buffer = AsyncIngestionBuffer()
    await buffer.start_worker(repo)
    frames = [_frame(embedding=[0.1, i]) for i in range(3)]

    await buffer.enqueue(frames)

    assert buffer.pending == 3
    await buffer.stop_worker()
    assert buffer.pending == 0
