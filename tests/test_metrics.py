from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import pytest
from prometheus_client import REGISTRY

from api.app import create_app
from ingestion.queue import AsyncIngestionBuffer
from tests.test_api_ingestion import FakeRepo, _frame, _frame_json

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


@asynccontextmanager
async def _client_with_repo(repo):
    buffer = AsyncIngestionBuffer(batch_size=50)
    app = create_app(store=repo, buffer=buffer)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


def _sample_value(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {})


async def test_metrics_endpoint_returns_200_with_prometheus_format():
    repo = FakeRepo()
    async with _client_with_repo(repo) as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    body = response.text
    assert "frame_ingestion_total" in body
    assert "ingestion_buffer_depth" in body
    assert "evaluation_latency_seconds" in body
    assert "drift_score_gauge" in body
    assert "drift_alerts_total" in body
    assert "db_batch_write_latency_seconds" in body


async def test_frame_ingestion_metric_increments():
    repo = FakeRepo()
    async with _client_with_repo(repo) as client:
        before = (
            _sample_value(
                "frame_ingestion_total", {"status": "accepted", "buffer_type": "memory"}
            )
            or 0.0
        )

        response = await client.post("/v1/telemetry/frames", json=_frame_json(_frame()))
        assert response.status_code == 202

        after = _sample_value(
            "frame_ingestion_total", {"status": "accepted", "buffer_type": "memory"}
        )
        assert after is not None
        assert after == pytest.approx(before + 1, abs=0.5)


async def test_evaluation_latency_metric_observed():
    repo = FakeRepo()
    async with _client_with_repo(repo) as client:
        baseline = [_frame(embedding=[0.1, 0.2])]
        current = [_frame(embedding=[0.1, 0.2], graph=PATH_GRAPH)]

        response = await client.post(
            "/v1/telemetry/evaluate",
            json={
                "baseline_frames": [_frame_json(f) for f in baseline],
                "current_frames": [_frame_json(f) for f in current],
            },
        )
        assert response.status_code == 200

        latency_count = _sample_value(
            "evaluation_latency_seconds_count", {"status": "success"}
        )
        assert latency_count is not None
        assert latency_count >= 1.0


async def test_drift_score_gauges_updated():
    repo = FakeRepo()
    async with _client_with_repo(repo) as client:
        baseline = [
            _frame(embedding=[0.1, 0.2], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
            _frame(
                embedding=[0.11, 0.21], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]
            ),
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

        response = await client.post(
            "/v1/telemetry/evaluate",
            json={
                "baseline_frames": [_frame_json(f) for f in baseline],
                "current_frames": [_frame_json(f) for f in current],
            },
        )
        assert response.status_code == 200

        jsd = _sample_value("drift_score_gauge", {"metric_type": "vector_jsd"})
        mmd = _sample_value("drift_score_gauge", {"metric_type": "vector_mmd"})
        spectral = _sample_value("drift_score_gauge", {"metric_type": "graph_spectral"})
        entropy = _sample_value("drift_score_gauge", {"metric_type": "swarm_entropy"})

        assert jsd is not None
        assert mmd is not None
        assert spectral is not None
        assert entropy is not None

        assert spectral > 0.0
        assert entropy > 0.0


async def test_db_batch_write_latency_observed_after_ingestion():
    repo = FakeRepo()
    async with _client_with_repo(repo) as client:
        before = _sample_value("db_batch_write_latency_seconds_count") or 0.0

        await client.post("/v1/telemetry/frames", json=_frame_json(_frame()))

        for _ in range(200):
            if repo.flush_attempts >= 1:
                break
            import asyncio as _asyncio

            await _asyncio.sleep(0.05)

        after = _sample_value("db_batch_write_latency_seconds_count")
        assert after is not None
        assert after > before
