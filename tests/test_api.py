from __future__ import annotations

import os
import tempfile

import httpx
import pytest

from api.main import create_app
from evaluator.metrics.results import DriftResult
from evaluator.storage import EvaluationRecord, JSONHistoryStore
from evaluator.temporal.models import DriftEvent


def make_drift_record(
    run_id: str,
    timestamp: float,
    value: float,
    system_version: str = "0.1.0",
    metadata: dict | None = None,
) -> EvaluationRecord:
    if metadata is None:
        metadata = {}
    return EvaluationRecord(
        run_id=run_id,
        timestamp=timestamp,
        system_version=system_version,
        metadata=metadata,
        metrics=[
            DriftResult(
                metric_name="js_divergence",
                value=value,
                current_run_id=run_id,
            )
        ],
    )


def _populate_store(store_path: str) -> None:
    store = JSONHistoryStore(store_path)
    for i in range(5):
        store.save(
            make_drift_record(f"r{i}", timestamp=i, value=0.05, system_version="0.1.0")
        )
    for i in range(5, 10):
        store.save(
            make_drift_record(
                f"r{i}",
                timestamp=i,
                value=0.30 + (i - 5) * 0.05,
                system_version="0.2.0",
            )
        )


@pytest.fixture
def store_path():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "history.jsonl")
    _populate_store(path)
    return path


@pytest.fixture
def empty_store_path():
    tmpdir = tempfile.mkdtemp()
    return os.path.join(tmpdir, "empty.jsonl")


def _make_client(store_path: str):
    app = create_app(store_path=store_path)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), app


# ── Drift Endpoint ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drift_endpoint_returns_events(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/drift",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["metric_name"] == "js_divergence"
    assert data["count"] > 0
    assert len(data["events"]) > 0
    assert "event_id" in data["events"][0]


@pytest.mark.asyncio
async def test_drift_endpoint_no_events(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/drift",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.99},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["events"] == []


# ── Attribution Endpoint ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attribution_endpoint_returns_factors(store_path):
    client, app = _make_client(store_path)
    try:
        drift_resp = await client.post(
            "/drift",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
        drift_data = drift_resp.json()
        assert drift_data["count"] > 0

        attr_resp = await client.post(
            "/attribution",
            json={"drift_event": drift_data["events"][0]},
        )
    finally:
        await client.aclose()

    assert attr_resp.status_code == 200
    data = attr_resp.json()
    assert "attribution" in data
    assert data["metric_name"] == "js_divergence"
    assert data["num_factors"] > 0
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_attribution_endpoint_invalid_drift_event(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/attribution",
            json={"drift_event": {"invalid": "payload"}},
        )
    finally:
        await client.aclose()

    assert response.status_code == 400


# ── Counterfactual Endpoint ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_counterfactual_endpoint_returns_results(store_path):
    client, app = _make_client(store_path)
    try:
        drift_resp = await client.post(
            "/drift",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
        drift_event = drift_resp.json()["events"][0]

        attr_resp = await client.post(
            "/attribution",
            json={"drift_event": drift_event},
        )
        attribution = attr_resp.json()["attribution"]

        cf_resp = await client.post(
            "/counterfactual",
            json={
                "drift_event": drift_event,
                "attribution": attribution,
                "top_k": 3,
            },
        )
    finally:
        await client.aclose()

    assert cf_resp.status_code == 200
    data = cf_resp.json()
    assert data["count"] > 0
    assert len(data["results"]) > 0
    assert "delta" in data["results"][0]
    assert "counterfactual_metric" in data["results"][0]


@pytest.mark.asyncio
async def test_counterfactual_endpoint_invalid_attribution(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/counterfactual",
            json={
                "drift_event": {
                    "metric_name": "test",
                    "start_timestamp": 0,
                    "end_timestamp": 1,
                    "magnitude": 0.5,
                },
                "attribution": {"invalid": "payload"},
                "top_k": 3,
            },
        )
    finally:
        await client.aclose()

    assert response.status_code == 400


# ── Optimization Endpoint ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_optimize_endpoint_full_pipeline(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/optimize",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    assert data["drift_events_found"] > 0
    assert data["attribution_factors"] > 0
    assert data["counterfactual_results"] > 0
    assert data["recommendations"] > 0

    plan = data["plan"]
    assert "summary" in plan
    assert "recommendations" in plan
    assert len(plan["recommendations"]) > 0
    top = plan["recommendations"][0]
    assert "action" in top
    assert "action_type" in top["action"]
    assert top["priority"] == 1
    assert top["expected_improvement"] > 0


@pytest.mark.asyncio
async def test_optimize_endpoint_no_drift(empty_store_path):
    client, app = _make_client(empty_store_path)
    try:
        response = await client.post(
            "/optimize",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.99},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["drift_events_found"] == 0
    assert "No drift detected" in data["plan"]["summary"]


@pytest.mark.asyncio
async def test_optimize_endpoint_summary(store_path):
    client, app = _make_client(store_path)
    try:
        response = await client.post(
            "/optimize",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    summary = data["plan"]["summary"]
    assert "recommendation" in summary.lower()


@pytest.mark.asyncio
async def test_optimize_endpoint_deterministic(store_path):
    client, app = _make_client(store_path)
    try:
        r1 = await client.post(
            "/optimize",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
        data1 = r1.json()

        r2 = await client.post(
            "/optimize",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
        data2 = r2.json()
    finally:
        await client.aclose()

    assert data1["plan"]["summary"] == data2["plan"]["summary"]
    assert data1["recommendations"] == data2["recommendations"]


# ── Error Handling ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drift_endpoint_empty_store(empty_store_path):
    client, app = _make_client(empty_store_path)
    try:
        response = await client.post(
            "/drift",
            json={"metric_name": "js_divergence", "window_size": 3, "threshold": 0.15},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_attribution_endpoint_empty_store(empty_store_path):
    client, app = _make_client(empty_store_path)
    try:
        drift_event = DriftEvent(
            metric_name="js_divergence",
            start_timestamp=3.0,
            end_timestamp=5.0,
            magnitude=0.45,
        )
        response = await client.post(
            "/attribution",
            json={"drift_event": drift_event.to_dict()},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    data = response.json()
    assert data["num_factors"] == 0
