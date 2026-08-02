import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

from alerting.notifier import DriftAlertNotifier
from cli import drift_cli
from evaluator.drift_monitor import DriftMonitor
from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    GraphTopologyPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)

DRIFTED_RESULT: Dict[str, Any] = {
    "vector_drift": {
        "js_divergence": 0.42,
        "mmd_score": 0.31,
        "is_drifted": True,
    },
    "graph_drift": {
        "spectral_distance": 0.87,
        "density_delta": 0.12,
        "node_count_delta": 3,
        "is_graph_drifted": True,
    },
    "swarm_drift": {
        "transition_entropy_delta": 1.24,
        "avg_reflection_iterations_delta": 2.0,
        "is_swarm_drifted": True,
    },
    "is_drifted": True,
}

STABLE_RESULT: Dict[str, Any] = {
    **DRIFTED_RESULT,
    "is_drifted": False,
    "graph_drift": {**DRIFTED_RESULT["graph_drift"], "is_graph_drifted": False},
    "swarm_drift": {**DRIFTED_RESULT["swarm_drift"], "is_swarm_drifted": False},
}

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
    def __init__(self, stats: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats or {
            "total_frames": 12,
            "by_rag_type": {"naive": 6, "graph_rag": 4, "swarm": 2},
            "frames_with_graph_payloads": 4,
            "frames_with_swarm_metadata": 2,
            "status": "healthy",
        }
        self.recent: List[RAGEvaluationFrame] = []
        self.recorded: List[RAGEvaluationFrame] = []

    async def get_store_stats(self) -> Dict[str, Any]:
        return self.stats

    async def get_recent_frames(
        self, rag_type: Optional[str] = None, limit: int = 100
    ) -> List[RAGEvaluationFrame]:
        return self.recent

    async def record_evaluation(
        self, frame: RAGEvaluationFrame, metrics: Dict[str, Any]
    ) -> None:
        self.recorded.append(frame)


class FakeNotifier:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.dispatched: List[Dict[str, Any]] = []

    async def notify_if_drifted(
        self, eval_result: Dict[str, Any], batch_id: Optional[str] = None
    ) -> bool:
        self.dispatched.append({"eval_result": eval_result, "batch_id": batch_id})
        return True

    def build_payload(
        self, eval_result: Dict[str, Any], batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return {"event": "drift_alert", "is_drifted": True, "batch_id": batch_id}


def _frame(
    *,
    embedding: Optional[List[float]] = None,
    rag_type: str = "naive",
    graph: Optional[Dict[str, Any]] = None,
    agent_hops: Optional[List[str]] = None,
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
        ),
        output=OutputPayload(generated_answer="answer"),
    )


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None


async def test_notifier_builds_structured_payload():
    notifier = DriftAlertNotifier(webhook_url="http://hooks.example/drift")

    payload = notifier.build_payload(DRIFTED_RESULT, batch_id="batch-1")

    assert payload["event"] == "drift_alert"
    assert payload["is_drifted"] is True
    assert payload["batch_id"] == "batch-1"
    assert payload["vector_drift"] == {"js_divergence": 0.42, "mmd_score": 0.31}
    assert payload["graph_drift"]["spectral_distance"] == 0.87
    assert payload["swarm_drift"]["transition_entropy_delta"] == 1.24


async def test_notifier_dispatches_alert_on_drift():
    client = AsyncMock()
    client.post = AsyncMock(return_value=_FakeResponse())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    notifier = DriftAlertNotifier(webhook_url="http://hooks.example/drift")
    with patch("alerting.notifier.httpx.AsyncClient", return_value=client):
        dispatched = await notifier.notify_if_drifted(DRIFTED_RESULT, batch_id="b1")

    assert dispatched is True
    client.post.assert_awaited_once()
    posted_url = client.post.await_args.args[0]
    payload = client.post.await_args.kwargs["json"]
    assert posted_url == "http://hooks.example/drift"
    assert payload["batch_id"] == "b1"
    assert payload["graph_drift"]["spectral_distance"] == 0.87


async def test_notifier_skips_dispatch_when_stable():
    notifier = DriftAlertNotifier(webhook_url="http://hooks.example/drift")

    with patch("alerting.notifier.httpx.AsyncClient") as mock_client_cls:
        dispatched = await notifier.notify_if_drifted(STABLE_RESULT)

    assert dispatched is False
    mock_client_cls.assert_not_called()


async def test_notifier_no_webhook_returns_false(monkeypatch):
    monkeypatch.delenv("DRIFT_ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setattr("alerting.notifier.config.DRIFT_ALERT_WEBHOOK_URL", None)
    notifier = DriftAlertNotifier()

    dispatched = await notifier.notify_if_drifted(DRIFTED_RESULT)

    assert dispatched is False


async def test_drift_monitor_triggers_alert_on_drift():
    notifier = AsyncMock(spec=DriftAlertNotifier)
    monitor = DriftMonitor(store=FakeRepo(), notifier=notifier)
    baseline = [
        _frame(embedding=[0.1, 0.2], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
        _frame(embedding=[0.11, 0.21], graph=PATH_GRAPH, agent_hops=["a", "b", "a"]),
    ]
    current = [
        _frame(
            embedding=[0.1, 0.2], graph=CYCLE_GRAPH, agent_hops=["a", "b", "c", "a"]
        ),
        _frame(
            embedding=[0.11, 0.21],
            graph=CYCLE_GRAPH,
            agent_hops=["a", "c", "b", "a"],
        ),
    ]

    result = await monitor.evaluate_frames(baseline, current)

    assert result["is_drifted"] is True
    notifier.notify_if_drifted.assert_awaited_once()
    sent_result = notifier.notify_if_drifted.await_args.args[0]
    assert sent_result["is_drifted"] is True


async def test_drift_monitor_skips_alert_when_stable():
    notifier = AsyncMock(spec=DriftAlertNotifier)
    monitor = DriftMonitor(store=FakeRepo(), notifier=notifier)
    frames = [
        _frame(embedding=[0.1, 0.2], graph=CYCLE_GRAPH, agent_hops=["a", "b"]),
        _frame(embedding=[0.11, 0.21], graph=CYCLE_GRAPH, agent_hops=["a", "b"]),
    ]

    result = await monitor.evaluate_frames(frames, frames)

    assert result["is_drifted"] is False
    notifier.notify_if_drifted.assert_awaited_once()
    sent_result = notifier.notify_if_drifted.await_args.args[0]
    assert sent_result["is_drifted"] is False


def test_cli_stats_prints_store_statistics(capsys):
    fake_store = FakeRepo()

    with patch("cli.drift_cli.DriftStore", return_value=fake_store):
        exit_code = drift_cli.main(["stats"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_frames"] == 12
    assert out["by_rag_type"]["graph_rag"] == 4
    assert out["frames_with_graph_payloads"] == 4
    assert out["frames_with_swarm_metadata"] == 2
    assert out["status"] == "healthy"


def test_cli_evaluate_prints_drift_assessment(capsys):
    fake_store = FakeRepo()
    fake_store.recent = [
        _frame(embedding=[0.1, 0.2], rag_type="naive"),
        _frame(embedding=[0.11, 0.21], rag_type="naive"),
    ]

    with (
        patch("cli.drift_cli.DriftStore", return_value=fake_store),
        patch("cli.drift_cli.DriftAlertNotifier", return_value=FakeNotifier()),
    ):
        exit_code = drift_cli.main(
            [
                "evaluate",
                "--baseline-id",
                "naive",
                "--current-id",
                "agentic",
                "--limit",
                "5",
            ]
        )

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"vector_drift", "graph_drift", "swarm_drift", "is_drifted"}
    assert out["is_drifted"] is False
    assert len(fake_store.recorded) == 2


def test_cli_test_alert_dispatches_payload(capsys):
    notifier = FakeNotifier(webhook_url="http://hooks.example/drift")

    with patch("cli.drift_cli.DriftAlertNotifier", return_value=notifier):
        exit_code = drift_cli.main(
            ["test-alert", "--webhook-url", "http://hooks.example/drift"]
        )

    assert exit_code == 0
    assert len(notifier.dispatched) == 1
    assert notifier.dispatched[0]["eval_result"]["is_drifted"] is True
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "alert dispatched"


def test_cli_test_alert_requires_webhook(capsys):
    notifier = FakeNotifier(webhook_url=None)

    with patch("cli.drift_cli.DriftAlertNotifier", return_value=notifier):
        exit_code = drift_cli.main(["test-alert"])

    assert exit_code == 1
    assert "No webhook URL configured" in capsys.readouterr().err
