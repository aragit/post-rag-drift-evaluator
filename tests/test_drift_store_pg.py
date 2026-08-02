import json
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from evaluator.db.init_drift_tables import init_drift_tables
from evaluator.drift_store import DriftStore
from evaluator.schemas.telemetry import (
    ExecutionMetadataPayload,
    OutputPayload,
    QueryPayload,
    RAGEvaluationFrame,
    RetrievalContextPayload,
)


class _Row:
    """Minimal stand-in for an ``asyncpg.Record`` supporting ``row["col"]``."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]


def _frame(rag_type: str = "naive") -> RAGEvaluationFrame:
    return RAGEvaluationFrame(
        trace_id="6f9c5e32-1f21-4b1a-9d3a-1234567890ab",
        query=QueryPayload(text="What is the capital?", embedding=[0.1, 0.2]),
        context=RetrievalContextPayload(text_chunks=["Paris is the capital."]),
        metadata=ExecutionMetadataPayload(rag_type=rag_type),
        output=OutputPayload(generated_answer="Paris.", confidence_score=0.9),
    )


def _patch_db(conn: AsyncMock) -> ExitStack:
    """Mock the drift store's schema bootstrap and DB pool access."""
    stack = ExitStack()
    stack.enter_context(
        patch("evaluator.drift_store.init_drift_tables", new=AsyncMock())
    )
    stack.enter_context(
        patch("evaluator.db.pool.acquire", new=AsyncMock(return_value=conn))
    )
    stack.enter_context(patch("evaluator.db.pool.release", new=AsyncMock()))
    return stack


@pytest.mark.asyncio
async def test_record_evaluation_inserts_full_frame_as_jsonb():
    conn = AsyncMock()
    frame = _frame()

    with _patch_db(conn):
        await DriftStore().record_evaluation(
            frame, {"js_divergence": 0.21, "mmd_score": 0.15, "is_drifted": True}
        )

    conn.execute.assert_awaited_once()
    sql, *params = conn.execute.call_args.args
    assert "INSERT INTO telemetry_evaluations" in sql

    row_id, trace_id, rag_type, timestamp, jsd, mmd, wasser, drifted, frame_json = (
        params
    )
    assert isinstance(row_id, uuid.UUID)
    assert trace_id == frame.trace_id
    assert rag_type == "naive"
    assert isinstance(timestamp, datetime)
    assert jsd == 0.21
    assert mmd == 0.15
    assert wasser is None
    assert drifted is True

    payload = json.loads(frame_json)
    assert payload["query"]["text"] == "What is the capital?"
    assert payload["metadata"]["rag_type"] == "naive"


@pytest.mark.asyncio
async def test_record_evaluation_defaults_metrics():
    conn = AsyncMock()

    with _patch_db(conn):
        await DriftStore().record_evaluation(_frame(), {"js_divergence": 0.1})

    _, *params = conn.execute.call_args.args
    _, _, _, _, jsd, mmd, wasser, drifted, _ = params
    assert jsd == 0.1
    assert mmd is None
    assert drifted is False


@pytest.mark.asyncio
async def test_get_recent_frames_deserializes_payloads():
    conn = AsyncMock()
    stored = _frame(rag_type="swarm")
    conn.fetch = AsyncMock(
        return_value=[_Row({"telemetry_frame": stored.model_dump_json()})]
    )

    with _patch_db(conn):
        frames = await DriftStore().get_recent_frames(limit=5)

    assert len(frames) == 1
    assert isinstance(frames[0], RAGEvaluationFrame)
    assert frames[0].trace_id == stored.trace_id
    assert frames[0].query.text == "What is the capital?"
    assert frames[0].metadata.rag_type == "swarm"


@pytest.mark.asyncio
async def test_get_recent_frames_filters_by_rag_type():
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])

    with _patch_db(conn):
        await DriftStore().get_recent_frames(rag_type="graph_rag", limit=3)

    sql, rag_type, limit = conn.fetch.call_args.args
    assert "WHERE rag_type = $1" in sql
    assert rag_type == "graph_rag"
    assert limit == 3


@pytest.mark.asyncio
async def test_get_recent_frames_skips_legacy_non_frame_payloads():
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            _Row({"telemetry_frame": json.dumps({"js_divergence": 0.5})}),
            _Row({"telemetry_frame": _frame().model_dump_json()}),
        ]
    )

    with _patch_db(conn):
        frames = await DriftStore().get_recent_frames()

    assert len(frames) == 1
    assert frames[0].query.text == "What is the capital?"


@pytest.mark.asyncio
async def test_record_drift_legacy_adapter():
    conn = AsyncMock()

    with _patch_db(conn):
        drift_id = await DriftStore().record_drift(
            {"js_divergence": 0.4, "mmd_score": 0.1, "is_drifted": False}
        )

    assert isinstance(drift_id, str)
    _, *params = conn.execute.call_args.args
    _, trace_id, rag_type, _, jsd, mmd, _, drifted, frame_json = params
    assert trace_id
    assert rag_type == "custom"
    assert jsd == 0.4
    assert mmd == 0.1
    assert drifted is False
    assert json.loads(frame_json)["js_divergence"] == 0.4


@pytest.mark.asyncio
async def test_get_latest_drift_returns_metrics_dict():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value=_Row(
            {
                "id": str(uuid.uuid4()),
                "trace_id": "trace-1",
                "rag_type": "custom",
                "timestamp": datetime.now(timezone.utc),
                "js_divergence": 0.31,
                "mmd_score": 0.12,
                "is_drifted": True,
                "telemetry_frame": json.dumps(
                    {"mmd_p_value": 0.005, "max_component_kl": 0.7}
                ),
            }
        )
    )

    with _patch_db(conn):
        latest = await DriftStore().get_latest_drift()

    assert latest is not None
    assert latest["jsd_score"] == 0.31
    assert latest["is_drifted"] is True
    assert latest["mmd_p_value"] == 0.005
    assert latest["max_component_kl"] == 0.7


@pytest.mark.asyncio
async def test_get_latest_drift_returns_none_when_empty():
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)

    with _patch_db(conn):
        latest = await DriftStore().get_latest_drift()

    assert latest is None


@pytest.mark.asyncio
async def test_get_recent_history_builds_polars_frame():
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            _Row(
                {
                    "id": uuid.uuid4(),
                    "trace_id": "trace-1",
                    "rag_type": "custom",
                    "timestamp": datetime.now(timezone.utc),
                    "js_divergence": 0.22,
                    "mmd_score": 0.1,
                    "is_drifted": True,
                    "telemetry_frame": json.dumps({"mmd_p_value": 0.01}),
                }
            )
        ]
    )

    with _patch_db(conn):
        history = await DriftStore().get_recent_history(hours=24)

    assert history.height == 1
    assert history["jsd_score"][0] == 0.22
    assert history["is_drifted"][0] is True


@pytest.mark.asyncio
async def test_clear_history_truncates_table():
    conn = AsyncMock()

    with _patch_db(conn):
        await DriftStore().clear_history()

    conn.execute.assert_awaited_once()
    assert "TRUNCATE TABLE telemetry_evaluations" in conn.execute.call_args.args[0]


@pytest.mark.asyncio
async def test_get_trend_returns_rolling_columns_when_enough_rows():
    conn = AsyncMock()
    now = datetime.now(timezone.utc)
    rows = [
        _Row({"timestamp": now, "js_divergence": 0.1 + i * 0.01}) for i in range(35)
    ]
    conn.fetch = AsyncMock(return_value=rows)

    with _patch_db(conn):
        trend = await DriftStore().get_trend(window=30)

    assert "rolling_mean" in trend.columns
    assert "rolling_std" in trend.columns


@pytest.mark.asyncio
async def test_close_releases_owned_pool():
    pool = AsyncMock()
    store = DriftStore(pool=pool)

    await store.close()

    pool.close.assert_awaited_once()
    assert store._pool is None


@pytest.mark.asyncio
async def test_init_drift_tables_runs_idempotent_ddl():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value = conn

    await init_drift_tables(pool=pool)

    pool.acquire.assert_awaited_once()
    pool.release.assert_awaited_once()
    conn.execute.assert_awaited_once()
    ddl = conn.execute.call_args.args[0]
    assert "CREATE TABLE IF NOT EXISTS telemetry_evaluations" in ddl
    assert "idx_telemetry_trace_id" in ddl
    assert "idx_telemetry_rag_type" in ddl
    assert "idx_telemetry_timestamp" in ddl
