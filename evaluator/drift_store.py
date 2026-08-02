import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from evaluator.db import pool as db_pool
from evaluator.db.init_drift_tables import init_drift_tables
from evaluator.schemas.telemetry import RAGEvaluationFrame

logger = logging.getLogger("DriftStore")

INSERT_EVALUATION_SQL = """
INSERT INTO telemetry_evaluations
    (id, trace_id, rag_type, timestamp, js_divergence, mmd_score,
     wasserstein_distance, is_drifted, telemetry_frame)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
"""


def _coerce_frame_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a stored JSONB telemetry payload, tolerating legacy dicts."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(raw, dict):
        return raw
    return None


class DriftStore:
    """PostgreSQL-backed persistence for drift metrics and telemetry frames.

    All access is async via the shared asyncpg pool (``evaluator.db.pool``)
    unless an explicit loop-local pool is injected via ``pool``.
    """

    def __init__(self, pool: Optional[Any] = None):
        self._pool = pool
        self._ready = False

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        await init_drift_tables(pool=self._pool)
        self._ready = True

    async def _connection(self) -> Tuple[Any, bool]:
        await self._ensure_ready()
        if self._pool is not None:
            conn = await self._pool.acquire()
            return conn, True
        conn = await db_pool.acquire()
        return conn, False

    async def _release(self, conn: Any, owned: bool) -> None:
        if owned:
            await self._pool.release(conn)
        else:
            await db_pool.release(conn)

    async def close(self) -> None:
        """Release any pool owned by this store instance.

        The shared application pool is managed by ``evaluator.db.pool``
        and is closed centrally during shutdown.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._ready = False
        logger.info("DriftStore released its connection pool.")

    async def record_evaluation(
        self, frame: RAGEvaluationFrame, metrics: Dict[str, Any]
    ) -> None:
        """Persist a full telemetry frame alongside calculated drift metrics."""
        conn, owned = await self._connection()
        try:
            await conn.execute(
                INSERT_EVALUATION_SQL,
                uuid.uuid4(),
                frame.trace_id,
                frame.metadata.rag_type,
                frame.timestamp,
                metrics.get("js_divergence"),
                metrics.get("mmd_score"),
                metrics.get("wasserstein_distance"),
                bool(metrics.get("is_drifted", False)),
                frame.model_dump_json(),
            )
        finally:
            await self._release(conn, owned)
        logger.info(
            "Evaluation frame %s (rag_type=%s) persisted.",
            frame.trace_id,
            frame.metadata.rag_type,
        )

    async def batch_store_frames(self, frames: List[RAGEvaluationFrame]) -> None:
        """Persist multiple raw telemetry frames in a single round trip.

        Used by the async ingestion buffer to batch-write ingested frames
        without blocking the HTTP ingestion endpoint.
        """
        if not frames:
            return
        conn, owned = await self._connection()
        try:
            await conn.executemany(
                INSERT_EVALUATION_SQL,
                [
                    (
                        uuid.uuid4(),
                        frame.trace_id,
                        frame.metadata.rag_type,
                        frame.timestamp,
                        None,
                        None,
                        None,
                        False,
                        frame.model_dump_json(),
                    )
                    for frame in frames
                ],
            )
        finally:
            await self._release(conn, owned)
        logger.info("Batch persisted %d evaluation frames.", len(frames))

    async def get_recent_frames(
        self, rag_type: Optional[str] = None, limit: int = 100
    ) -> List[RAGEvaluationFrame]:
        """Retrieve and deserialize historical evaluation frames."""
        conn, owned = await self._connection()
        try:
            if rag_type is not None:
                rows = await conn.fetch(
                    """
                    SELECT telemetry_frame
                    FROM telemetry_evaluations
                    WHERE rag_type = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                    """,
                    rag_type,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT telemetry_frame
                    FROM telemetry_evaluations
                    ORDER BY timestamp DESC
                    LIMIT $1
                    """,
                    limit,
                )
        finally:
            await self._release(conn, owned)

        frames: List[RAGEvaluationFrame] = []
        for row in rows:
            raw = row["telemetry_frame"]
            try:
                if isinstance(raw, str):
                    frames.append(RAGEvaluationFrame.model_validate_json(raw))
                else:
                    frames.append(RAGEvaluationFrame.model_validate(raw))
            except Exception as e:  # noqa: BLE001 - legacy/foreign payloads tolerated
                logger.warning(
                    "Skipping non-frame telemetry row during deserialization: %s", e
                )
        return frames

    async def get_frames_by_time_window(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[RAGEvaluationFrame]:
        """Fetch evaluation frames within a sliding *hours* window.

        Frames are returned oldest-first so they can be used for chronological
        analysis and sliding-baseline threshold calibration.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        conn, owned = await self._connection()
        try:
            rows = await conn.fetch(
                """
                    SELECT telemetry_frame
                    FROM telemetry_evaluations
                    WHERE timestamp >= $1
                    ORDER BY timestamp ASC
                    LIMIT $2
                    """,
                    cutoff,
                    limit,
                )
        finally:
            await self._release(conn, owned)

        frames: List[RAGEvaluationFrame] = []
        for row in rows:
            raw = row["telemetry_frame"]
            try:
                if isinstance(raw, str):
                    frames.append(RAGEvaluationFrame.model_validate_json(raw))
                else:
                    frames.append(RAGEvaluationFrame.model_validate(raw))
            except Exception as e:  # noqa: BLE001 - legacy/foreign payloads tolerated
                logger.warning(
                    "Skipping non-frame telemetry row during time-window retrieval: %s",
                    e,
                )
        return frames

    async def record_drift(self, drift_result: Dict[str, Any]) -> str:
        """Legacy adapter: persist a distribution-level drift computation.

        The full result dict is stored as the JSONB ``telemetry_frame`` so
        existing ``DriftMonitor.compute_comprehensive_drift`` callers keep
        working until they migrate onto ``record_evaluation``.
        """
        conn, owned = await self._connection()
        row_id = uuid.uuid4()
        try:
            await conn.execute(
                INSERT_EVALUATION_SQL,
                row_id,
                str(uuid.uuid4()),
                "custom",
                datetime.now(timezone.utc),
                drift_result.get("js_divergence"),
                drift_result.get("mmd_score"),
                drift_result.get("wasserstein_distance"),
                bool(drift_result.get("is_drifted", False)),
                json.dumps(drift_result),
            )
        finally:
            await self._release(conn, owned)
        logger.info(
            "Drift record persisted: JSD=%s",
            drift_result.get("js_divergence"),
        )
        return str(row_id)

    async def get_recent_history(self, hours: int = 24) -> pl.DataFrame:
        """Return recent evaluation rows as a Polars frame with legacy columns."""
        conn, owned = await self._connection()
        try:
            rows = await conn.fetch(
                """
                SELECT id, trace_id, rag_type, timestamp, js_divergence, mmd_score,
                       is_drifted, telemetry_frame
                FROM telemetry_evaluations
                WHERE timestamp >= $1
                ORDER BY timestamp
                """,
                datetime.now(timezone.utc) - timedelta(hours=hours),
            )
        finally:
            await self._release(conn, owned)

        data: Dict[str, List[Any]] = {
            "id": [],
            "trace_id": [],
            "rag_type": [],
            "timestamp": [],
            "jsd_score": [],
            "mmd_score": [],
            "mmd_p_value": [],
            "max_component_kl": [],
            "is_drifted": [],
            "sample_size": [],
            "metadata": [],
        }
        for row in rows:
            payload = _coerce_frame_payload(row["telemetry_frame"]) or {}
            data["id"].append(str(row["id"]))
            data["trace_id"].append(row["trace_id"])
            data["rag_type"].append(row["rag_type"])
            data["timestamp"].append(row["timestamp"])
            data["jsd_score"].append(row["js_divergence"])
            data["mmd_score"].append(row["mmd_score"])
            data["mmd_p_value"].append(payload.get("mmd_p_value"))
            data["max_component_kl"].append(payload.get("max_component_kl"))
            data["is_drifted"].append(bool(row["is_drifted"]))
            data["sample_size"].append(payload.get("sample_size", 0))
            data["metadata"].append(json.dumps(payload))
        return pl.DataFrame(data)

    async def get_store_stats(self) -> Dict[str, Any]:
        """Return high-level store statistics for the CLI diagnostics."""
        conn, owned = await self._connection()
        try:
            total = await conn.fetchval("SELECT COUNT(*) FROM telemetry_evaluations")
            by_type_rows = await conn.fetch(
                """
                SELECT rag_type, COUNT(*) AS count
                FROM telemetry_evaluations
                GROUP BY rag_type
                ORDER BY rag_type
                """
            )
            graph_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM telemetry_evaluations
                WHERE jsonb_typeof(telemetry_frame -> 'context' -> 'graph_topology')
                      = 'object'
                """
            )
            swarm_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM telemetry_evaluations
                WHERE jsonb_typeof(telemetry_frame -> 'metadata' -> 'agent_hops')
                      = 'array'
                """
            )
        finally:
            await self._release(conn, owned)

        return {
            "total_frames": int(total or 0),
            "by_rag_type": {row["rag_type"]: int(row["count"]) for row in by_type_rows},
            "frames_with_graph_payloads": int(graph_count or 0),
            "frames_with_swarm_metadata": int(swarm_count or 0),
            "status": "healthy",
        }

    async def get_latest_drift(self) -> Optional[Dict[str, Any]]:
        conn, owned = await self._connection()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, trace_id, rag_type, timestamp, js_divergence, mmd_score,
                       is_drifted, telemetry_frame
                FROM telemetry_evaluations
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )
        finally:
            await self._release(conn, owned)

        if row is None:
            return None
        payload = _coerce_frame_payload(row["telemetry_frame"]) or {}
        return {
            "id": str(row["id"]),
            "trace_id": row["trace_id"],
            "rag_type": row["rag_type"],
            "timestamp": row["timestamp"],
            "jsd_score": row["js_divergence"],
            "mmd_score": row["mmd_score"],
            "mmd_p_value": payload.get("mmd_p_value"),
            "max_component_kl": payload.get("max_component_kl"),
            "is_drifted": bool(row["is_drifted"]),
            "metadata": payload,
        }

    async def get_trend(self, window: int = 30) -> pl.DataFrame:
        conn, owned = await self._connection()
        try:
            rows = await conn.fetch(
                """
                SELECT timestamp, js_divergence
                FROM telemetry_evaluations
                ORDER BY timestamp
                """
            )
        finally:
            await self._release(conn, owned)

        if not rows:
            return pl.DataFrame({"timestamp": [], "jsd_score": []})

        df = pl.DataFrame(
            {
                "timestamp": [row["timestamp"] for row in rows],
                "jsd_score": [row["js_divergence"] for row in rows],
            }
        )

        if df.height < window:
            return df

        return df.with_columns(
            [
                pl.col("jsd_score")
                .rolling_mean(window_size=window)
                .alias("rolling_mean"),
                pl.col("jsd_score")
                .rolling_std(window_size=window)
                .alias("rolling_std"),
            ]
        )

    async def detect_anomaly(
        self, window: int = 30, std_multiplier: float = 2.0
    ) -> bool:
        trend = await self.get_trend(window=window)
        if trend.height < window:
            return False

        latest = trend["jsd_score"][-1]
        rolling_mean = trend["rolling_mean"][-1]
        rolling_std = trend["rolling_std"][-1]

        if rolling_mean is None or rolling_std is None or rolling_std == 0:
            return False

        is_anomaly = latest > rolling_mean + std_multiplier * rolling_std
        if is_anomaly:
            logger.warning(
                f"Anomaly detected: JSD={latest:.4f} > mean={rolling_mean:.4f} "
                f"+ {std_multiplier}*std={rolling_std:.4f}"
            )
        return is_anomaly

    async def clear_history(self) -> None:
        conn, owned = await self._connection()
        try:
            await conn.execute("TRUNCATE TABLE telemetry_evaluations")
        finally:
            await self._release(conn, owned)
        logger.info("Telemetry evaluation history cleared.")
