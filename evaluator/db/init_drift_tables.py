import asyncio
import logging
from typing import Optional

import asyncpg

from evaluator.db import pool as db_pool

logger = logging.getLogger("DriftSchema")

CREATE_TELEMETRY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_evaluations (
    id UUID PRIMARY KEY,
    trace_id VARCHAR(64) NOT NULL,
    rag_type VARCHAR(32) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    js_divergence FLOAT,
    mmd_score FLOAT,
    wasserstein_distance FLOAT,
    is_drifted BOOLEAN DEFAULT FALSE,
    telemetry_frame JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_telemetry_trace_id ON telemetry_evaluations(trace_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_rag_type ON telemetry_evaluations(rag_type);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_evaluations(timestamp);
"""


async def init_drift_tables(pool: Optional[asyncpg.Pool] = None) -> None:
    """Ensure the telemetry evaluation schema exists in PostgreSQL.

    Uses the shared asyncpg pool from ``evaluator.db.pool`` unless an
    explicit pool (e.g. a UI-owned, loop-local pool) is supplied.
    """
    if pool is None:
        pool = await db_pool.get_pool()
    conn = await pool.acquire()
    try:
        await conn.execute(CREATE_TELEMETRY_TABLE_SQL)
        logger.info("telemetry_evaluations schema ensured.")
    finally:
        await pool.release(conn)


if __name__ == "__main__":
    asyncio.run(init_drift_tables())
