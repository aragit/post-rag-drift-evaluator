import asyncio

import asyncpg
import pytest
import redis

from evaluator.config import config
from evaluator.db import pool as db_pool


@pytest.fixture(autouse=True)
async def reset_db_pool():
    """Reset the module-level asyncpg pool and lock before each test."""
    if db_pool._pool is not None:
        try:
            await db_pool._pool.close()
        except Exception:
            pass
    db_pool._pool = None
    db_pool._pool_lock = asyncio.Lock()
    yield


@pytest.fixture(autouse=True)
def flush_redis():
    try:
        r = redis.from_url("redis://localhost:6379/0")
        r.flushdb()
    except Exception:
        pass


@pytest.fixture(scope="session")
def postgres_available() -> bool:
    """Whether a reachable PostgreSQL instance matches the configured URL."""

    async def _probe() -> bool:
        conn = await asyncpg.connect(config.DATABASE_URL, timeout=3)
        await conn.close()
        return True

    try:
        return asyncio.run(_probe())
    except Exception:
        return False
