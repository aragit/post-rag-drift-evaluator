import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from evaluator.config import config

logger = logging.getLogger("DBPool")

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                logger.info("Creating asyncpg connection pool.")
                _pool = await asyncpg.create_pool(
                    dsn=config.DATABASE_URL,
                    min_size=config.DB_POOL_MIN_SIZE,
                    max_size=config.DB_POOL_MAX_SIZE,
                    max_queries=config.DB_POOL_MAX_QUERIES,
                )
                logger.info(
                    f"Connection pool created: min={config.DB_POOL_MIN_SIZE}, "
                    f"max={config.DB_POOL_MAX_SIZE}"
                )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed.")


async def acquire() -> asyncpg.Connection:
    pool = await get_pool()
    return await pool.acquire()


async def release(conn: asyncpg.Connection) -> None:
    pool = await get_pool()
    await pool.release(conn)


@asynccontextmanager
async def connection() -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pooled connection as an async context manager.

    The connection is always released when the block exits — even on error —
    replacing manual ``acquire()`` / ``release()`` call sites and preventing
    leaks under concurrent load.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn
