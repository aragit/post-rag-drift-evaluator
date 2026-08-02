import asyncio
import logging
import signal

from prometheus_client import Gauge, start_http_server

from evaluator.alerts import AlertManager
from evaluator.config import config
from evaluator.db.pool import close_pool, get_pool
from evaluator.drift_store import DriftStore

logger = logging.getLogger("Lifecycle")

shutdown_event = asyncio.Event()
_metrics_gauges: dict[str, Gauge] = {}


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(graceful_shutdown()))


async def graceful_shutdown() -> None:
    logger.info("Shutdown signal received. Initiating graceful shutdown...")
    shutdown_event.set()

    store = DriftStore()
    await store.close()
    logger.info("Drift store connection pool released.")

    await close_pool()
    logger.info("Database pool closed.")

    alert_mgr = AlertManager()
    alert_mgr._flush()
    logger.info("Alert queue flushed.")

    logger.info("Graceful shutdown complete.")


async def run_with_lifecycle(coro):
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)
    try:
        return await coro
    except asyncio.CancelledError:
        logger.info("Task cancelled during shutdown.")
    finally:
        await graceful_shutdown()


def start_metrics_server(port: int = config.METRICS_PORT) -> None:
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


def register_health_gauges() -> None:
    _metrics_gauges["pool_status"] = Gauge(
        "evaluator_pool_status",
        "Database pool connection status (1=connected, 0=disconnected)",
    )
    _metrics_gauges["drift_store_status"] = Gauge(
        "evaluator_drift_store_status",
        "PostgreSQL drift store availability (1=available, 0=unavailable)",
    )
    _metrics_gauges["last_drift_check"] = Gauge(
        "evaluator_last_drift_check_timestamp",
        "Unix timestamp of last drift check",
    )


def update_health_gauges(
    pool_connected: bool,
    postgres_available: bool,
    last_check_ts: float = 0.0,
) -> None:
    _metrics_gauges["pool_status"].set(1.0 if pool_connected else 0.0)
    _metrics_gauges["drift_store_status"].set(1.0 if postgres_available else 0.0)
    _metrics_gauges["last_drift_check"].set(last_check_ts)


async def check_health() -> None:
    """Probe PostgreSQL availability and publish the health gauges."""
    import time

    pool_connected = False
    try:
        pool = await get_pool()
        conn = await pool.acquire()
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await pool.release(conn)
        pool_connected = True
    except Exception as e:
        logger.error(f"PostgreSQL health probe failed: {e}")

    update_health_gauges(
        pool_connected=pool_connected,
        postgres_available=pool_connected,
        last_check_ts=time.time(),
    )


async def run_evaluator():
    import argparse

    from evaluator.benchmark import run_benchmark

    parser = argparse.ArgumentParser(description="Post-RAG Drift Evaluator")
    parser.add_argument("--queries", nargs="+", default=[])
    args = parser.parse_args()

    register_health_gauges()
    start_metrics_server()

    queries = args.queries or [
        "What are the strict physiological boundaries for patient eligibility?",
        "Explain the transaction state commit constraints.",
    ]

    await run_with_lifecycle(run_benchmark(queries))
