import logging
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from evaluator.config import config

logger = logging.getLogger("MetricsExporter")

rag_drift_jsd_score = Gauge(
    "rag_drift_jsd_score",
    "Current Jensen-Shannon Divergence score",
)
rag_drift_mmd_score = Gauge(
    "rag_drift_mmd_score",
    "Current Maximum Mean Discrepancy score",
)
rag_drift_is_detected = Gauge(
    "rag_drift_is_detected",
    "1 if drift is detected, 0 if stable",
)
rag_pipeline_latency_seconds = Histogram(
    "rag_pipeline_latency_seconds",
    "Per-pipeline latency in seconds",
    ["pipeline_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
rag_pipeline_tokens_total = Counter(
    "rag_pipeline_tokens_total",
    "Total tokens consumed per pipeline",
    ["pipeline_name"],
)
rag_pipeline_errors_total = Counter(
    "rag_pipeline_errors_total",
    "Total errors per pipeline",
    ["pipeline_name", "error_type"],
)
rag_db_pool_connections = Gauge(
    "rag_db_pool_connections",
    "Current database pool size",
)
rag_cache_hit_total = Counter(
    "rag_cache_hit_total",
    "Total cache hits",
    ["cache_type"],
)
rag_cache_miss_total = Counter(
    "rag_cache_miss_total",
    "Total cache misses",
    ["cache_type"],
)

METRICS_PORT = getattr(config, "METRICS_PORT", 8000)


def start_metrics_server() -> None:
    try:
        start_http_server(METRICS_PORT)
        logger.info(f"Prometheus metrics server started on port {METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")


def record_drift_metrics(result: dict) -> None:
    rag_drift_jsd_score.set(result.get("js_divergence", 0.0))
    rag_drift_mmd_score.set(result.get("mmd_score", 0.0))
    rag_drift_is_detected.set(1.0 if result.get("is_drifted") else 0.0)


def record_pipeline_latency(pipeline_name: str, latency_seconds: float) -> None:
    rag_pipeline_latency_seconds.labels(pipeline_name=pipeline_name).observe(
        latency_seconds
    )


def record_pipeline_tokens(pipeline_name: str, tokens: int) -> None:
    rag_pipeline_tokens_total.labels(pipeline_name=pipeline_name).inc(tokens)


def record_pipeline_error(pipeline_name: str, error_type: str) -> None:
    rag_pipeline_errors_total.labels(
        pipeline_name=pipeline_name, error_type=error_type
    ).inc()


def record_db_pool_size(size: int) -> None:
    rag_db_pool_connections.set(float(size))


def record_cache_hit(cache_type: str) -> None:
    rag_cache_hit_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    rag_cache_miss_total.labels(cache_type=cache_type).inc()
