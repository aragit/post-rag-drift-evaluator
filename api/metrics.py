from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

FRAME_INGESTION_TOTAL = Counter(
    "frame_ingestion_total",
    "Total number of telemetry frames ingested via the HTTP API.",
    ["status", "buffer_type"],
)

INGESTION_BUFFER_DEPTH = Gauge(
    "ingestion_buffer_depth",
    "Current number of frames awaiting persistence in the ingestion buffer.",
    ["buffer_type"],
)

EVALUATION_LATENCY_SECONDS = Histogram(
    "evaluation_latency_seconds",
    "Duration of multi-modal drift evaluation requests in seconds.",
    ["status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

DRIFT_SCORE_GAUGE = Gauge(
    "drift_score_gauge",
    "Latest calculated drift metric value.",
    ["metric_type"],
)

DRIFT_ALERTS_TOTAL = Counter(
    "drift_alerts_total",
    "Total number of drift alerts dispatched (or attempted).",
    ["status"],
)

DB_BATCH_WRITE_LATENCY_SECONDS = Histogram(
    "db_batch_write_latency_seconds",
    "Latency of database batch write operations in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def render_metrics() -> bytes:
    """Return the prometheus-formatted metrics payload as UTF-8 bytes."""
    return generate_latest()


def get_content_type() -> str:
    """Convenience accessor for the prometheus content type header value."""
    return CONTENT_TYPE_LATEST
