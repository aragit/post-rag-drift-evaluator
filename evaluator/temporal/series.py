from __future__ import annotations

from typing import Any

from evaluator.storage import JSONHistoryStore


def get_metric_series(
    store: JSONHistoryStore,
    metric_name: str,
) -> list[tuple[float, float]]:
    """Extract a sorted time series for a single metric.

    Returns:
        ``[(timestamp, value), ...]`` sorted by timestamp ascending.
    """
    series_with_runs = get_metric_series_with_runs(store, metric_name)
    return [(ts, val) for ts, val, _ in series_with_runs]


def get_metric_series_with_runs(
    store: JSONHistoryStore,
    metric_name: str,
) -> list[tuple[float, float, str]]:
    """Extract a sorted time series with run IDs.

    Returns:
        ``[(timestamp, value, run_id), ...]`` sorted by timestamp ascending.
    """
    records = store.load_all()

    points: list[tuple[float, float, str]] = []
    for record in records:
        for metric in record.metrics:
            if metric.metric_name == metric_name:
                ts = record.timestamp or 0.0
                run_id = _extract_run_id(record, metric)
                points.append((ts, float(metric.value), run_id))

    points.sort(key=lambda p: p[0])
    return points


def _extract_run_id(record: Any, metric: Any) -> str:
    """Best-effort extraction of run_id from a record + metric."""
    if hasattr(metric, "run_id") and metric.run_id is not None:
        return metric.run_id
    if hasattr(metric, "current_run_id") and metric.current_run_id is not None:
        return metric.current_run_id
    if hasattr(record, "run_id") and record.run_id is not None:
        return record.run_id
    return ""
