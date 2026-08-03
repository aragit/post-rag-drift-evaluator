from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from evaluator.temporal.models import DriftEvent


def estimate_metric_after_intervention(
    drift_event: DriftEvent,
    modified_store: Any,
    metric_name: str,
    method: Literal["mean", "ewma", "trend_adjusted"] = "trend_adjusted",
    alpha: float = 0.3,
) -> float:
    """Estimate the metric value that would have existed after removing a change.

    Supports three estimation methods:

    - ``"mean"``: Unweighted arithmetic mean of pre-window metric values
      (legacy default fallback).
    - ``"ewma"``: Exponentially weighted moving average giving higher
      weights to metrics recorded closer to ``drift_event.start_timestamp``.
    - ``"trend_adjusted"``: Fits a linear slope (y = mx + c) on pre-window
      timestamps and evaluates the expected value at
      ``drift_event.start_timestamp``.  Falls back to ``"ewma"`` if fewer
      than 3 pre-window data points exist.

    If no pre-window records with the target metric exist, falls back to
    the mean of all available records.  If no records exist at all,
    returns ``0.0``.

    Args:
        drift_event: The drift event being simulated.
        modified_store: The store *after* intervention(s) have been applied.
        metric_name: Name of the metric to estimate.
        method: Estimation method — ``"mean"``, ``"ewma"``, or
            ``"trend_adjusted"`` (default).
        alpha: Smoothing factor for EWMA (default 0.3). Higher alpha
            gives more weight to recent observations.

    Returns:
        Estimated counterfactual metric value (float).
    """
    records = sorted(modified_store.load_all(), key=lambda r: r.timestamp or 0.0)

    pre_window_values: list[tuple[float, float]] = []
    all_values: list[float] = []

    for record in records:
        ts = record.timestamp or 0.0
        for metric in record.metrics:
            if metric.metric_name == metric_name:
                all_values.append(metric.value)
                if ts < drift_event.start_timestamp:
                    pre_window_values.append((ts, metric.value))

    if not pre_window_values:
        if all_values:
            return sum(all_values) / len(all_values)
        return 0.0

    if method == "mean":
        values = [v for _, v in pre_window_values]
        return sum(values) / len(values)

    if method == "ewma":
        return _ewma_estimate(pre_window_values, alpha)

    if method == "trend_adjusted":
        if len(pre_window_values) < 3:
            return _ewma_estimate(pre_window_values, alpha)
        return _trend_adjusted_estimate(pre_window_values, drift_event.start_timestamp)

    # Fallback
    values = [v for _, v in pre_window_values]
    return sum(values) / len(values)


def _ewma_estimate(
    pre_window_values: list[tuple[float, float]],
    alpha: float,
) -> float:
    """Exponentially weighted moving average, weighting by recency.

    Standard forward EWMA: most recent observation gets weight alpha.
        EWMA_0 = x_0
        EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}

    With alpha=1.0, returns the most recent value.
    """
    pre_window_values.sort(key=lambda x: x[0])
    values = [v for _, v in pre_window_values]

    if not values:
        return 0.0

    result = values[0]
    for val in values[1:]:
        result = alpha * val + (1 - alpha) * result

    return result


def _trend_adjusted_estimate(
    pre_window_values: list[tuple[float, float]],
    target_timestamp: float,
) -> float:
    """Fit linear regression on (timestamp, value) pairs and extrapolate.

    Uses least-squares OLS: y = m * t + c
    Returns m * target_timestamp + c.
    """
    timestamps = [ts for ts, _ in pre_window_values]
    values = [v for _, v in pre_window_values]

    n = len(timestamps)
    mean_t = sum(timestamps) / n
    mean_v = sum(values) / n

    # Compute slope m = sum((t - mean_t)(v - mean_v)) / sum((t - mean_t)^2)
    numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(timestamps, values))
    denominator = sum((t - mean_t) ** 2 for t in timestamps)

    if abs(denominator) < 1e-12:
        # All timestamps are the same — fall back to mean
        return mean_v

    m = numerator / denominator
    c = mean_v - m * mean_t

    return m * target_timestamp + c
