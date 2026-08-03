from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from evaluator.temporal.models import DriftEvent

if TYPE_CHECKING:
    from evaluator.storage import JSONHistoryStore


def detect_drift_events(
    series: list[tuple[float, float, str]],
    window_size: int = 3,
    threshold: float = 0.15,
    metric_name: str = "unknown",
) -> list[DriftEvent]:
    """Detect mean-shift drift events using a sliding window comparison.

    Algorithm:
        For each position ``i`` from ``window_size`` to
        ``len(series) - window_size`` (stepping by ``window_size``):

        1. ``previous_window = series[i - window_size : i]``
        2. ``current_window  = series[i : i + window_size]``
        3. Compute ``mean_prev`` and ``mean_curr``.
        4. If ``|mean_curr - mean_prev| > threshold`` → emit DriftEvent.

    Only full windows (exactly ``window_size`` points) are compared.
    """
    if len(series) < window_size * 2:
        return []

    events: list[DriftEvent] = []

    for i in range(window_size, len(series) - window_size + 1, window_size):
        prev_window = series[i - window_size : i]
        curr_window = series[i : i + window_size]

        if len(curr_window) < window_size:
            continue

        prev_values = [v for _, v, _ in prev_window]
        curr_values = [v for _, v, _ in curr_window]

        mean_prev = statistics.mean(prev_values)
        mean_curr = statistics.mean(curr_values)
        magnitude = abs(mean_curr - mean_prev)

        if magnitude > threshold:
            events.append(
                DriftEvent(
                    metric_name=metric_name,
                    start_timestamp=curr_window[0][0],
                    end_timestamp=curr_window[-1][0],
                    magnitude=magnitude,
                    involved_run_ids=[r for _, _, r in curr_window],
                    metadata={
                        "method": "mean_shift",
                        "window_size": window_size,
                        "threshold": threshold,
                        "mean_previous": mean_prev,
                        "mean_current": mean_curr,
                    },
                )
            )

    return events


def detect_drift_from_store(
    store: JSONHistoryStore,
    metric_name: str = "js_divergence",
    window_size: int = 3,
    threshold: float = 0.15,
) -> list[DriftEvent]:
    """Convenience: extract series from store, then detect drift events.

    1. Calls :func:`get_metric_series_with_runs` to load and sort.
    2. Calls :func:`detect_drift_events` on the resulting series.
    """
    from evaluator.temporal.series import get_metric_series_with_runs

    series = get_metric_series_with_runs(store, metric_name)
    return detect_drift_events(
        series,
        window_size=window_size,
        threshold=threshold,
        metric_name=metric_name,
    )
