from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluator.storage import JSONHistoryStore
    from evaluator.temporal.models import DriftEvent


def estimate_metric_after_intervention(
    drift_event: DriftEvent,
    modified_store: JSONHistoryStore,
    metric_name: str,
) -> float:
    """Estimate the metric value that would have existed after removing a change.

    Uses a **deterministic, history-based baseline**: after the intervention
    reverts the system to its pre-change state, the metric in the drift
    window is estimated as the mean of all metric values observed *before*
    the drift window (``timestamp < start_timestamp``).

    If no pre-window records with the target metric exist, falls back to
    the mean of *all* available records.  If no records exist at all,
    returns ``0.0``.

    Args:
        drift_event: The drift event being simulated.
        modified_store: The store *after* intervention(s) have been applied.
        metric_name: Name of the metric to estimate (e.g. ``"js_divergence"``).

    Returns:
        Estimated counterfactual metric value (float).
    """
    records = sorted(modified_store.load_all(), key=lambda r: r.timestamp or 0.0)

    pre_window_values: list[float] = []
    all_values: list[float] = []

    for record in records:
        ts = record.timestamp or 0.0
        for metric in record.metrics:
            if metric.metric_name == metric_name:
                all_values.append(metric.value)
                if ts < drift_event.start_timestamp:
                    pre_window_values.append(metric.value)

    if pre_window_values:
        return sum(pre_window_values) / len(pre_window_values)

    if all_values:
        return sum(all_values) / len(all_values)

    return 0.0
