from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluator.causal.models import ChangeEvent

if TYPE_CHECKING:
    from evaluator.temporal.models import DriftEvent


_CHANGE_TYPE_WEIGHTS: dict[str, float] = {
    "model_update": 1.0,
    "version_change": 0.8,
    "config_change": 0.6,
    "unknown": 0.3,
}


def build_drift_features(
    drift_event: DriftEvent,
    changes: list[ChangeEvent],
) -> list[dict[str, Any]]:
    """Build feature vectors for each change relative to a drift window.

    For each change, computes:

    - ``time_delta``: seconds from the change to the drift window
      (0 if the change falls inside the window)
    - ``in_window``: True when the change timestamp is within
      ``[start_timestamp, end_timestamp]``
    - ``change_type``: the :class:`ChangeEvent` type string
    - ``drift_magnitude``: the ``magnitude`` from the :class:`DriftEvent`
    - ``change_type_weight``: heuristic weight for the change type
    """
    features: list[dict[str, Any]] = []
    window_start = drift_event.start_timestamp
    window_end = drift_event.end_timestamp

    for change in sorted(changes, key=lambda c: c.timestamp):
        in_window = window_start <= change.timestamp <= window_end

        if in_window:
            time_delta = 0.0
        else:
            time_delta = min(
                abs(change.timestamp - window_start),
                abs(change.timestamp - window_end),
            )

        features.append(
            {
                "change_id": change.change_id,
                "run_id": change.run_id,
                "change_type": change.change_type,
                "change_type_weight": _CHANGE_TYPE_WEIGHTS.get(
                    change.change_type, _CHANGE_TYPE_WEIGHTS["unknown"]
                ),
                "time_delta": time_delta,
                "in_window": in_window,
                "drift_magnitude": drift_event.magnitude,
                "details": dict(change.details),
            }
        )

    return features
