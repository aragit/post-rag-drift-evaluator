"""Real-time drift alerting via configurable webhook channels.

``DriftAlertNotifier`` formats a structured JSON payload whenever a drift
evaluation reports ``is_drifted: True`` and dispatches it to a Slack or
generic HTTP webhook without blocking the evaluation caller.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from evaluator.config import config
from evaluator.logging_config import get_logger

logger = get_logger("DriftAlertNotifier")

ALERT_TIMEOUT_SECONDS = 5.0


def _drift_alerts_total():
    """Lazy resolver for the Prometheus counter to avoid circular imports."""
    from api.metrics import DRIFT_ALERTS_TOTAL
    return DRIFT_ALERTS_TOTAL


class DriftAlertNotifier:
    """Dispatch structured drift alerts to a configured webhook URL."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = (
            webhook_url
            or os.environ.get("DRIFT_ALERT_WEBHOOK_URL")
            or config.DRIFT_ALERT_WEBHOOK_URL
        )

    def build_payload(
        self, eval_result: Dict[str, Any], batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the structured JSON alert payload from an evaluation result."""
        vector = eval_result.get("vector_drift", {})
        graph = eval_result.get("graph_drift", {})
        swarm = eval_result.get("swarm_drift", {})
        return {
            "event": "drift_alert",
            "is_drifted": bool(eval_result.get("is_drifted", False)),
            "batch_id": batch_id,
            "vector_drift": {
                "js_divergence": vector.get("js_divergence", 0.0),
                "mmd_score": vector.get("mmd_score", 0.0),
            },
            "graph_drift": {
                "spectral_distance": graph.get("spectral_distance", 0.0),
                "density_delta": graph.get("density_delta", 0.0),
            },
            "swarm_drift": {
                "transition_entropy_delta": swarm.get("transition_entropy_delta", 0.0),
                "avg_reflection_iterations_delta": swarm.get(
                    "avg_reflection_iterations_delta", 0.0
                ),
            },
        }

    async def notify_if_drifted(
        self, eval_result: Dict[str, Any], batch_id: Optional[str] = None
    ) -> bool:
        """Dispatch an alert when ``eval_result["is_drifted"]`` is true.

        Returns ``True`` only when an alert was triggered and the HTTP
        dispatch succeeded; otherwise returns ``False`` without raising.
        """
        if not eval_result.get("is_drifted"):
            return False

        if not self.webhook_url:
            logger.warning("Drift detected but no webhook configured; skipping alert.")
            return False

        payload = self.build_payload(eval_result, batch_id)
        try:
            async with httpx.AsyncClient(timeout=ALERT_TIMEOUT_SECONDS) as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - alerting must not break evaluation
            _drift_alerts_total().labels(status="failed").inc()
            logger.error("Alert dispatch to %s failed: %s", self.webhook_url, exc)
            return False

        logger.info(
            "Drift alert dispatched to %s (HTTP %s).",
            self.webhook_url,
            response.status_code,
        )
        _drift_alerts_total().labels(status="dispatched").inc()
        return True
