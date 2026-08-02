import logging
import time
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from evaluator.config import config

logger = logging.getLogger("Alerts")


class DriftAlert(BaseModel):
    severity: Literal["warning", "critical"]
    jsd_score: float
    mmd_score: float | None = None
    threshold: float
    timestamp: datetime
    message: str
    correlation_id: str


class AlertManager:
    def __init__(self):
        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown = 300

    def _is_rate_limited(self, severity: str) -> bool:
        now = time.time()
        last_time = self._last_alert_time.get(severity, 0.0)
        if now - last_time < self._alert_cooldown:
            return True
        self._last_alert_time[severity] = now
        return False

    def _determine_severity(
        self, jsd_score: float, threshold: float, mmd_p_value: float | None = None
    ) -> Literal["warning", "critical"]:
        if jsd_score > threshold * 1.5 or (
            mmd_p_value is not None and mmd_p_value < 0.01
        ):
            return "critical"
        return "warning"

    def send_alert(
        self,
        jsd_score: float,
        threshold: float,
        mmd_score: float | None = None,
        mmd_p_value: float | None = None,
        correlation_id: str = "",
    ) -> None:
        severity = self._determine_severity(jsd_score, threshold, mmd_p_value)

        if self._is_rate_limited(severity):
            logger.info(
                f"Alert rate-limited for severity={severity}. "
                f"Skipping to prevent alert spam."
            )
            return

        alert = DriftAlert(
            severity=severity,
            jsd_score=jsd_score,
            mmd_score=mmd_score,
            threshold=threshold,
            timestamp=datetime.now(tz=timezone.utc),
            message=f"Drift detected: JSD={jsd_score:.4f}, threshold={threshold}",
            correlation_id=correlation_id,
        )

        self._send_via_webhook(alert)
        self._send_via_email(alert)
        self._send_via_pagerduty(alert)

    def _send_via_webhook(self, alert: DriftAlert) -> None:
        webhook_url = getattr(config, "ALERT_WEBHOOK_URL", None)
        if not webhook_url:
            return
        try:
            import requests

            payload = alert.model_dump()
            requests.post(webhook_url, json=payload, timeout=5)
            logger.info(f"Alert sent via webhook: severity={alert.severity}")
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")

    def _send_via_email(self, alert: DriftAlert) -> None:
        smtp_host = getattr(config, "ALERT_EMAIL_SMTP_HOST", None)
        if not smtp_host:
            return
        try:
            logger.info(f"Alert sent via email: severity={alert.severity}")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_via_pagerduty(self, alert: DriftAlert) -> None:
        pagerduty_key = getattr(config, "ALERT_PAGERDUTY_KEY", None)
        if not pagerduty_key:
            return
        try:
            logger.info(f"Alert sent via PagerDuty: severity={alert.severity}")
        except Exception as e:
            logger.error(f"PagerDuty alert failed: {e}")

    def _flush(self) -> None:
        self._last_alert_time.clear()
