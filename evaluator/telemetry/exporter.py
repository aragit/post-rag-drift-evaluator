"""Telemetry & metrics exporter using OpenTelemetry conventions.

Provides :class:`SentrixMetricsExporter` which exports:

- **Gauges**: latent drift scores per track/metric, estimated impact deltas
- **Counters**: total drift events by severity, total optimization actions
  by status

Falls back to a no-op meter if OpenTelemetry is not installed.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_OTEL_ENABLED = True


class _NoopMeter:
    """Fallback meter when OTel is not installed."""

    def create_observable_gauge(self, name: str, callback):
        return _NoopInstrument()

    def create_observable_counter(self, name: str, callback):
        return _NoopInstrument()

    def create_counter(self, name: str):
        return _NoopInstrument()

    def create_histogram(self, name: str):
        return _NoopInstrument()

    def create_gauge(self, name: str):
        return _NoopInstrument()


class _NoopInstrument:
    """No-op instrument that silently ignores all updates."""

    def add(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass


class SentrixMetricsExporter:
    """Exports Sentrix metrics via OpenTelemetry (with graceful fallback).

    Metrics exposed:

    Gauges (observable):
      - ``sentrix_latent_drift_score{track, metric}``
      - ``sentrix_estimated_impact_delta{metric_name}``

    Counters:
      - ``sentrix_drift_events_total{severity}``
      - ``sentrix_optimization_actions_total{status, rule_violated}``
      - ``sentrix_counterfactual_evaluations_total``

    Args:
        service_name: Service identifier for OTel resource attributes.
        otel_enabled: Whether to attempt OTel backend connection.
    """

    def __init__(
        self,
        service_name: str = "sentrix-evaluator",
        otel_enabled: bool = _DEFAULT_OTEL_ENABLED,
    ):
        self.service_name = service_name
        self.otel_enabled = otel_enabled
        self._metrics: dict[str, float] = {}
        self._counters: dict[str, int] = {}
        self._meter = self._init_meter()

    def _init_meter(self) -> Any:
        """Initialize the OTel meter, falling back to no-op if unavailable."""
        if not self.otel_enabled:
            return _NoopMeter()

        try:
            from importlib import util

            if not util.find_spec("opentelemetry"):
                return _NoopMeter()
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(provider)
            return metrics.get_meter("sentrix-evaluator")
        except ImportError:
            return _NoopMeter()

    def record_latent_drift_score(
        self,
        score: float,
        track: str = "unified",
        metric: str = "mmd",
    ) -> None:
        """Record a latent drift score gauge value.

        Args:
            score: The drift score (0–1).
            track: Embedding track ("retrieval", "generation", "unified").
            metric: Distance metric ("mmd", "swd", "jsd").
        """
        key = f"drift_score:{track}:{metric}"
        self._metrics[key] = score

    def record_estimated_impact(
        self,
        delta: float,
        metric_name: str = "js_divergence",
    ) -> None:
        """Record an estimated impact delta gauge value.

        Args:
            delta: The counterfactual delta (improvement estimate).
            metric_name: The metric being improved.
        """
        key = f"estimated_impact:{metric_name}"
        self._metrics[key] = delta

    def increment_drift_events(self, severity: str = "high") -> None:
        """Increment the drift events counter.

        Args:
            severity: Drift severity category ("low", "medium", "high").
        """
        key = f"drift_events:{severity}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def increment_optimization_actions(
        self,
        status: str = "approved",
        rule_violated: str | None = None,
    ) -> None:
        """Increment the optimization actions counter.

        Args:
            status: Action status ("approved", "blocked", "no_action").
            rule_violated: Which guardrail rule was violated (if blocked).
        """
        key = f"optimization_actions:{status}:{rule_violated or 'none'}"
        self._counters[key] = self._counters.get(key, 0) + 1

    def increment_counterfactual_evaluations(self) -> None:
        """Increment the counterfactual evaluation counter."""
        key = "counterfactual_evaluations"
        self._counters[key] = self._counters.get(key, 0) + 1

    def get_metrics_snapshot(self) -> dict[str, float]:
        """Return a snapshot of all recorded gauge metrics."""
        return dict(self._metrics)

    def get_counters_snapshot(self) -> dict[str, int]:
        """Return a snapshot of all recorded counters."""
        return dict(self._counters)

    def reset(self) -> None:
        """Reset all metrics and counters (useful for testing)."""
        self._metrics.clear()
        self._counters.clear()
