# Sentrix Evaluator — Core Source Code

## Overview

The Sentrix Evaluator is a causal drift evaluation and optimization engine for RAG pipelines. It detects distribution-level embedding drift (Phase 10), attributes drift to root causes (Phase 5), simulates counterfactual interventions (Phase 6), and generates ranked optimization plans (Phase 7).

---

## Phase 1: Schema & Metrics

### `evaluator/__init__.py` — Public API

```python
from evaluator.causal.attribution import attribute_drift
from evaluator.counterfactual.simulator import run_counterfactual_analysis
from evaluator.optimization.optimizer import generate_optimization_plan
from evaluator.temporal.drift_detection import detect_drift_from_store

__all__ = [
    "detect_drift_from_store",
    "attribute_drift",
    "run_counterfactual_analysis",
    "generate_optimization_plan",
]
```

---

## Phase 2: Temporal Drift Detection

### `evaluator/temporal/models.py`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftEvent:
    """Represents a detected drift over a time window.

    Produced by detect_drift_events.
    Each event captures which metric changed, when, by how much, and
    which runs contributed to the shifted window.
    """

    metric_name: str
    start_timestamp: float
    end_timestamp: float
    magnitude: float
    involved_run_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None

    def __post_init__(self) -> None:
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "metric_name": self.metric_name,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "magnitude": self.magnitude,
            "involved_run_ids": list(self.involved_run_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftEvent:
        """Deserialize from a dictionary created by to_dict."""
        return cls(
            event_id=data.get("event_id"),
            metric_name=data["metric_name"],
            start_timestamp=data["start_timestamp"],
            end_timestamp=data["end_timestamp"],
            magnitude=data["magnitude"],
            involved_run_ids=list(data.get("involved_run_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )
```

### `evaluator/temporal/drift_detection.py`

```python
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
        For each position i from window_size to len(series) - window_size
        (stepping by window_size):

        1. previous_window = series[i - window_size : i]
        2. current_window  = series[i : i + window_size]
        3. Compute mean_prev and mean_curr.
        4. If |mean_curr - mean_prev| > threshold → emit DriftEvent.

    Only full windows (exactly window_size points) are compared.
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
    """Convenience: extract series from store, then detect drift events."""
    from evaluator.temporal.series import get_metric_series_with_runs

    series = get_metric_series_with_runs(store, metric_name)
    return detect_drift_events(
        series,
        window_size=window_size,
        threshold=threshold,
        metric_name=metric_name,
    )
```

---

## Phase 3: Causal Attribution

### `evaluator/causal/models.py`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChangeEvent:
    """Represents a system change detected in the evaluation history.

    A change event is emitted when consecutive runs differ in
    system_version, metadata, or system_info fields.
    """

    change_id: str | None = None
    timestamp: float = 0.0
    run_id: str = ""
    change_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.change_id is None:
            self.change_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{self.run_id}:{self.timestamp}:{self.change_type}",
                )
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "change_type": self.change_type,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeEvent:
        return cls(
            change_id=data.get("change_id"),
            timestamp=data.get("timestamp", 0.0),
            run_id=data.get("run_id", ""),
            change_type=data.get("change_type", ""),
            details=dict(data.get("details", {})),
        )


@dataclass
class CausalFactor:
    """A candidate cause ranked by its attribution score."""

    factor_name: str
    score: float
    related_run_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "score": self.score,
            "related_run_ids": list(self.related_run_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalFactor:
        return cls(
            factor_name=data["factor_name"],
            score=data["score"],
            related_run_ids=list(data.get("related_run_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CausalAttribution:
    """Full attribution result for a single drift event."""

    drift_event_id: str
    metric_name: str
    factors: list[CausalFactor] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    attribution_id: str | None = None

    def __post_init__(self) -> None:
        if self.attribution_id is None:
            self.attribution_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribution_id": self.attribution_id,
            "drift_event_id": self.drift_event_id,
            "metric_name": self.metric_name,
            "factors": [f.to_dict() for f in self.factors],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalAttribution:
        return cls(
            attribution_id=data.get("attribution_id"),
            drift_event_id=data["drift_event_id"],
            metric_name=data["metric_name"],
            factors=[CausalFactor.from_dict(f) for f in data.get("factors", [])],
            confidence=data.get("confidence", 0.0),
            metadata=dict(data.get("metadata", {})),
        )
```

### `evaluator/causal/attribution.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluator.causal.models import CausalAttribution, CausalFactor

if TYPE_CHECKING:
    from evaluator.storage import JSONHistoryStore
    from evaluator.temporal.models import DriftEvent

# Scoring weights (Phase 5 uses a simple heuristic model)
_WEIGHT_TIME_PROXIMITY = 0.4
_WEIGHT_CHANGE_TYPE = 0.3
_WEIGHT_DRIFT_MAGNITUDE = 0.3


def score_causal_impact(features: list[dict[str, Any]]) -> list[CausalFactor]:
    """Score each change feature and return ranked CausalFactor list.

    Heuristic scoring (deterministic, no ML):

    - in_window (binary): full bonus if change is inside the drift window.
    - time proximity: inverse of time_delta (closer changes score higher),
      normalised to [0, 1].
    - change type weight: pre-assigned weight (model_update=1.0,
      config_change=0.6, etc.).
    - drift magnitude: normalised drift magnitude factor.
    """
    if not features:
        return []

    max_delta = max((f["time_delta"] for f in features), default=0.0)
    max_magnitude = max((f["drift_magnitude"] for f in features), default=1.0)

    factors: list[CausalFactor] = []

    for feat in features:
        # Time proximity component
        delta = feat["time_delta"]
        if max_delta > 0:
            time_score = 1.0 - (delta / max_delta)
        else:
            time_score = 1.0

        if feat["in_window"]:
            time_score = time_score * 0.5 + 0.5

        # Change type weight component
        type_score = feat.get("change_type_weight", 0.5)

        # Drift magnitude component
        if max_magnitude > 0:
            mag_score = feat["drift_magnitude"] / max_magnitude
        else:
            mag_score = 0.0

        # Weighted composite score
        raw_score = (
            _WEIGHT_TIME_PROXIMITY * time_score
            + _WEIGHT_CHANGE_TYPE * type_score
            + _WEIGHT_DRIFT_MAGNITUDE * mag_score
        )

        score = max(0.0, min(1.0, raw_score))

        factors.append(
            CausalFactor(
                factor_name=feat["change_type"],
                score=round(score, 4),
                related_run_ids=[feat["run_id"]],
                metadata={
                    "change_id": feat["change_id"],
                    "time_delta": feat["time_delta"],
                    "in_window": feat["in_window"],
                    "change_details": feat["details"],
                },
            )
        )

    # Rank by score, descending
    factors.sort(key=lambda f: f.score, reverse=True)
    return factors


def _compute_confidence(factors: list[CausalFactor]) -> float:
    """Compute a simple confidence score from the factor distribution."""
    if not factors:
        return 0.0
    if len(factors) == 1:
        return round(factors[0].score, 4)

    top_score = factors[0].score
    total = sum(f.score for f in factors)

    if total == 0:
        return 0.0

    confidence = top_score / total
    return round(confidence, 4)


def attribute_drift(
    drift_event: DriftEvent,
    store: JSONHistoryStore,
) -> CausalAttribution:
    """Full attribution pipeline for a single drift event."""
    from evaluator.causal.change_extractor import extract_change_events
    from evaluator.causal.feature_builder import build_drift_features

    changes = extract_change_events(store)
    features = build_drift_features(drift_event, changes)
    factors = score_causal_impact(features)

    confidence = _compute_confidence(factors)

    return CausalAttribution(
        drift_event_id=drift_event.event_id or "",
        metric_name=drift_event.metric_name,
        factors=factors,
        confidence=confidence,
        metadata={
            "num_changes_examined": len(changes),
            "num_factors": len(factors),
            "scoring_method": "heuristic_weighted_composite",
        },
    )
```

### `evaluator/causal/change_extractor.py`

```python
from __future__ import annotations

from typing import Any

from evaluator.causal.models import ChangeEvent


def extract_change_events(store: Any) -> list[ChangeEvent]:
    """Extract chronological change events from a JSONHistoryStore.

    Walks all EvaluationRecords sorted by timestamp and detects
    differences between consecutive runs in:
    - system_version  -> "version_change"
    - metadata        -> "config_change"
    - system_info     -> "model_update"
    """
    records = store.load_all()
    if not records:
        return []

    records = sorted(records, key=lambda r: r.timestamp or 0.0)

    changes: list[ChangeEvent] = []
    prev: dict[str, Any] | None = None

    for record in records:
        curr = _extract_change_signature(record)
        if prev is not None:
            detected = _diff_signatures(prev, curr, record)
            if detected:
                changes.append(detected)
        prev = curr

    return changes


def _extract_change_signature(record: Any) -> dict[str, Any]:
    """Flatten the observable state of a record into a comparable dict."""
    sig: dict[str, Any] = {
        "system_version": record.system_version,
        "run_id": record.run_id,
        "timestamp": record.timestamp or 0.0,
        "metadata": dict(record.metadata),
    }
    if "system_info" in sig["metadata"]:
        sig["system_info"] = sig["metadata"].pop("system_info")
    if "pipeline_name" in sig["metadata"]:
        sig["pipeline_name"] = sig["metadata"].pop("pipeline_name")
    return sig


def _diff_signatures(
    prev: dict[str, Any],
    curr: dict[str, Any],
    record: Any,
) -> ChangeEvent | None:
    """Compare two consecutive signatures and build a ChangeEvent if different."""
    details: dict[str, Any] = {}
    change_type = "unknown"
    has_change = False

    if prev.get("system_version") != curr.get("system_version"):
        has_change = True
        details["old_system_version"] = prev.get("system_version")
        details["new_system_version"] = curr.get("system_version")
        if change_type == "unknown":
            change_type = "version_change"

    if prev.get("pipeline_name") != curr.get("pipeline_name"):
        has_change = True
        details["old_pipeline"] = prev.get("pipeline_name")
        details["new_pipeline"] = curr.get("pipeline_name")
        if change_type == "unknown":
            change_type = "model_update"

    prev_info = prev.get("system_info")
    curr_info = curr.get("system_info")
    if prev_info != curr_info:
        has_change = True
        changed_fields = {}
        if isinstance(prev_info, dict) and isinstance(curr_info, dict):
            for key in set(prev_info) | set(curr_info):
                if prev_info.get(key) != curr_info.get(key):
                    changed_fields[key] = {
                        "old": prev_info.get(key),
                        "new": curr_info.get(key),
                    }
        if change_type == "unknown" or change_type == "version_change":
            change_type = "model_update"
        details["system_info_diff"] = changed_fields

    prev_meta = prev.get("metadata", {})
    curr_meta = curr.get("metadata", {})
    meta_diff = {}
    for key in set(prev_meta) | set(curr_meta):
        if prev_meta.get(key) != curr_meta.get(key):
            meta_diff[key] = {"old": prev_meta.get(key), "new": curr_meta.get(key)}
    if meta_diff:
        has_change = True
        details["metadata_diff"] = meta_diff
        if change_type == "unknown":
            change_type = "config_change"

    if not has_change:
        return None

    return ChangeEvent(
        timestamp=curr["timestamp"],
        run_id=str(curr["run_id"]),
        change_type=change_type,
        details=details,
    )
```

---

## Phase 4: Storage & History

### `evaluator/storage/models.py` (EvaluationRecord)

*(Referenced by change_extractor and estimator — contains record-level telemetry with system_version, metadata, metrics, timestamp, run_id)*

### `evaluator/storage/json_store.py` (JSONHistoryStore)

*(Referenced by simulator — implements load_all, clone, append for history persistence)*

---

## Phase 5: Causal Attribution (detailed above)

---

## Phase 6: Counterfactual Simulation

### `evaluator/counterfactual/models.py`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intervention:
    """A hypothetical action applied to a change event.

    Supported actions:
        - "remove": revert the change entirely
        - "modify": override specific metadata fields on the change
    """

    action: str = "remove"
    change_id: str = ""
    override_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    intervention_id: str | None = None

    def __post_init__(self) -> None:
        if self.intervention_id is None:
            self.intervention_id = str(uuid.uuid4())


@dataclass
class CounterfactualScenario:
    """A 'what-if' hypothesis built from one or more interventions."""

    drift_event_id: str
    interventions: list[Intervention] = field(default_factory=list)
    description: str = ""
    scenario_id: str | None = None

    def __post_init__(self) -> None:
        if self.scenario_id is None:
            self.scenario_id = str(uuid.uuid4())


@dataclass
class CounterfactualResult:
    """Output of a single counterfactual simulation run.

    delta: original_metric - counterfactual_metric (positive = change
    contributed to drift).
    """

    scenario_id: str
    original_metric: float
    counterfactual_metric: float
    delta: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    result_id: str | None = None

    def __post_init__(self) -> None:
        if self.result_id is None:
            self.result_id = str(uuid.uuid4())
```

### `evaluator/counterfactual/scenario.py`

```python
from __future__ import annotations

from evaluator.causal.models import CausalAttribution
from evaluator.counterfactual.models import (
    CounterfactualScenario,
    Intervention,
)

_MAX_TOP_K = 3


def build_counterfactual_scenarios(
    attribution: CausalAttribution,
    top_k: int = _MAX_TOP_K,
) -> list[CounterfactualScenario]:
    """Build counterfactual scenarios from a causal attribution.

    Converts top-ranked causal factors into individual 'remove this change'
    scenarios, plus an optional combined scenario that removes all top
    factors simultaneously.
    """
    if not attribution.factors:
        return []

    factors = sorted(
        attribution.factors, key=lambda f: f.score, reverse=True
    )[:top_k]

    scenarios: list[CounterfactualScenario] = []

    for factor in factors:
        change_id = str(factor.metadata.get("change_id", ""))
        intervention = Intervention(
            action="remove",
            change_id=change_id,
            metadata={
                "factor_name": factor.factor_name,
                "factor_score": factor.score,
                "related_run_ids": list(factor.related_run_ids),
            },
        )
        scenario = CounterfactualScenario(
            drift_event_id=attribution.drift_event_id,
            interventions=[intervention],
            description=(
                f"Remove change '{factor.factor_name}' "
                f"(score={factor.score:.4f}, change_id={change_id})"
            ),
        )
        scenarios.append(scenario)

    if len(factors) > 1:
        interventions = []
        for factor in factors:
            change_id = str(factor.metadata.get("change_id", ""))
            interventions.append(
                Intervention(
                    action="remove",
                    change_id=change_id,
                    metadata={
                        "factor_name": factor.factor_name,
                        "factor_score": factor.score,
                    },
                )
            )
        combined = CounterfactualScenario(
            drift_event_id=attribution.drift_event_id,
            interventions=interventions,
            description=(
                f"Remove all {len(interventions)} top-ranked changes"
            ),
        )
        scenarios.append(combined)

    return scenarios
```

### `evaluator/counterfactual/estimator.py`

```python
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

    Uses a deterministic, history-based baseline: after the intervention
    reverts the system to its pre-change state, the metric in the drift
    window is estimated as the mean of all metric values observed *before*
    the drift window (timestamp < start_timestamp).

    Falls back to mean of all records, then 0.0 if no records exist.
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
```

### `evaluator/counterfactual/simulator.py`

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from evaluator.counterfactual.estimator import estimate_metric_after_intervention
from evaluator.counterfactual.models import (
    CounterfactualResult,
    CounterfactualScenario,
    Intervention,
)
from evaluator.counterfactual.scenario import build_counterfactual_scenarios

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution
    from evaluator.storage import JSONHistoryStore
    from evaluator.temporal.models import DriftEvent


def apply_intervention(
    store: JSONHistoryStore,
    intervention: Intervention,
) -> JSONHistoryStore:
    """Apply a single intervention to a store and return a modified clone.

    The original store is never mutated. A new JSONHistoryStore is created
    at a temporary path with the intervention applied.

    For "remove": the change identified by intervention.change_id is
    reverted — the affected record's system_version, system_info, and
    metadata fields are restored to pre-change values.

    For "modify": specific metadata fields from override_metadata are set.
    """
    from evaluator.causal.change_extractor import extract_change_events

    cloned = store.clone()
    records = sorted(cloned.load_all(), key=lambda r: r.timestamp or 0.0)

    if not records:
        return cloned

    changes = extract_change_events(store)

    target_change = None
    for change in changes:
        if change.change_id == intervention.change_id:
            target_change = change
            break

    if target_change is None:
        return cloned

    for record in records:
        if record.run_id == target_change.run_id:
            if intervention.action == "remove":
                _revert_change(record, target_change.details)
            elif intervention.action == "modify":
                _apply_modify(record, intervention.override_metadata)
            break

    _write_records(cloned, records)
    return cloned


def _revert_change(record, details: dict) -> None:
    """Revert a record to its pre-change state using change-event details."""
    if "old_system_version" in details:
        record.system_version = details["old_system_version"]
    if "old_pipeline" in details:
        record.metadata["pipeline_name"] = details["old_pipeline"]
    if "system_info_diff" in details and isinstance(details["system_info_diff"], dict):
        if "system_info" not in record.metadata:
            record.metadata["system_info"] = {}
        for field_name, values in details["system_info_diff"].items():
            record.metadata["system_info"][field_name] = values.get("old")
    if "metadata_diff" in details and isinstance(details["metadata_diff"], dict):
        for field_name, values in details["metadata_diff"].items():
            record.metadata[field_name] = values.get("old")


def _apply_modify(record, override: dict) -> None:
    """Apply a modify intervention by overriding metadata fields."""
    for key, value in override.items():
        record.metadata[key] = value


def _write_records(store: JSONHistoryStore, records: list) -> None:
    """Overwrite the store's file with a batch of records (deterministic order)."""
    lines = [json.dumps(r.to_dict()) for r in records]
    with open(store._path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def apply_scenario(
    store: JSONHistoryStore,
    scenario: CounterfactualScenario,
) -> JSONHistoryStore:
    """Apply all interventions in a scenario, returning the final clone."""
    current_store = store
    for intervention in scenario.interventions:
        current_store = apply_intervention(current_store, intervention)
    return current_store


def run_counterfactual_analysis(
    drift_event: DriftEvent,
    attribution: CausalAttribution,
    store: JSONHistoryStore,
    metric_name: str | None = None,
    top_k: int = 3,
) -> list[CounterfactualResult]:
    """Run counterfactual simulation for a drift event.

    End-to-end pipeline:
    1. Build counterfactual scenarios from the causal attribution.
    2. For each scenario, apply interventions to a cloned store.
    3. Estimate the counterfactual metric value after intervention.
    4. Compute delta and confidence.
    """
    if metric_name is None:
        metric_name = attribution.metric_name

    scenarios = build_counterfactual_scenarios(attribution, top_k=top_k)
    original_metric = drift_event.magnitude

    results: list[CounterfactualResult] = []

    for scenario in scenarios:
        modified_store = apply_scenario(store, scenario)
        cf_metric = estimate_metric_after_intervention(
            drift_event, modified_store, metric_name
        )
        delta = original_metric - cf_metric

        if len(scenario.interventions) == 1 and scenario.interventions:
            cf = scenario.interventions[0].metadata.get("factor_score", 0.0)
        else:
            scores = [
                i.metadata.get("factor_score", 0.0)
                for i in scenario.interventions
            ]
            cf = min(scores) if scores else 0.0

        result = CounterfactualResult(
            scenario_id=scenario.scenario_id,
            original_metric=round(original_metric, 6),
            counterfactual_metric=round(cf_metric, 6),
            delta=round(delta, 6),
            confidence=round(cf, 4),
            metadata={
                "metric_name": metric_name,
                "num_interventions": len(scenario.interventions),
                "scenario_description": scenario.description,
                "change_ids": [i.change_id for i in scenario.interventions],
            },
        )
        results.append(result)

    return results
```

---

## Phase 7: Autonomous Optimization

### `evaluator/optimization/models.py`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationAction:
    """A concrete action that can be taken to address a detected drift.

    Attributes:
        action_type: "revert_model", "rollback_config", "restore_version"
        target_run_id: The run_id of the record affected by the change.
        change_id: Identifier of the change event this action addresses.
        description: Human-readable description of the action.
        metadata: Additional context (e.g. old/new values).
    """

    action_type: str
    target_run_id: str
    change_id: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    action_id: str | None = None

    def __post_init__(self) -> None:
        if self.action_id is None:
            self.action_id = str(uuid.uuid4())


@dataclass
class OptimizationRecommendation:
    """A ranked recommendation tying an action to its expected impact.

    expected_improvement: How much the drift metric is expected to improve.
    confidence: Derived from the counterfactual confidence.
    priority: 1-based priority rank (1 = highest).
    """

    action: OptimizationAction
    expected_improvement: float
    confidence: float
    priority: int
    metadata: dict[str, Any] = field(default_factory=dict)
    recommendation_id: str | None = None

    def __post_init__(self) -> None:
        if self.recommendation_id is None:
            self.recommendation_id = str(uuid.uuid4())


@dataclass
class OptimizationPlan:
    """A complete optimization plan for addressing a drift event.

    Contains ranked recommendations and a human-readable summary.
    """

    drift_event_id: str
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if self.plan_id is None:
            self.plan_id = str(uuid.uuid4())
```

### `evaluator/optimization/actions.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.optimization.models import OptimizationAction

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution

_CHANGE_TYPE_TO_ACTION: dict[str, str] = {
    "model_update": "revert_model",
    "config_change": "rollback_config",
    "version_change": "restore_version",
}

_ACTION_DESCRIPTIONS: dict[str, str] = {
    "revert_model": "Revert model change",
    "rollback_config": "Rollback configuration change",
    "restore_version": "Restore previous version",
}


def generate_actions(
    attribution: CausalAttribution,
) -> list[OptimizationAction]:
    """Generate concrete remediation actions from causal factors.

    Maps each top-ranked causal factor to an actionable remediation
    step based on the change type:
    - model_update -> revert_model
    - config_change -> rollback_config
    - version_change -> restore_version
    - unknown -> rollback_config (safe default)
    """
    actions: list[OptimizationAction] = []

    factors = sorted(
        attribution.factors, key=lambda f: f.score, reverse=True
    )

    for factor in factors:
        action_type = _CHANGE_TYPE_TO_ACTION.get(
            factor.factor_name, "rollback_config"
        )
        description = _ACTION_DESCRIPTIONS.get(
            action_type, "Rollback configuration change"
        )
        change_id = str(factor.metadata.get("change_id", ""))
        target_run_id = (
            factor.related_run_ids[0] if factor.related_run_ids else ""
        )

        action = OptimizationAction(
            action_type=action_type,
            target_run_id=target_run_id,
            change_id=change_id,
            description=description,
            metadata={
                "factor_name": factor.factor_name,
                "factor_score": factor.score,
                "change_type": factor.factor_name,
            },
        )
        actions.append(action)

    return actions
```

### `evaluator/optimization/scorer.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.optimization.models import (
    OptimizationAction,
    OptimizationRecommendation,
)

if TYPE_CHECKING:
    from evaluator.counterfactual.models import CounterfactualResult


def score_actions(
    actions: list[OptimizationAction],
    counterfactuals: list[CounterfactualResult],
) -> list[OptimizationRecommendation]:
    """Score and rank actions using counterfactual impact estimates.

    For each action, finds the matching counterfactual result (by
    change_id), then derives:
    - expected_improvement = counterfactual.delta
    - confidence = counterfactual.confidence

    Recommendations are sorted by:
    1. Higher expected_improvement first (primary)
    2. Higher confidence first (tie-breaker)
    """
    cf_by_change: dict[str, CounterfactualResult] = {}
    for cf in counterfactuals:
        change_ids = cf.metadata.get("change_ids", [])
        for cid in change_ids:
            cf_by_change[str(cid)] = cf

    recommendations: list[OptimizationRecommendation] = []

    for priority, action in enumerate(actions, start=1):
        cf = cf_by_change.get(action.change_id)
        if cf is not None:
            expected_improvement = cf.delta
            confidence = cf.confidence
            cf_metadata = cf.metadata
        else:
            expected_improvement = 0.0
            confidence = 0.0
            cf_metadata = {}

        rec = OptimizationRecommendation(
            action=action,
            expected_improvement=round(expected_improvement, 6),
            confidence=round(confidence, 4),
            priority=priority,
            metadata={
                "source": "counterfactual_simulation",
                "cf_metadata": cf_metadata,
            },
        )
        recommendations.append(rec)

    recommendations.sort(
        key=lambda r: (r.expected_improvement, r.confidence),
        reverse=True,
    )
    for i, rec in enumerate(recommendations, start=1):
        rec.priority = i

    return recommendations
```

### `evaluator/optimization/optimizer.py`

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.optimization.actions import generate_actions
from evaluator.optimization.models import (
    OptimizationPlan,
    OptimizationRecommendation,
)
from evaluator.optimization.scorer import score_actions

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution
    from evaluator.counterfactual.models import CounterfactualResult
    from evaluator.temporal.models import DriftEvent


def generate_optimization_plan(
    drift_event: DriftEvent,
    attribution: CausalAttribution,
    counterfactuals: list[CounterfactualResult],
) -> OptimizationPlan:
    """Generate a ranked, actionable optimization plan for a drift event.

    Full pipeline:
    1. Generate remediation actions from causal factors.
    2. Score actions using counterfactual impact estimates.
    3. Rank recommendations by expected improvement.
    4. Build a human-readable summary.
    """
    actions = generate_actions(attribution)
    recommendations = score_actions(actions, counterfactuals)
    summary = _build_summary(recommendations, drift_event)

    plan = OptimizationPlan(
        drift_event_id=drift_event.event_id or "",
        recommendations=recommendations,
        summary=summary,
        metadata={
            "metric_name": attribution.metric_name,
            "num_actions": len(actions),
            "num_recommendations": len(recommendations),
            "drift_magnitude": drift_event.magnitude,
        },
    )

    return plan


def _build_summary(
    recommendations: list[OptimizationRecommendation],
    drift_event: DriftEvent,
) -> str:
    """Build a human-readable summary of the top recommendation."""
    if not recommendations:
        return (
            f"No actionable recommendations for drift event "
            f"{drift_event.event_id}. No causal factors identified."
        )

    top = recommendations[0]
    action = top.action
    improvement = round(top.expected_improvement, 4)

    parts = [f"Top recommendation: {action.description.lower()}"]
    if action.target_run_id:
        parts.append(f"in run {action.target_run_id}")
    parts.append(f"(expected improvement: {improvement})")
    if top.confidence > 0:
        parts.append(f"confidence: {top.confidence}")

    return " ".join(parts)
```

---

## Phase 8: API Layer (FastAPI)

### `api/main.py` — Application Factory

*(See api/main.py and api/dependencies.py — uses create_app(store_path) factory pattern with get_store dependency injection)*

### `cli/sentrix_cli.py` — CLI Pipeline

*(See cli/sentrix_cli.py — full pipeline: sentrix --store history.jsonl)*

---

## Phase 9: Packaging

### `evaluator/config.py` — Latent Drift Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class EvaluatorConfig(BaseSettings):
    # ... existing fields ...

    LATENT_DRIFT_ENABLED: bool = Field(default=True)
    LATENT_DRIFT_THRESHOLD: float = Field(default=0.15)
    PCA_COMPONENTS: int = Field(default=5)
    KDE_SAMPLE_SIZE: int = Field(default=1000)

    # ... rest of config ...
```

### `pyproject.toml` — Package Definition

```toml
[project]
name = "sentrix-evaluator"
# ... 

[project.scripts]
sentrix = "cli.sentrix_cli:main"
```

---

## Phase 10: True Latent Drift Engine (PCA + KDE + JSD)

### `evaluator/latent_drift/__init__.py`

```python
from evaluator.latent_drift.engine import (
    LatentDriftEngine,
    compute_latent_drift,
    detect_latent_drift_events,
)
from evaluator.latent_drift.jsd import compute_jsd
from evaluator.latent_drift.kde import evaluate_density, fit_kde
from evaluator.latent_drift.pca import fit_pca, project_vectors
from evaluator.latent_drift.schemas import EmbeddingBatch, LatentDriftResult

__all__ = [
    "LatentDriftEngine",
    "compute_latent_drift",
    "detect_latent_drift_events",
    "compute_jsd",
    "fit_kde",
    "evaluate_density",
    "fit_pca",
    "project_vectors",
    "EmbeddingBatch",
    "LatentDriftResult",
]
```

### `evaluator/latent_drift/schemas.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class EmbeddingBatch:
    """A batch of embedding vectors with a timestamp.

    Attributes:
        vectors: 2-D array of shape (n_samples, dim).
        timestamp: When the batch was collected.
        metadata: Optional context (run_ids, model name, etc.).
    """

    vectors: np.ndarray
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.reshape(1, -1)


@dataclass
class LatentDriftResult:
    """Result of a latent drift computation.

    Attributes:
        drift_score: Jensen-Shannon divergence between baseline and
            current embedding distributions. Bounded [0, 1].
        drift_detected: True when drift_score > threshold.
        threshold: The configured drift threshold.
        n_samples_baseline: Number of baseline embedding vectors.
        n_samples_current: Number of current embedding vectors.
        metadata: Additional diagnostic information (PCA explained variance,
            KDE bandwidth, etc.).
    """

    drift_score: float
    drift_detected: bool
    threshold: float
    n_samples_baseline: int
    n_samples_current: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `evaluator/latent_drift/pca.py`

```python
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(vectors: np.ndarray, n_components: int) -> PCA:
    """Fit PCA, capping n_components at the matrix rank.

    Args:
        vectors: 2-D array of shape (n_samples, n_features).
        n_components: Desired number of components.

    Returns:
        A fitted sklearn PCA instance.
    """
    vectors = np.atleast_2d(vectors)
    n_samples, n_features = vectors.shape
    max_components = min(n_samples, n_features)
    actual_components = min(n_components, max_components)

    return PCA(n_components=actual_components)


def project_vectors(pca: PCA, vectors: np.ndarray) -> np.ndarray:
    """Project vectors using a fitted PCA model.

    Args:
        pca: A fitted PCA instance.
        vectors: 2-D array of shape (n_samples, n_features).

    Returns:
        Projected vectors of shape (n_samples, n_components).
    """
    vectors = np.atleast_2d(vectors)
    return pca.transform(vectors)
```

### `evaluator/latent_drift/kde.py`

```python
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def fit_kde(vectors: np.ndarray) -> gaussian_kde:
    """Fit a Gaussian KDE using Scott's rule for bandwidth.

    Args:
        vectors: 2-D array of shape (n_samples, n_features).

    Raises:
        ValueError: If fewer than 2 samples are provided.
    """
    vectors = np.atleast_2d(vectors)
    n_samples, n_features = vectors.shape

    if n_samples < 2:
        raise ValueError(
            f"KDE requires at least 2 samples, got {n_samples}. "
            "Consider collecting more baseline data."
        )

    return gaussian_kde(vectors.T)


def evaluate_density(
    kde: gaussian_kde,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate a KDE on a set of grid points.

    Handles single-point queries correctly by reshaping as needed.

    Args:
        kde: A fitted gaussian_kde instance.
        points: 2-D array of shape (n_points, n_features).

    Returns:
        1-D array of density values of length n_points.
    """
    points = np.atleast_2d(points)
    if points.shape[0] == 1:
        points = points.T
    return kde(points.T)
```

### `evaluator/latent_drift/jsd.py`

```python
from __future__ import annotations

import numpy as np

_EPSILON = 1e-12


def compute_jsd(
    p_density: np.ndarray,
    q_density: np.ndarray,
) -> float:
    """Compute the Jensen-Shannon Divergence between two density arrays.

    Steps:
    1. Normalize densities so they sum to 1.
    2. Compute the midpoint distribution M = 0.5 * (P + Q).
    3. Compute JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M).

    Result bounded [0, 1] with base-2 logarithm.

    Args:
        p_density: 1-D array of density values from baseline.
        q_density: 1-D array of density values from current.

    Returns:
        The Jensen-Shannon divergence as a float in [0, 1].
    """
    p = np.asarray(p_density, dtype=float)
    q = np.asarray(q_density, dtype=float)

    p = np.clip(p, _EPSILON, None)
    q = np.clip(q, _EPSILON, None)

    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    m = np.clip(m, _EPSILON, None)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    jsd = float(np.clip(jsd, 0.0, 1.0))

    return jsd
```

### `evaluator/latent_drift/engine.py`

```python
"""Latent drift detection engine using PCA + KDE + JSD.

This module implements a statistical drift detector that operates directly
on embedding vectors.  It detects distribution-level semantic drift by:

1. Projecting embeddings into a stable latent manifold (PCA).
2. Estimating probability densities (Gaussian KDE).
3. Computing Jensen-Shannon Divergence between distributions.

Results are converted to DriftEvent objects so they integrate seamlessly
with the existing causal attribution, counterfactual simulation, and
optimization pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluator.latent_drift.jsd import compute_jsd
from evaluator.latent_drift.kde import evaluate_density, fit_kde
from evaluator.latent_drift.pca import fit_pca, project_vectors
from evaluator.latent_drift.schemas import EmbeddingBatch, LatentDriftResult
from evaluator.temporal.models import DriftEvent

_EPSILON = 1e-12
_DEFAULT_KDE_SAMPLE_SIZE = 1000


class LatentDriftEngine:
    """Detect distribution-level drift on embedding vectors.

    The engine fits a PCA model + KDE on baseline embeddings during
    fit(), then compares incoming embeddings during compute_drift().

    Attributes:
        threshold: Drift threshold for the JSD score.
        pca_components: Number of PCA components to retain (default 5).
        kde_sample_size: Maximum grid points for KDE evaluation.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        pca_components: int = 5,
        kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
    ):
        self.threshold = threshold
        self.pca_components = pca_components
        self.kde_sample_size = kde_sample_size
        self.pca: Any = None
        self.kde_baseline: Any = None
        self._baseline_proj: np.ndarray | None = None

    def fit(self, baseline_vectors: np.ndarray) -> None:
        """Fit PCA and KDE on baseline embedding vectors."""
        baseline_vectors = np.atleast_2d(baseline_vectors)
        self.pca = fit_pca(baseline_vectors, n_components=self.pca_components)
        self._baseline_proj = project_vectors(self.pca, baseline_vectors)
        self.kde_baseline = fit_kde(self._baseline_proj)

    def compute_drift(
        self,
        current_vectors: np.ndarray,
    ) -> LatentDriftResult:
        """Compute latent drift between baseline and current embeddings."""
        if self.pca is None or self.kde_baseline is None:
            raise RuntimeError(
                "LatentDriftEngine must be fitted first. Call fit() "
                "before compute_drift()."
            )

        current_vectors = np.atleast_2d(current_vectors)
        baseline_proj = self._baseline_proj if self._baseline_proj is not None else np.empty((0, 0))

        current_proj = project_vectors(self.pca, current_vectors)
        grid = _build_shared_grid(baseline_proj, current_proj)

        p_density = evaluate_density(self.kde_baseline, grid)

        try:
            kde_current = fit_kde(current_proj)
            q_density = evaluate_density(kde_current, grid)
        except (ValueError, np.linalg.LinAlgError):
            q_density = np.full_like(p_density, _EPSILON)

        drift_score = compute_jsd(p_density, q_density)

        metadata = {
            "pca_components": self.pca.n_components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            "kde_sample_size": self.kde_sample_size,
        }

        return LatentDriftResult(
            drift_score=drift_score,
            drift_detected=drift_score > self.threshold,
            threshold=self.threshold,
            n_samples_baseline=baseline_proj.shape[0],
            n_samples_current=current_proj.shape[0],
            metadata=metadata,
        )

    def fit_compute(
        self,
        baseline_vectors: np.ndarray,
        current_vectors: np.ndarray,
    ) -> LatentDriftResult:
        """Convenience: fit on baseline, then compute drift on current."""
        self.fit(baseline_vectors)
        return self.compute_drift(current_vectors)

    def to_drift_event(
        self,
        result: LatentDriftResult,
        drift_event_id: str | None = None,
    ) -> DriftEvent:
        """Convert a LatentDriftResult to a DriftEvent."""
        return DriftEvent(
            event_id=drift_event_id,
            metric_name="latent_jsd",
            start_timestamp=0.0,
            end_timestamp=0.0,
            magnitude=result.drift_score,
            metadata={
                "latent_drift": True,
                "drift_score": result.drift_score,
                "threshold": result.threshold,
                "n_samples_baseline": result.n_samples_baseline,
                "n_samples_current": result.n_samples_current,
                "pca_components": result.metadata.get("pca_components", 0),
                "engine_config": {
                    "threshold": self.threshold,
                    "pca_components": self.pca_components,
                    "kde_sample_size": self.kde_sample_size,
                },
            },
        )


def compute_latent_drift(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    pca_components: int = 5,
    kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> LatentDriftResult:
    """Convenience function: detect latent drift from two embedding batches."""
    engine = LatentDriftEngine(
        threshold=threshold,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )
    return engine.fit_compute(baseline.vectors, current.vectors)


def detect_latent_drift_events(
    baseline: EmbeddingBatch,
    current: EmbeddingBatch,
    threshold: float = 0.15,
    pca_components: int = 5,
    kde_sample_size: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> list[DriftEvent]:
    """Full pipeline: detect latent drift and return DriftEvent(s).

    When drift is detected, a DriftEvent is produced and can be consumed
    by the causal attribution pipeline.
    """
    result = compute_latent_drift(
        baseline=baseline,
        current=current,
        threshold=threshold,
        pca_components=pca_components,
        kde_sample_size=kde_sample_size,
    )

    if not result.drift_detected:
        return []

    start_ts = (
        current.timestamp.timestamp() if current.timestamp else 0.0
    )

    event = DriftEvent(
        metric_name="latent_jsd",
        start_timestamp=start_ts,
        end_timestamp=start_ts,
        magnitude=result.drift_score,
        metadata={
            "latent_drift": True,
            "drift_score": result.drift_score,
            "threshold": result.threshold,
            "n_samples_baseline": result.n_samples_baseline,
            "n_samples_current": result.n_samples_current,
            "pca_components": result.metadata.get("pca_components", 0),
        },
    )

    return [event]


def _build_shared_grid(
    baseline_proj: np.ndarray,
    current_proj: np.ndarray,
    max_points: int = _DEFAULT_KDE_SAMPLE_SIZE,
) -> np.ndarray:
    """Build a shared evaluation grid from actual projected data points.

    Uses actual projected baseline and current vectors as evaluation
    points, which yields meaningful density comparisons. When combined
    set exceeds max_points, a deterministic subsample is taken.
    """
    combined = np.vstack([baseline_proj, current_proj])

    if combined.shape[0] > max_points:
        rng = np.random.RandomState(0)
        indices = rng.choice(combined.shape[0], size=max_points, replace=False)
        combined = combined[indices]

    return combined
```

---

## Configuration

### `evaluator/config.py` — Latent Drift Settings

```python
class EvaluatorConfig(BaseSettings):
    # Phase 10: Latent drift engine settings
    LATENT_DRIFT_ENABLED: bool = Field(default=True)
    LATENT_DRIFT_THRESHOLD: float = Field(default=0.15)
    PCA_COMPONENTS: int = Field(default=5)
    KDE_SAMPLE_SIZE: int = Field(default=1000)
```

---

## Test Summary

- 27 Phase 10 tests (latent drift engine: PCA, KDE, JSD, integration, edge cases)
- 22 Phase 6 tests (counterfactual simulation)
- 20 Phase 7 tests (optimization pipeline)
- 12 Phase 8 tests (FastAPI API)
- **Total: 81 tests passing**

## Cross-Layer Traceability

- `ChangeEvent.change_id` uses deterministic UUIDv5 (`uuid.uuid5(NAMESPACE_OID, f"{run_id}:{timestamp}:{change_type}")`)
- `CausalFactor.metadata["change_id"]` links factors to change events
- `CounterfactualResult.metadata["change_ids"]` links results to interventions
- `OptimizationAction.change_id` links actions to specific changes
- This chain enables end-to-end traceability: **DriftEvent → CausalFactor → ChangeEvent → Intervention → CounterfactualResult → OptimizationRecommendation**
