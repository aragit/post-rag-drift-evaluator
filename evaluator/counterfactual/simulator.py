from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Union

from evaluator.counterfactual.estimator import estimate_metric_after_intervention
from evaluator.counterfactual.models import (
    CounterfactualResult,
    CounterfactualScenario,
    Intervention,
)
from evaluator.counterfactual.scenario import build_counterfactual_scenarios

if TYPE_CHECKING:
    from evaluator.causal.models import CausalAttribution
    from evaluator.storage import InMemoryHistoryStore, JSONHistoryStore
    from evaluator.temporal.models import DriftEvent

StoreType = Union["JSONHistoryStore", "InMemoryHistoryStore"]


def _to_in_memory(store: StoreType) -> InMemoryHistoryStore:
    """Convert a store to InMemoryHistoryStore if it isn't one already."""
    from evaluator.storage.in_memory_store import InMemoryHistoryStore

    if isinstance(store, InMemoryHistoryStore):
        return store
    if hasattr(store, "to_in_memory"):
        return store.to_in_memory()
    # Fallback: wrap whatever load_all() returns
    return InMemoryHistoryStore(records=store.load_all())


def apply_intervention(
    store: StoreType,
    intervention: Intervention,
) -> InMemoryHistoryStore:
    """Apply a single intervention to a store and return a modified in-memory clone.

    The original store is **never mutated**.  A new
    :class:`InMemoryHistoryStore` is created with the intervention applied
    entirely in memory — no disk I/O occurs during simulation.

    For ``action="remove"``: the change identified by
    ``intervention.change_id`` is reverted — the affected record's
    ``system_version``, ``system_info``, and ``metadata`` fields are
    restored to their pre-change values, using the diff details stored in
    the :class:`ChangeEvent`.

    For ``action="modify"``: specific metadata fields listed in
    ``intervention.override_metadata`` are set on the affected record.

    Args:
        store: The original (read-only) history store — either
            :class:`JSONHistoryStore` or :class:`InMemoryHistoryStore`.
        intervention: The intervention to apply.

    Returns:
        A new :class:`InMemoryHistoryStore` with the intervention applied.
    """
    from evaluator.causal.change_extractor import extract_change_events

    # Convert to in-memory for all subsequent operations
    in_mem = _to_in_memory(store)
    cloned = in_mem.clone()
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

    cloned._records = records
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


def apply_scenario(
    store: StoreType,
    scenario: CounterfactualScenario,
) -> InMemoryHistoryStore:
    """Apply all interventions in a scenario, returning the final in-memory clone.

    Interventions are applied sequentially: each intervention operates on
    the result of the previous one.  All operations are performed
    entirely in memory — no disk writes occur during simulation.
    """
    current_store = _to_in_memory(store)
    for intervention in scenario.interventions:
        current_store = apply_intervention(current_store, intervention)
    return current_store


def run_counterfactual_analysis(
    drift_event: DriftEvent,
    attribution: CausalAttribution,
    store: StoreType,
    metric_name: str | None = None,
    top_k: int = 3,
) -> list[CounterfactualResult]:
    """Run counterfactual simulation for a drift event.

    End-to-end pipeline:
    1. Build counterfactual scenarios from the causal attribution.
    2. For each scenario, apply the interventions to a cloned store
       (in-memory only — no disk I/O).
    3. Estimate the counterfactual metric value after intervention.
    4. Compute the delta and confidence.

    Args:
        drift_event: The drift event that triggered the attribution.
        attribution: The causal attribution result for this drift event.
        store: The original (read-only) history store.
        metric_name: Optional override for the metric to estimate.
        top_k: Maximum number of individual-factor scenarios to generate.

    Returns:
        A list of :class:`CounterfactualResult`, one per scenario.
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
                i.metadata.get("factor_score", 0.0) for i in scenario.interventions
            ]
            cf = min(scores) if scores else 0.0

        result = CounterfactualResult(
            scenario_id=scenario.scenario_id or str(uuid.uuid4()),
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
