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

    The original store is **never mutated**.  A new :class:`JSONHistoryStore`
    is created at a temporary path with the intervention applied.

    For ``action="remove"``: the change identified by ``intervention.change_id``
    is reverted — the affected record's ``system_version``, ``system_info``,
    and ``metadata`` fields are restored to their pre-change values, using
    the diff details stored in the :class:`ChangeEvent`.

    For ``action="modify"``: specific metadata fields listed in
    ``intervention.override_metadata`` are set on the affected record.

    Args:
        store: The original (read-only) history store.
        intervention: The intervention to apply.

    Returns:
        A new :class:`JSONHistoryStore` with the intervention applied.
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
    """Apply an 'modify' intervention by overriding metadata fields."""
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
    """Apply all interventions in a scenario, returning the final clone.

    Interventions are applied sequentially: each intervention operates on
    the result of the previous one.
    """
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
    2. For each scenario, apply the interventions to a cloned store.
    3. Estimate the counterfactual metric value after intervention.
    4. Compute the delta and confidence.

    Args:
        drift_event: The drift event that triggered the attribution.
        attribution: The causal attribution result for this drift event.
        store: The original (read-only) history store.
        metric_name: Optional override for the metric to estimate.
            Defaults to ``attribution.metric_name``.
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
                i.metadata.get("factor_score", 0.0)
                for i in scenario.interventions
            ]
            if scores:
                cf = min(scores)
            else:
                cf = 0.0

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
