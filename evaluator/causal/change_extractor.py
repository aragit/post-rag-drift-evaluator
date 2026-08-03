from __future__ import annotations

from typing import Any

from evaluator.causal.models import ChangeEvent


def extract_change_events(store: Any) -> list[ChangeEvent]:
    """Extract chronological change events from a JSONHistoryStore.

    Walks all EvaluationRecords sorted by timestamp and detects
    differences between consecutive runs in:

    - ``system_version``  → ``"version_change"``
    - ``metadata``        → ``"config_change"``
    - ``system_info`` (if present in metadata) → ``"model_update"``

    Returns a chronologically sorted list of :class:`ChangeEvent`.
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
    # If system_info was stored in metadata (by RAGResponse.from_ragrun),
    # pull it to top-level for easier diffing
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

    # system_version changes
    if prev.get("system_version") != curr.get("system_version"):
        has_change = True
        details["old_system_version"] = prev.get("system_version")
        details["new_system_version"] = curr.get("system_version")
        if change_type == "unknown":
            change_type = "version_change"

    # pipeline name changes
    if prev.get("pipeline_name") != curr.get("pipeline_name"):
        has_change = True
        details["old_pipeline"] = prev.get("pipeline_name")
        details["new_pipeline"] = curr.get("pipeline_name")
        if change_type == "unknown":
            change_type = "model_update"

    # system_info changes (model, embedding_model, retriever, version)
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

    # metadata changes (anything left in metadata)
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
