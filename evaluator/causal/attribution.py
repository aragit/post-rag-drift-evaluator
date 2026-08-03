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
    """Score each change feature and return ranked :class:`CausalFactor` list.

    Heuristic scoring (deterministic, no ML):

    - **in_window** (binary): full bonus if change is inside the drift
      window.
    - **time proximity**: inverse of ``time_delta`` (closer changes score
      higher), normalised to [0, 1].
    - **change type weight**: pre-assigned weight (model_update=1.0,
      config_change=0.6, etc.).
    - **drift magnitude**: normalised drift magnitude factor.
    """
    if not features:
        return []

    max_delta = max((f["time_delta"] for f in features), default=0.0)
    max_magnitude = max((f["drift_magnitude"] for f in features), default=1.0)

    factors: list[CausalFactor] = []

    for feat in features:
        # Time proximity component (higher when closer to drift window)
        delta = feat["time_delta"]
        if max_delta > 0:
            time_score = 1.0 - (delta / max_delta)
        else:
            time_score = 1.0

        if feat["in_window"]:
            time_score = time_score * 0.5 + 0.5  # boost in-window changes

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

        # Normalise to [0, 1] range
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
    """Compute a simple confidence score from the factor distribution.

    Uses a normalised max-score approach: the top factor's score
    represents how much it dominates.  A higher concentration
    (one factor clearly dominant) yields higher confidence.
    """
    if not factors:
        return 0.0
    if len(factors) == 1:
        return round(factors[0].score, 4)

    top_score = factors[0].score
    total = sum(f.score for f in factors)

    if total == 0:
        return 0.0

    # Confidence = top score / total score (normalised dominance)
    confidence = top_score / total
    return round(confidence, 4)


def attribute_drift(
    drift_event: DriftEvent,
    store: JSONHistoryStore,
) -> CausalAttribution:
    """Full attribution pipeline for a single drift event.

    Steps:
    1. Extract change events from the history store.
    2. Build feature vectors for each change relative to the drift.
    3. Score and rank causal factors.
    4. Compute confidence.
    """
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
