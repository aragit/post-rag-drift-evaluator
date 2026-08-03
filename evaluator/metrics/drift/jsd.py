from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from evaluator.drift_monitor import DriftMonitor
from evaluator.metrics.results import DriftResult

if TYPE_CHECKING:
    from ingestion.run_schema import RAGRun


def _run_to_embedding_series(run: RAGRun) -> pl.DataFrame:
    """Extract the best available embedding vector from a RAGRun.

    Fallback order: ``query_embedding`` → ``answer_embedding`` →
    ``retrieved_embeddings[0]``.

    Returns a Polars DataFrame with a single ``embedding`` column
    containing one row (the selected vector as a list).
    """
    if run.query_embedding is not None:
        vec = np.asarray(run.query_embedding, dtype=float)
    elif run.answer_embedding is not None:
        vec = np.asarray(run.answer_embedding, dtype=float)
    elif run.retrieved_embeddings and run.retrieved_embeddings[0] is not None:
        vec = np.asarray(run.retrieved_embeddings[0], dtype=float)
    else:
        raise ValueError(
            f"RAGRun {run.run_id} has no usable embedding for drift evaluation"
        )
    return pl.DataFrame({"embedding": [vec.tolist()]})


def evaluate_drift(
    baseline_run: RAGRun,
    current_run: RAGRun,
) -> DriftResult:
    """Compute Jensen-Shannon drift between two RAGRun embeddings.

    Extracts embeddings from both runs, delegates to the existing
    :class:`DriftMonitor` JSD implementation, and wraps the result
    in a :class:`DriftResult`.

    The mathematical code in ``DriftMonitor`` is **not** modified —
    this function is a thin adapter that maps the ``RAGRun``-based
    interface onto the existing DataFrame-based API.
    """
    baseline_run.validate()
    current_run.validate()

    monitor = DriftMonitor()
    baseline_df = _run_to_embedding_series(baseline_run)
    current_df = _run_to_embedding_series(current_run)

    js_score, is_drifted = monitor.compute_jensen_shannon_drift(
        baseline_df, current_df, "embedding"
    )

    return DriftResult(
        metric_name="js_divergence",
        value=js_score,
        metadata={"is_drifted": is_drifted, "method": "jensen_shannon"},
        baseline_run_id=baseline_run.run_id,
        current_run_id=current_run.run_id,
    )
