from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.metrics.drift import evaluate_drift
from evaluator.metrics.quality import evaluate_all_from_run
from evaluator.metrics.results import DriftResult, QualityResult

if TYPE_CHECKING:
    from evaluator.storage import JSONHistoryStore
    from ingestion.run_schema import RAGRun


class RAGEvaluator:
    """Orchestration layer that coordinates metric domains.

    The evaluator **does not** compute metrics itself — it delegates
    to the drift and quality domains and assembles their structured
    results into a single response.

    Usage::

        evaluator = RAGEvaluator()
        result = evaluator.evaluate(baseline_run, current_run)
        # result["drift"]   -> list[DriftResult]
        # result["quality"]  -> list[QualityResult]
    """

    def __init__(self, history_store: JSONHistoryStore | None = None):
        self.history_store = history_store

    def evaluate(
        self,
        baseline_run: RAGRun,
        current_run: RAGRun,
    ) -> dict[str, list[DriftResult | QualityResult]]:
        """Run drift + quality metrics across two RAGRun objects.

        Parameters
        ----------
        baseline_run:
            The reference / baseline evaluation run.
        current_run:
            The new evaluation run to compare against the baseline.

        Returns
        -------
        dict
            ``{"drift": [...], "quality": [...]}`` — each list contains
            structured result objects.
        """
        drift_result = evaluate_drift(baseline_run, current_run)
        quality_results = evaluate_all_from_run(current_run)

        all_results = [drift_result, *quality_results.values()]
        self._persist_if_needed(baseline_run, current_run, all_results)

        return {
            "drift": [drift_result],
            "quality": list(quality_results.values()),
        }

    def _persist_if_needed(
        self,
        baseline_run: RAGRun,
        current_run: RAGRun,
        metrics: list[DriftResult | QualityResult],
    ) -> None:
        """Store an EvaluationRecord if a history store is configured."""
        if self.history_store is None:
            return
        from evaluator.storage import EvaluationRecord

        record = EvaluationRecord(
            run_id=current_run.run_id,
            metrics=metrics,
            metadata={
                "baseline_run_id": baseline_run.run_id,
                "pipeline_name": current_run.system_info.name
                if current_run.system_info
                else None,
            },
            system_version=current_run.system_version,
        )
        self.history_store.save(record)
