from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.metrics.drift import evaluate_drift
from evaluator.metrics.quality import evaluate_all_from_run
from evaluator.metrics.results import DriftResult, QualityResult

if TYPE_CHECKING:
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

        return {
            "drift": [drift_result],
            "quality": list(quality_results.values()),
        }
