from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.config import config
from evaluator.metrics.quality.faithfulness import evaluate_faithfulness
from evaluator.metrics.results import QualityResult

if TYPE_CHECKING:
    from ingestion.run_schema import RAGRun


def evaluate_context_precision(
    run: RAGRun,
    model: str = config.DEFAULT_MODEL,
) -> QualityResult:
    """Evaluate context precision using a canonical :class:`RAGRun`.

    Delegates to the existing :func:`evaluate_context_precision` in
    ``evaluator.utils.metrics`` while wrapping the scalar result in a
    structured :class:`QualityResult`.
    """
    from evaluator.utils.metrics import evaluate_context_precision as _evaluate

    run.validate()
    score = _evaluate(
        query=run.query,
        contexts=run.retrieved_docs,
        model=model,
    )
    return QualityResult(
        metric_name="context_precision",
        value=score,
        run_id=run.run_id,
    )


def evaluate_all_from_run(
    run: RAGRun, model: str = config.DEFAULT_MODEL
) -> dict[str, QualityResult]:
    """Run both faithfulness and context-precision on one RAGRun.

    Returns a dict mapping metric names to :class:`QualityResult`
    objects.  Existing callers that expect raw floats should use the
    individual ``evaluate_*`` functions directly.
    """
    return {
        "faithfulness": evaluate_faithfulness(run, model=model),
        "context_precision": evaluate_context_precision(run, model=model),
    }
