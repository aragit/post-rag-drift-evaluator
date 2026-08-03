from __future__ import annotations

from typing import TYPE_CHECKING

from evaluator.config import config
from evaluator.metrics.results import QualityResult

if TYPE_CHECKING:
    from ingestion.run_schema import RAGRun


def evaluate_faithfulness(
    run: RAGRun,
    model: str = config.DEFAULT_MODEL,
) -> QualityResult:
    """Evaluate answer faithfulness using a canonical :class:`RAGRun`.

    Delegates to the existing :func:`evaluate_faithfulness` in
    ``evaluator.utils.metrics`` while wrapping the scalar result in a
    structured :class:`QualityResult`.
    """
    from evaluator.utils.metrics import evaluate_faithfulness as _evaluate_faithfulness

    run.validate()
    score = _evaluate_faithfulness(
        query=run.query,
        contexts=run.retrieved_docs,
        answer=run.answer or "",
        model=model,
    )
    return QualityResult(
        metric_name="faithfulness",
        value=score,
        run_id=run.run_id,
    )


def evaluate_faithfulness_from_run(
    run: RAGRun, model: str = config.DEFAULT_MODEL
) -> float:
    """Backward-compatible wrapper returning a raw float.

    .. deprecated::
        Use :func:`evaluate_faithfulness` instead for structured results.
    """
    return evaluate_faithfulness(run, model=model).value
