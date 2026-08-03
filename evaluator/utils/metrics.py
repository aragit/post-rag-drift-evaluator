from __future__ import annotations

import json
from typing import TYPE_CHECKING

import litellm

from evaluator.config import config
from evaluator.logging_config import get_logger
from evaluator.utils.mock_embedding import is_mock_key
from evaluator.utils.retry import call_with_retry

if TYPE_CHECKING:
    from ingestion.run_schema import RAGRun

logger = get_logger("MetricsJudge")

MOCK_SCORE = 0.85


def evaluate_faithfulness(
    query: str,
    contexts: list[str],
    answer: str,
    model: str = config.DEFAULT_MODEL,
) -> float:
    """Measures if the generated answer is strictly grounded in the provided context (0.0 to 1.0)."""
    context_str = "\n".join(contexts)
    prompt = (
        f"You are an objective scoring judge.\n"
        f"Given the Context and the Answer below, determine if the Answer contains ANY hallucinations or claims not directly supported by the Context.\n"
        f"Context: {context_str}\n"
        f"Answer: {answer}\n"
        f"Output ONLY a valid JSON object with a single key 'score' containing a float between 0.0 (contains hallucinations) and 1.0 (strictly faithful)."
    )

    try:
        if is_mock_key(config.OPENAI_API_KEY):
            logger.info("Using mock score for offline mode.")
            return MOCK_SCORE
        response = call_with_retry(
            litellm.completion,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error("Faithfulness eval failed: malformed LLM JSON response")
        return 0.0
    except Exception as e:
        logger.error(f"Faithfulness eval failed after retries: {e}")
        return 0.0


def evaluate_context_precision(
    query: str, contexts: list[str], model: str = config.DEFAULT_MODEL
) -> float:
    """Measures the signal-to-noise ratio of the retrieved contexts (0.0 to 1.0)."""
    context_str = "\n".join(contexts)
    prompt = (
        f"You are an objective scoring judge.\n"
        f"Evaluate the following retrieved Context for its relevance to the Query.\n"
        f"Query: {query}\n"
        f"Context: {context_str}\n"
        f"Output ONLY a valid JSON object with a single key 'score' containing a float between 0.0 (completely irrelevant) and 1.0 (highly relevant and precise)."
    )

    try:
        if is_mock_key(config.OPENAI_API_KEY):
            logger.info("Using mock score for offline mode.")
            return MOCK_SCORE
        response = call_with_retry(
            litellm.completion,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.error("Context precision eval failed: malformed LLM JSON response")
        return 0.0
    except Exception as e:
        logger.error(f"Context precision eval failed after retries: {e}")
        return 0.0


# ── RAGRun-aware accessors ────────────────────────────────────────────────


def evaluate_faithfulness_from_run(
    run: RAGRun, model: str = config.DEFAULT_MODEL
) -> float:
    """Evaluate faithfulness using a canonical :class:`RAGRun`.

    Extracts the query, retrieved docs, and generated answer from
    the run and delegates to :func:`evaluate_faithfulness`.
    """
    run.validate()
    return evaluate_faithfulness(
        query=run.query,
        contexts=run.retrieved_docs,
        answer=run.answer or "",
        model=model,
    )


def evaluate_context_precision_from_run(
    run: RAGRun, model: str = config.DEFAULT_MODEL
) -> float:
    """Evaluate context precision using a canonical :class:`RAGRun`.

    Extracts the query and retrieved docs from the run and delegates
    to :func:`evaluate_context_precision`.
    """
    run.validate()
    return evaluate_context_precision(
        query=run.query,
        contexts=run.retrieved_docs,
        model=model,
    )


def evaluate_all_from_run(
    run: RAGRun, model: str = config.DEFAULT_MODEL
) -> dict[str, float]:
    """Convenience helper: run both faithfulness and precision on one RAGRun.

    Returns a dict with keys ``faithfulness`` and ``context_precision``.
    """
    return {
        "faithfulness": evaluate_faithfulness_from_run(run, model=model),
        "context_precision": evaluate_context_precision_from_run(run, model=model),
    }
