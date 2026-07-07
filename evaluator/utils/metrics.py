import json
import logging
import litellm
from typing import List
from evaluator.config import config
from evaluator.utils.mock_embedding import is_mock_key

logger = logging.getLogger("MetricsJudge")

MOCK_SCORE = 0.85

def evaluate_faithfulness(query: str, contexts: List[str], answer: str, model: str = config.DEFAULT_MODEL) -> float:
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
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except Exception as e:
        logger.error(f"Faithfulness eval failed: {e}")
        return 0.0

def evaluate_context_precision(query: str, contexts: List[str], model: str = config.DEFAULT_MODEL) -> float:
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
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return float(result.get("score", 0.0))
    except Exception as e:
        logger.error(f"Context precision eval failed: {e}")
        return 0.0
