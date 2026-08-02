from unittest.mock import patch

import pytest

from evaluator.drift_monitor import DriftMonitor
from evaluator.rag_pipelines.agentic_rag import AgenticRAG
from evaluator.rag_pipelines.naive_rag import NaiveRAG


@pytest.mark.asyncio
async def test_naive_rag_db_connection_timeout():
    pipeline = NaiveRAG()

    with patch(
        "evaluator.rag_pipelines.naive_rag.acquire",
        side_effect=TimeoutError("Connection timeout"),
    ):
        result = await pipeline._execute_vector_search([0.1] * 1536, k=2)

    assert len(result) == 1
    assert "Fallback" in result[0]


@pytest.mark.asyncio
async def test_agentic_rag_partial_db_failure():
    pipeline = AgenticRAG()
    call_count = 0

    async def mock_search(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["ctx1"]
        return []

    async def mock_reflect(*args, **kwargs):
        return {
            "answer_sufficient": True,
            "claims_supported": True,
            "missing_context": [],
            "confidence_score": 0.9,
        }

    with patch.object(pipeline, "_execute_vector_search", side_effect=mock_search):
        with patch.object(pipeline, "_decompose_query", return_value=["sub1", "sub2"]):
            with patch.object(pipeline, "_synthesize", return_value="answer"):
                with patch.object(
                    pipeline, "_reflect_on_answer", side_effect=mock_reflect
                ):
                    with patch(
                        "evaluator.rag_pipelines.agentic_rag.generate_mock_completion"
                    ) as mock_comp:
                        mock_comp.return_value.choices[
                            0
                        ].message.content = '["sub1", "sub2"]'
                        mock_comp.return_value.get.return_value = {"total_tokens": 50}

                        response = await pipeline.execute("test query")

    assert response.generated_answer == "answer"


def test_drift_monitor_with_empty_baseline():
    monitor = DriftMonitor()
    import numpy as np
    import polars as pl

    empty_df = pl.DataFrame({"embedding": []})
    current_df = pl.DataFrame({"embedding": np.random.randn(10, 128).tolist()})

    with pytest.raises(Exception):
        monitor.compute_jensen_shannon_drift(empty_df, current_df)


@pytest.mark.asyncio
async def test_drift_monitor_with_mismatched_dimensions():
    monitor = DriftMonitor()
    import numpy as np
    import polars as pl

    baseline = pl.DataFrame({"embedding": np.random.randn(10, 128).tolist()})
    current = pl.DataFrame({"embedding": np.random.randn(10, 256).tolist()})

    with pytest.raises(Exception):
        await monitor.compute_comprehensive_drift(baseline, current)
