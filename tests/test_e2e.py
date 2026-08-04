from unittest.mock import AsyncMock, patch

import polars as pl

from evaluator.benchmark import run_benchmark


@patch("evaluator.rag_pipelines.naive_rag.litellm.acompletion", new_callable=AsyncMock)
@patch("evaluator.rag_pipelines.naive_rag.litellm.aembedding", new_callable=AsyncMock)
@patch("evaluator.rag_pipelines.agentic_rag.litellm.acompletion", new_callable=AsyncMock)
@patch("evaluator.rag_pipelines.agentic_rag.litellm.aembedding", new_callable=AsyncMock)
@patch("evaluator.utils.metrics.litellm.completion")
async def test_end_to_end_benchmark_execution(
    mock_judge_comp,
    mock_agentic_embed,
    mock_agentic_comp,
    mock_naive_embed,
    mock_naive_comp,
):
    """Verifies that the entire execution and evaluation block functions seamlessly without dependency breaks."""
    mock_naive_embed.return_value = {"data": [{"embedding": [0.1] * 1536}]}
    mock_agentic_embed.return_value = {"data": [{"embedding": [0.1] * 1536}]}

    mock_naive_comp.return_value.choices[
        0
    ].message.content = "Naive Output Answer text string structure payload."
    mock_agentic_comp.return_value.choices[
        0
    ].message.content = '["Sub-query context target one", "Sub-query target two"]'

    mock_judge_comp.return_value.choices[0].message.content = '{"score": 0.95}'

    test_queries = ["Run functional end-to-end telemetry system validations."]
    summary_df = await run_benchmark(test_queries)

    assert isinstance(summary_df, pl.DataFrame)
    assert "Pipeline" in summary_df.columns
    assert "Avg Precision" in summary_df.columns
    assert summary_df.shape[0] == 2
