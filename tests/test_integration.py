from contextlib import asynccontextmanager
from unittest.mock import patch

import asyncpg
import numpy as np
import polars as pl
import pytest

from evaluator.alerts import AlertManager
from evaluator.config import config
from evaluator.drift_monitor import DriftMonitor
from evaluator.drift_store import DriftStore
from evaluator.rag_pipelines.agentic_rag import AgenticRAG
from evaluator.rag_pipelines.naive_rag import NaiveRAG


@asynccontextmanager
async def _loop_local_store():
    """Yield a DriftStore backed by a dedicated pool bound to this test's loop."""
    pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=2)
    try:
        yield DriftStore(pool=pool)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_full_research_workflow(postgres_available):
    """End-to-end test: seed DB → run pipelines → compute drift → store → verify."""
    if not postgres_available:
        pytest.skip("PostgreSQL is not available")

    np.random.seed(42)
    n_vectors = 20
    dim = 32

    baseline_vectors = np.random.normal(0, 1, (n_vectors, dim)).tolist()
    current_vectors = np.random.normal(0, 1, (n_vectors, dim)).tolist()

    async with _loop_local_store() as store:
        await store.clear_history()

        naive_pipeline = NaiveRAG()
        agentic_pipeline = AgenticRAG()

        with patch.object(
            naive_pipeline, "_execute_vector_search", return_value=["ctx1", "ctx2"]
        ):
            with patch(
                "evaluator.rag_pipelines.naive_rag.generate_mock_completion"
            ) as mock_naive_comp:
                mock_naive_comp.return_value.choices[0].message.content = "Naive answer"
                mock_naive_comp.return_value.get.return_value = {"total_tokens": 50}

                naive_response = await naive_pipeline.execute("test query")

        assert naive_response.query == "test query"
        assert naive_response.generated_answer == "Naive answer"

        with patch.object(
            agentic_pipeline, "_execute_vector_search", return_value=["ctx1", "ctx2"]
        ):
            with patch.object(
                agentic_pipeline, "_decompose_query", return_value=["sub1"]
            ):
                with patch.object(
                    agentic_pipeline, "_synthesize", return_value="Agentic answer"
                ):
                    with patch.object(
                        agentic_pipeline,
                        "_reflect_on_answer",
                        return_value={
                            "answer_sufficient": True,
                            "claims_supported": True,
                            "missing_context": [],
                            "confidence_score": 0.95,
                        },
                    ):
                        with patch(
                            "evaluator.rag_pipelines.agentic_rag.generate_mock_completion"
                        ) as mock_agentic_comp:
                            mock_agentic_comp.return_value.choices[
                                0
                            ].message.content = '["sub1"]'
                            mock_agentic_comp.return_value.get.return_value = {
                                "total_tokens": 50
                            }

                            agentic_response = await agentic_pipeline.execute(
                                "test query"
                            )

        assert agentic_response.query == "test query"

        baseline_df = pl.DataFrame({"embedding": baseline_vectors})
        current_df = pl.DataFrame({"embedding": current_vectors})

        monitor = DriftMonitor(store=store)
        drift_result = await monitor.compute_comprehensive_drift(
            baseline_df, current_df
        )

        assert "js_divergence" in drift_result
        assert "mmd_score" in drift_result
        assert "max_component_kl" in drift_result
        assert "is_drifted" in drift_result

        history = await store.get_recent_history(hours=24)
        assert history.height >= 1


@pytest.mark.asyncio
async def test_drift_alert_integration(postgres_available):
    """Test that drift detection triggers alerts and is persisted."""
    if not postgres_available:
        pytest.skip("PostgreSQL is not available")

    np.random.seed(42)
    n_vectors = 20
    dim = 32

    baseline_vectors = np.random.normal(0, 1, (n_vectors, dim)).tolist()
    shifted_vectors = np.random.normal(5.0, 1, (n_vectors, dim)).tolist()

    async with _loop_local_store() as store:
        await store.clear_history()

        baseline_df = pl.DataFrame({"embedding": baseline_vectors})
        shifted_df = pl.DataFrame({"embedding": shifted_vectors})

        monitor = DriftMonitor(store=store)
        drift_result = await monitor.compute_comprehensive_drift(
            baseline_df, shifted_df
        )

        assert drift_result["is_drifted"]

        history = await store.get_recent_history(hours=24)
        assert history.height >= 1

        alert_manager = AlertManager()
        with patch.object(alert_manager, "_send_via_webhook"):
            alert_manager.send_alert(
                jsd_score=drift_result["js_divergence"],
                threshold=0.15,
                mmd_score=drift_result["mmd_score"],
                mmd_p_value=drift_result["mmd_p_value"],
            )

        latest = await store.get_latest_drift()
        assert latest is not None
        assert latest["is_drifted"] == 1
