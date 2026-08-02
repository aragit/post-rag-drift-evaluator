from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluator.rag_pipelines.agentic_rag import AgenticRAG
from evaluator.rag_pipelines.base import RAGResponse
from evaluator.rag_pipelines.naive_rag import NaiveRAG


@pytest.mark.asyncio
async def test_naive_rag_executes_vector_search():
    pipeline = NaiveRAG()
    MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [
        {"content": "context one"},
        {"content": "context two"},
    ]

    with patch("evaluator.rag_pipelines.naive_rag.acquire", return_value=mock_conn):
        with patch("evaluator.rag_pipelines.naive_rag.release"):
            result = await pipeline._execute_vector_search([0.1] * 1536, k=2)

    assert result == ["context one", "context two"]
    mock_conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_naive_rag_returns_correct_response_shape():
    pipeline = NaiveRAG()

    with patch.object(
        pipeline, "_execute_vector_search", return_value=["ctx1", "ctx2"]
    ):
        with patch(
            "evaluator.rag_pipelines.naive_rag.generate_mock_completion"
        ) as mock_comp:
            mock_comp.return_value.choices[0].message.content = "Test answer"
            mock_comp.return_value.get.return_value = {"total_tokens": 100}

            response = await pipeline.execute("test query")

    assert isinstance(response, RAGResponse)
    assert response.query == "test query"
    assert response.retrieved_contexts == ["ctx1", "ctx2"]
    assert response.generated_answer == "Test answer"
    assert response.reflection_iterations == 0
    assert response.final_confidence == 0.0


@pytest.mark.asyncio
async def test_naive_rag_handles_db_failure():
    pipeline = NaiveRAG()

    with patch(
        "evaluator.rag_pipelines.naive_rag.acquire", side_effect=Exception("DB error")
    ):
        result = await pipeline._execute_vector_search([0.1] * 1536, k=2)

    assert len(result) == 1
    assert "Fallback" in result[0]


@pytest.mark.asyncio
async def test_naive_rag_mock_mode_determinism():
    pipeline = NaiveRAG()
    query = "deterministic test query"

    with patch.object(pipeline, "_execute_vector_search", return_value=["ctx"]):
        with patch(
            "evaluator.rag_pipelines.naive_rag.generate_mock_completion"
        ) as mock_comp:
            mock_comp.return_value.choices[0].message.content = "Answer"
            mock_comp.return_value.get.return_value = {"total_tokens": 50}

            r1 = await pipeline.execute(query)
            r2 = await pipeline.execute(query)

    assert r1.query_embedding == r2.query_embedding


@pytest.mark.asyncio
async def test_agentic_rag_decomposes_query():
    pipeline = AgenticRAG()

    with patch.object(pipeline, "_execute_vector_search", return_value=["ctx"]):
        sub_queries = await pipeline._decompose_query(
            "complex query about budget and eligibility"
        )

    assert isinstance(sub_queries, list)
    assert len(sub_queries) > 0


@pytest.mark.asyncio
async def test_agentic_rag_performs_multiple_searches():
    pipeline = AgenticRAG()
    search_count = 0

    async def mock_search(*args, **kwargs):
        nonlocal search_count
        search_count += 1
        return ["ctx"]

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

                        await pipeline.execute("test query")

    assert search_count >= 2


@pytest.mark.asyncio
async def test_agentic_rag_reflection_loop_low_confidence():
    pipeline = AgenticRAG()
    reflection_count = 0

    async def mock_reflect(*args, **kwargs):
        nonlocal reflection_count
        reflection_count += 1
        return {
            "answer_sufficient": False,
            "claims_supported": False,
            "missing_context": ["more data needed"],
            "confidence_score": 0.5,
        }

    with patch.object(pipeline, "_execute_vector_search", return_value=["ctx"]):
        with patch.object(pipeline, "_decompose_query", return_value=["sub1"]):
            with patch.object(pipeline, "_synthesize", return_value="answer"):
                with patch.object(
                    pipeline, "_reflect_on_answer", side_effect=mock_reflect
                ):
                    with patch(
                        "evaluator.rag_pipelines.agentic_rag.generate_mock_completion"
                    ) as mock_comp:
                        mock_comp.return_value.choices[0].message.content = '["sub1"]'
                        mock_comp.return_value.get.return_value = {"total_tokens": 50}

                        response = await pipeline.execute("test query")

    assert response.reflection_iterations > 0


@pytest.mark.asyncio
async def test_agentic_rag_reflection_loop_max_iterations():
    pipeline = AgenticRAG()

    async def mock_reflect_always_low(*args, **kwargs):
        return {
            "answer_sufficient": False,
            "claims_supported": False,
            "missing_context": ["missing"],
            "confidence_score": 0.3,
        }

    with patch.object(pipeline, "_execute_vector_search", return_value=["ctx"]):
        with patch.object(pipeline, "_decompose_query", return_value=["sub1"]):
            with patch.object(pipeline, "_synthesize", return_value="answer"):
                with patch.object(
                    pipeline, "_reflect_on_answer", side_effect=mock_reflect_always_low
                ):
                    with patch(
                        "evaluator.rag_pipelines.agentic_rag.generate_mock_completion"
                    ) as mock_comp:
                        mock_comp.return_value.choices[0].message.content = '["sub1"]'
                        mock_comp.return_value.get.return_value = {"total_tokens": 50}

                        response = await pipeline.execute("test query")

    assert response.reflection_iterations <= 2


@pytest.mark.asyncio
async def test_agentic_rag_synthesis_uses_all_contexts():
    pipeline = AgenticRAG()
    all_contexts = []

    async def mock_synthesize(query, contexts):
        all_contexts.extend(contexts)
        return "synthesized answer"

    async def mock_reflect(*args, **kwargs):
        return {
            "answer_sufficient": True,
            "claims_supported": True,
            "missing_context": [],
            "confidence_score": 0.9,
        }

    with patch.object(
        pipeline, "_execute_vector_search", return_value=["ctx1", "ctx2"]
    ):
        with patch.object(pipeline, "_decompose_query", return_value=["sub1", "sub2"]):
            with patch.object(pipeline, "_synthesize", side_effect=mock_synthesize):
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

                        await pipeline.execute("test query")

    assert len(all_contexts) >= 2


@pytest.mark.asyncio
async def test_agentic_rag_returns_correct_response_shape():
    pipeline = AgenticRAG()

    async def mock_reflect(*args, **kwargs):
        return {
            "answer_sufficient": True,
            "claims_supported": True,
            "missing_context": [],
            "confidence_score": 0.9,
        }

    with patch.object(pipeline, "_execute_vector_search", return_value=["ctx"]):
        with patch.object(pipeline, "_decompose_query", return_value=["sub1"]):
            with patch.object(pipeline, "_synthesize", return_value="answer"):
                with patch.object(
                    pipeline, "_reflect_on_answer", side_effect=mock_reflect
                ):
                    with patch(
                        "evaluator.rag_pipelines.agentic_rag.generate_mock_completion"
                    ) as mock_comp:
                        mock_comp.return_value.choices[0].message.content = '["sub1"]'
                        mock_comp.return_value.get.return_value = {"total_tokens": 50}

                        response = await pipeline.execute("test query")

    assert isinstance(response, RAGResponse)
    assert response.query == "test query"
    assert response.reflection_iterations >= 0
    assert response.final_confidence >= 0.0
