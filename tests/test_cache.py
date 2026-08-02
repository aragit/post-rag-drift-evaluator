import json
from unittest.mock import MagicMock, patch

from evaluator.cache import EmbeddingCache, ResultCache, _make_key
from evaluator.rag_pipelines.naive_rag import NaiveRAG
from evaluator.rag_pipelines.agentic_rag import AgenticRAG
from evaluator.rag_pipelines.base import RAGResponse


class TestEmbeddingCache:
    def test_cache_hit_returns_correct_embedding(self):
        mock_redis = MagicMock()
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = json.dumps(embedding)

        cache = EmbeddingCache()
        cache._client = mock_redis

        result = cache.get("test query")
        assert result == embedding
        mock_redis.get.assert_called_once()

    def test_cache_miss_calls_litellm_and_stores_result(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cache = EmbeddingCache()
        cache._client = mock_redis

        result = cache.get("new query")
        assert result is None
        mock_redis.get.assert_called_once()

        cache.set("new query", [0.4, 0.5, 0.6])
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == _make_key("embedding", "new query")
        assert json.loads(call_args[0][2]) == [0.4, 0.5, 0.6]

    def test_ttl_expiration(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cache = EmbeddingCache(ttl=1)
        cache._client = mock_redis

        cache.set("query", [0.1, 0.2])
        assert mock_redis.setex.call_count == 1
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 1

    def test_redis_failure_returns_none(self):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis down")

        cache = EmbeddingCache()
        cache._client = mock_redis

        result = cache.get("query")
        assert result is None


class TestResultCache:
    def test_cache_hit_returns_correct_response(self):
        mock_redis = MagicMock()
        response = RAGResponse(
            query="test",
            retrieved_contexts=["ctx1"],
            generated_answer="answer",
            query_embedding=[0.1, 0.2],
        )
        mock_redis.get.return_value = response.model_dump_json()

        cache = ResultCache()
        cache._client = mock_redis

        result = cache.get("test", "NaiveRAG")
        assert result is not None
        assert result.query == "test"
        assert result.generated_answer == "answer"

    def test_cache_miss_returns_none(self):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cache = ResultCache()
        cache._client = mock_redis

        result = cache.get("test", "NaiveRAG")
        assert result is None

    def test_redis_failure_returns_none(self):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis down")

        cache = ResultCache()
        cache._client = mock_redis

        result = cache.get("test", "NaiveRAG")
        assert result is None


class TestNaiveRAGCacheIntegration:
    @patch("evaluator.rag_pipelines.naive_rag.EmbeddingCache")
    @patch("evaluator.rag_pipelines.naive_rag.ResultCache")
    @patch("evaluator.rag_pipelines.naive_rag.is_mock_key")
    async def test_naive_rag_uses_embedding_cache(
        self, mock_is_mock, mock_result_cache_cls, mock_embed_cache_cls
    ):
        mock_is_mock.return_value = True

        mock_embed_cache = MagicMock()
        mock_embed_cache.get.return_value = [0.1, 0.2, 0.3]
        mock_embed_cache_cls.return_value = mock_embed_cache

        mock_result_cache = MagicMock()
        mock_result_cache.get.return_value = None
        mock_result_cache_cls.return_value = mock_result_cache

        pipeline = NaiveRAG()

        with patch.object(pipeline, "_execute_vector_search", return_value=["ctx1"]):
            with patch(
                "evaluator.rag_pipelines.naive_rag.generate_mock_completion"
            ) as mock_gen:
                mock_gen.return_value.choices[0].message.content = "answer"
                mock_gen.return_value.get.return_value = {"total_tokens": 10}
                response = await pipeline.execute("cached query")

        mock_embed_cache.get.assert_called_once_with("cached query")
        assert response.query == "cached query"

    @patch("evaluator.rag_pipelines.naive_rag.EmbeddingCache")
    @patch("evaluator.rag_pipelines.naive_rag.ResultCache")
    @patch("evaluator.rag_pipelines.naive_rag.is_mock_key")
    async def test_naive_rag_result_cache_hit(
        self, mock_is_mock, mock_result_cache_cls, mock_embed_cache_cls
    ):
        mock_is_mock.return_value = True

        mock_embed_cache = MagicMock()
        mock_embed_cache.get.return_value = None
        mock_embed_cache_cls.return_value = mock_embed_cache

        cached_response = RAGResponse(
            query="cached query",
            retrieved_contexts=["ctx1"],
            generated_answer="cached answer",
            query_embedding=[0.1, 0.2],
        )
        mock_result_cache = MagicMock()
        mock_result_cache.get.return_value = cached_response
        mock_result_cache_cls.return_value = mock_result_cache

        pipeline = NaiveRAG()
        response = await pipeline.execute("cached query")

        mock_result_cache.get.assert_called_once_with("cached query", "NaiveRAG")
        assert response.generated_answer == "cached answer"


class TestAgenticRAGCacheIntegration:
    @patch("evaluator.rag_pipelines.agentic_rag.EmbeddingCache")
    @patch("evaluator.rag_pipelines.agentic_rag.ResultCache")
    @patch("evaluator.rag_pipelines.agentic_rag.is_mock_key")
    async def test_agentic_rag_embedding_cache_used(
        self, mock_is_mock, mock_result_cache_cls, mock_embed_cache_cls
    ):
        mock_is_mock.return_value = True

        mock_embed_cache = MagicMock()
        mock_embed_cache.get.return_value = None
        mock_embed_cache_cls.return_value = mock_embed_cache

        mock_result_cache = MagicMock()
        mock_result_cache.get.return_value = None
        mock_result_cache_cls.return_value = mock_result_cache

        pipeline = AgenticRAG()

        with patch.object(pipeline, "_decompose_query", return_value=["sub1"]):
            with patch.object(
                pipeline, "_execute_vector_search", return_value=["ctx1"]
            ):
                with patch.object(pipeline, "_synthesize", return_value="answer"):
                    with patch.object(
                        pipeline,
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
                        ) as mock_gen:
                            mock_gen.return_value.choices[
                                0
                            ].message.content = '["sub1"]'
                            mock_gen.return_value.get.return_value = {
                                "total_tokens": 10
                            }
                            response = await pipeline.execute("test query")

        mock_embed_cache.get.assert_called()
        assert response.query == "test query"

    @patch("evaluator.rag_pipelines.agentic_rag.EmbeddingCache")
    @patch("evaluator.rag_pipelines.agentic_rag.ResultCache")
    @patch("evaluator.rag_pipelines.agentic_rag.is_mock_key")
    async def test_agentic_rag_skips_result_cache(
        self, mock_is_mock, mock_result_cache_cls, mock_embed_cache_cls
    ):
        mock_is_mock.return_value = True

        mock_embed_cache = MagicMock()
        mock_embed_cache.get.return_value = None
        mock_embed_cache_cls.return_value = mock_embed_cache

        mock_result_cache = MagicMock()
        mock_result_cache.get.return_value = None
        mock_result_cache_cls.return_value = mock_result_cache

        pipeline = AgenticRAG()

        with patch.object(pipeline, "_decompose_query", return_value=["sub1"]):
            with patch.object(
                pipeline, "_execute_vector_search", return_value=["ctx1"]
            ):
                with patch.object(pipeline, "_synthesize", return_value="answer"):
                    with patch.object(
                        pipeline,
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
                        ) as mock_gen:
                            mock_gen.return_value.choices[
                                0
                            ].message.content = '["sub1"]'
                            mock_gen.return_value.get.return_value = {
                                "total_tokens": 10
                            }
                            response = await pipeline.execute("test query")

        mock_result_cache.get.assert_not_called()
        assert response.query == "test query"
