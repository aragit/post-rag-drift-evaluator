import litellm

from evaluator.cache import EmbeddingCache, ResultCache
from evaluator.config import config
from evaluator.db.pool import connection
from evaluator.logging_config import get_logger
from evaluator.rag_pipelines.base import BaseRAGPipeline, RAGResponse
from evaluator.utils.mock_embedding import (
    generate_mock_completion,
    generate_mock_embedding,
    is_mock_key,
)
from evaluator.utils.retry import async_call_with_retry

logger = get_logger("NaiveRAG")


class NaiveRAG(BaseRAGPipeline):
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.embedding_model = config.EMBEDDING_MODEL
        self._embedding_cache = EmbeddingCache()
        self._result_cache = ResultCache()

    async def _execute_vector_search(
        self, embedding: list[float], k: int = 3
    ) -> list[str]:
        query = """
            SELECT content
            FROM document_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """
        try:
            async with connection() as conn:
                records = await conn.fetch(query, embedding, k)
                return [row["content"] for row in records]
        except Exception as e:
            logger.error(f"Database vector extraction aborted: {e}")
            return [
                "Fallback: Database connectivity failure context execution placeholder."
            ]

    async def execute(self, query: str) -> RAGResponse:
        cached = self._result_cache.get(query, "NaiveRAG")
        if cached is not None:
            logger.info("Result cache hit for NaiveRAG query.")
            return cached

        cached_embedding = self._embedding_cache.get(query)
        if cached_embedding is not None:
            query_vector = cached_embedding
            logger.info("Embedding cache hit for NaiveRAG query.")
        elif is_mock_key(config.OPENAI_API_KEY):
            query_vector = generate_mock_embedding(query)
            logger.info("Using mock embedding for offline mode.")
        else:
            embed_resp = await async_call_with_retry(
                litellm.aembedding, model=self.embedding_model, input=[query]
            )
            query_vector = embed_resp["data"][0]["embedding"]
            self._embedding_cache.set(query, query_vector)

        contexts = await self._execute_vector_search(query_vector)

        formatted_context = "\n".join(contexts)
        prompt = f"Answer the query using ONLY the context provided.\nContext:\n{formatted_context}\n\nQuery: {query}"

        if is_mock_key(config.OPENAI_API_KEY):
            response = generate_mock_completion(prompt)
            logger.info("Using mock completion for offline mode.")
        else:
            response = await async_call_with_retry(
                litellm.acompletion,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        answer = response.choices[0].message.content

        result = RAGResponse(
            query=query,
            retrieved_contexts=contexts,
            generated_answer=answer,
            query_embedding=query_vector,
            metadata={
                "pipeline_name": self.__class__.__name__,
                "model": self.model_name,
                "embedding_model": self.embedding_model,
                "token_usage": dict(response.get("usage", {})),
            },
        )
        self._result_cache.set(query, "NaiveRAG", result)
        return result
