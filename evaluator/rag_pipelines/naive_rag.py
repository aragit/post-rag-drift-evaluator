import litellm
from typing import List
from evaluator.config import config
from evaluator.db.pool import acquire, release
from evaluator.rag_pipelines.base import BaseRAGPipeline, RAGResponse
from evaluator.utils.mock_embedding import (
    is_mock_key,
    generate_mock_embedding,
    generate_mock_completion,
)
from evaluator.utils.retry import call_with_retry
from evaluator.logging_config import get_logger
from evaluator.cache import EmbeddingCache, ResultCache

logger = get_logger("NaiveRAG")


class NaiveRAG(BaseRAGPipeline):
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.embedding_model = config.EMBEDDING_MODEL
        self._embedding_cache = EmbeddingCache()
        self._result_cache = ResultCache()

    async def _execute_vector_search(
        self, embedding: List[float], k: int = 3
    ) -> List[str]:
        query = """
            SELECT content
            FROM document_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2;
        """
        conn = None
        try:
            conn = await acquire()
            records = await conn.fetch(query, embedding, k)
            return [row["content"] for row in records]
        except Exception as e:
            logger.error(f"Database vector extraction aborted: {e}")
            return [
                "Fallback: Database connectivity failure context execution placeholder."
            ]
        finally:
            if conn is not None:
                await release(conn)

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
            embed_resp = call_with_retry(
                litellm.embedding, model=self.embedding_model, input=[query]
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
            response = call_with_retry(
                litellm.completion,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        answer = response.choices[0].message.content

        result = RAGResponse(
            query=query,
            retrieved_contexts=contexts,
            generated_answer=answer,
            query_embedding=query_vector,
            metadata={"token_usage": dict(response.get("usage", {}))},
        )
        self._result_cache.set(query, "NaiveRAG", result)
        return result
