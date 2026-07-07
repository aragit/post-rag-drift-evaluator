import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import litellm
from typing import List
from evaluator.config import config
from evaluator.rag_pipelines.base import BaseRAGPipeline, RAGResponse
from evaluator.utils.mock_embedding import is_mock_key, generate_mock_embedding, generate_mock_completion

logger = logging.getLogger("NaiveRAG")

class NaiveRAG(BaseRAGPipeline):
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.embedding_model = config.EMBEDDING_MODEL

    def _execute_vector_search(self, embedding: List[float], k: int = 3) -> List[str]:
        """Executes a real Cosine Distance vector search over live pgvector records."""
        query = """
            SELECT content
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg2.connect(config.DATABASE_URL) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (embedding, k))
                    records = cur.fetchall()
                    return [row['content'] for row in records]
        except Exception as e:
            logger.error(f"Database vector extraction aborted: {e}")
            return ["Fallback: Database connectivity failure context execution placeholder."]

    def execute(self, query: str) -> RAGResponse:
        if is_mock_key(config.OPENAI_API_KEY):
            query_vector = generate_mock_embedding(query)
            logger.info("Using mock embedding for offline mode.")
        else:
            embed_resp = litellm.embedding(model=self.embedding_model, input=[query])
            query_vector = embed_resp['data'][0]['embedding']

        contexts = self._execute_vector_search(query_vector)

        formatted_context = "\n".join(contexts)
        prompt = f"Answer the query using ONLY the context provided.\nContext:\n{formatted_context}\n\nQuery: {query}"

        if is_mock_key(config.OPENAI_API_KEY):
            response = generate_mock_completion(prompt)
            logger.info("Using mock completion for offline mode.")
        else:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
        answer = response.choices[0].message.content

        return RAGResponse(
            query=query,
            retrieved_contexts=contexts,
            generated_answer=answer,
            query_embedding=query_vector,
            metadata={"token_usage": dict(response.get("usage", {}))}
        )
