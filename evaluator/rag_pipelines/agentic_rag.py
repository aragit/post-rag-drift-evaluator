import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import litellm
import json
from typing import List
from evaluator.config import config
from evaluator.rag_pipelines.base import BaseRAGPipeline, RAGResponse

logger = logging.getLogger("AgenticRAG")

class AgenticRAG(BaseRAGPipeline):
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

    def _decompose_query(self, query: str) -> List[str]:
        """Step 1: Neuro reasoning loop plans sub-queries to resolve semantic multi-hop dependencies."""
        planner_prompt = (
            f"Deconstruct this complex user query into exactly two distinct sub-queries "
            f"for optimization. Return as a raw JSON array of strings only. Query: {query}"
        )
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": planner_prompt}],
            response_format={"type": "json_object"}
        )
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [query]
        except Exception:
            logger.warning("Query planner failed JSON coercion. Falling back to default query splitting.")
            return [query, f"Context clarification for {query}"]

    def execute(self, query: str) -> RAGResponse:
        logger.info("Initiating Step 1: Query Deconstruction Planning Chain.")
        sub_queries = self._decompose_query(query)

        all_contexts = []
        primary_vector = None

        for sub_q in sub_queries:
            embed_resp = litellm.embedding(model=self.embedding_model, input=[sub_q])
            vec = embed_resp['data'][0]['embedding']
            if primary_vector is None:
                primary_vector = vec

            contexts = self._execute_vector_search(vec)
            all_contexts.extend(contexts)

        logger.info("Initiating Step 2: Synthesis and Self-Reflection Gen Loop.")
        synthesis_prompt = (
            f"Synthesize an authoritative response from the following multi-hop contexts:\n"
            f"{chr(10).join(all_contexts)}\n\nOriginal Intent: {query}"
        )

        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        answer = response.choices[0].message.content

        return RAGResponse(
            query=query,
            retrieved_contexts=all_contexts,
            generated_answer=answer,
            query_embedding=primary_vector,
            metadata={"sub_queries_generated": sub_queries, "token_usage": dict(response.get("usage", {}))}
        )
