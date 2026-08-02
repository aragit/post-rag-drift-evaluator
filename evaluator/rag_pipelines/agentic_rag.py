import litellm
import json
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

logger = get_logger("AgenticRAG")


class AgenticRAG(BaseRAGPipeline):
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
            return []
        finally:
            if conn is not None:
                await release(conn)

    async def _decompose_query(self, query: str) -> List[str]:
        planner_prompt = (
            f"Deconstruct this complex user query into exactly two distinct sub-queries "
            f"for optimization. Return as a raw JSON array of strings only. Query: {query}"
        )
        if is_mock_key(config.OPENAI_API_KEY):
            response = generate_mock_completion(planner_prompt, response_format="json")
            logger.info("Using mock completion for offline mode.")
        else:
            response = call_with_retry(
                litellm.completion,
                model=self.model_name,
                messages=[{"role": "user", "content": planner_prompt}],
                response_format={"type": "json_object"},
            )
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [query]
        except Exception:
            logger.warning(
                "Query planner failed JSON coercion. Falling back to default query splitting."
            )
            return [query, f"Context clarification for {query}"]

    async def _reflect_on_answer(
        self, query: str, answer: str, contexts: List[str]
    ) -> dict:
        reflection_prompt = (
            f"Evaluate the following synthesized answer against the original query and retrieved contexts.\n"
            f"Original Query: {query}\n"
            f"Answer: {answer}\n"
            f"Retrieved Contexts: {chr(10).join(contexts)}\n"
            f"Respond with a JSON object containing: answer_sufficient (bool), claims_supported (bool), "
            f"missing_context (list of strings), confidence_score (float 0.0-1.0)."
        )
        if is_mock_key(config.OPENAI_API_KEY):
            response = generate_mock_completion(
                reflection_prompt, response_format={"type": "json_object"}
            )
            logger.info("Using mock reflection for offline mode.")
        else:
            response = call_with_retry(
                litellm.completion,
                model=self.model_name,
                messages=[{"role": "user", "content": reflection_prompt}],
                response_format={"type": "json_object"},
            )
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed
        except Exception:
            logger.warning(
                "Reflection eval failed JSON coercion. Assuming answer is sufficient."
            )
            return {
                "answer_sufficient": True,
                "claims_supported": True,
                "missing_context": [],
                "confidence_score": 1.0,
            }

    async def _synthesize(self, query: str, contexts: List[str]) -> str:
        synthesis_prompt = (
            f"Synthesize an authoritative response from the following multi-hop contexts:\n"
            f"{chr(10).join(contexts)}\n\nOriginal Intent: {query}"
        )
        if is_mock_key(config.OPENAI_API_KEY):
            response = generate_mock_completion(synthesis_prompt)
            logger.info("Using mock completion for offline mode.")
        else:
            response = call_with_retry(
                litellm.completion,
                model=self.model_name,
                messages=[{"role": "user", "content": synthesis_prompt}],
            )
        return response.choices[0].message.content

    async def execute(self, query: str) -> RAGResponse:
        logger.info("Initiating Step 1: Query Deconstruction Planning Chain.")
        sub_queries = await self._decompose_query(query)

        all_contexts = []
        primary_vector = None

        for sub_q in sub_queries:
            cached_embedding = self._embedding_cache.get(sub_q)
            if cached_embedding is not None:
                vec = cached_embedding
                logger.info("Embedding cache hit for AgenticRAG sub-query.")
            elif is_mock_key(config.OPENAI_API_KEY):
                vec = generate_mock_embedding(sub_q)
                logger.info("Using mock embedding for offline mode.")
            else:
                embed_resp = call_with_retry(
                    litellm.embedding, model=self.embedding_model, input=[sub_q]
                )
                vec = embed_resp["data"][0]["embedding"]
                self._embedding_cache.set(sub_q, vec)

            if primary_vector is None:
                primary_vector = vec

            contexts = await self._execute_vector_search(vec)
            all_contexts.extend(contexts)

        answer = await self._synthesize(query, all_contexts)

        reflection_iterations = 0
        final_confidence = 0.0

        for i in range(config.MAX_REFLECTION_ITERATIONS):
            reflection = await self._reflect_on_answer(query, answer, all_contexts)
            confidence = reflection.get("confidence_score", 1.0)
            missing = reflection.get("missing_context", [])
            final_confidence = confidence
            reflection_iterations = i + 1

            if confidence >= 0.7 and not missing:
                logger.info(
                    f"Reflection iteration {reflection_iterations}: answer sufficient (confidence={confidence})."
                )
                break

            logger.info(
                f"Reflection iteration {reflection_iterations}: confidence={confidence}, "
                f"missing_context={missing}. Re-querying..."
            )

            for missing_item in missing:
                cached_embedding = self._embedding_cache.get(missing_item)
                if cached_embedding is not None:
                    vec = cached_embedding
                elif is_mock_key(config.OPENAI_API_KEY):
                    vec = generate_mock_embedding(missing_item)
                else:
                    embed_resp = call_with_retry(
                        litellm.embedding,
                        model=self.embedding_model,
                        input=[missing_item],
                    )
                    vec = embed_resp["data"][0]["embedding"]
                    self._embedding_cache.set(missing_item, vec)

                new_contexts = await self._execute_vector_search(vec)
                all_contexts.extend(new_contexts)

            answer = await self._synthesize(query, all_contexts)

        result = RAGResponse(
            query=query,
            retrieved_contexts=all_contexts,
            generated_answer=answer,
            query_embedding=primary_vector,
            reflection_iterations=reflection_iterations,
            final_confidence=final_confidence,
            metadata={"sub_queries_generated": sub_queries, "token_usage": {}},
        )
        return result
