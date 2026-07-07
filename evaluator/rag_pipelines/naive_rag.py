import logging
import litellm
from typing import List
from evaluator.config import config
from evaluator.rag_pipelines.base import BaseRAGPipeline, RAGResponse

logger = logging.getLogger("NaiveRAG")

class NaiveRAG(BaseRAGPipeline):
    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        self.model_name = model_name
        self.embedding_model = config.EMBEDDING_MODEL

    def _mock_pgvector_retrieval(self, embedding: List[float], k: int = 3) -> List[str]:
        """Simulates an exact L2/Cosine distance match from a pgvector database instance."""
        logger.info(f"Simulating pgvector Top-{k} retrieval matching against vector space.")
        return [
            "Clinical protocol payload: Patient eligibility relies on strict physiological boundaries.",
            "Database Context: Maximum allowable budget ceiling for campaign cluster Alpha is $50,000.",
            "System Constraint: Direct neural generations must bypass unvetted transactional state commits."
        ]

    def execute(self, query: str) -> RAGResponse:
        embed_resp = litellm.embedding(model=self.embedding_model, input=[query])
        query_vector = embed_resp['data'][0]['embedding']

        contexts = self._mock_pgvector_retrieval(query_vector)

        formatted_context = "\n".join(contexts)
        prompt = f"Answer the query using ONLY the context provided.\nContext:\n{formatted_context}\n\nQuery: {query}"

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
