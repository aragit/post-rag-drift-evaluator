from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    query_embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseRAGPipeline(ABC):
    @abstractmethod
    def execute(self, query: str) -> RAGResponse:
        """Executes the full retrieval and generation loop."""
        pass
