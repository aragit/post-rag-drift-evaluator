import numpy as np
import hashlib
from typing import Any, List


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockCompletionResponse:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]
        self._usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def get(self, key: str, default: Any = None) -> Any:
        return self._usage if key == "usage" else default


def is_mock_key(key: str | None) -> bool:
    """Check if API key is a placeholder that should trigger mock mode."""
    if key is None:
        return True
    return key.startswith("sk-your-") or key.startswith("sk-mock-key")


def generate_mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """Generate a deterministic mock embedding based on text hash.
    
    Uses MD5 hash of input text as seed for reproducible embeddings.
    Returns normalized random vector that is deterministic for same input.
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()
    seed = int(text_hash[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def generate_mock_completion(prompt: str, response_format: str | None = None) -> MockCompletionResponse:
    """Generate a deterministic mock completion response.

    Returns a structure matching litellm.completion() output, bypassing
    any live API call when a placeholder key is detected.
    """
    if response_format == "json":
        content = (
            '["What are the eligibility criteria for protocol Alpha?",'
            '"What are the budget constraints for Q3 operations?"]'
        )
    else:
        content = (
            "Based on the provided context, this is a synthesized mock response "
            "generated for offline functional verification."
        )
    return MockCompletionResponse(content)


def generate_random_embedding(dim: int = 1536) -> List[float]:
    """Generate a random mock embedding (non-deterministic)."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()
