import numpy as np
import hashlib
from typing import List


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


def generate_random_embedding(dim: int = 1536) -> List[float]:
    """Generate a random mock embedding (non-deterministic)."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()
