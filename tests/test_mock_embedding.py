import numpy as np
from evaluator.utils.mock_embedding import is_mock_key, generate_mock_embedding, generate_random_embedding


def test_is_mock_key_with_none():
    assert is_mock_key(None) is True


def test_is_mock_key_with_placeholder():
    assert is_mock_key("sk-your-api-key") is True
    assert is_mock_key("sk-mock-key-1234") is True


def test_is_mock_key_with_real_key():
    assert is_mock_key("sk-real-openai-key-1234") is False
    assert is_mock_key("sk-proj-abc123") is False


def test_generate_mock_embedding_deterministic():
    vec1 = generate_mock_embedding("test input")
    vec2 = generate_mock_embedding("test input")
    assert np.allclose(vec1, vec2)


def test_generate_mock_embedding_different_inputs():
    vec1 = generate_mock_embedding("input one")
    vec2 = generate_mock_embedding("input two")
    assert not np.allclose(vec1, vec2)


def test_generate_mock_embedding_dimension():
    vec = generate_mock_embedding("test")
    assert len(vec) == 1536


def test_generate_mock_embedding_normalized():
    vec = generate_mock_embedding("test")
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_generate_random_embedding_dimension():
    vec = generate_random_embedding()
    assert len(vec) == 1536


def test_generate_random_embedding_normalized():
    vec = generate_random_embedding()
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_generate_random_embedding_unique():
    vec1 = generate_random_embedding()
    vec2 = generate_random_embedding()
    assert not np.allclose(vec1, vec2)
