import hashlib
import json
import logging
from typing import TYPE_CHECKING, Optional

import redis

from evaluator.config import config
from evaluator.metrics_exporter import record_cache_hit, record_cache_miss

if TYPE_CHECKING:
    from evaluator.rag_pipelines.base import RAGResponse

logger = logging.getLogger("Cache")


def _make_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


class EmbeddingCache:
    def __init__(
        self, redis_url: str = config.REDIS_URL, ttl: int = config.EMBEDDING_CACHE_TTL
    ):
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Optional[redis.Redis] = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def get(self, query_text: str) -> Optional[list[float]]:
        key = _make_key("embedding", query_text)
        try:
            raw = self._get_client().get(key)
            if raw is None:
                record_cache_miss("embedding")
                return None
            record_cache_hit("embedding")
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"EmbeddingCache get failed: {e}")
            record_cache_miss("embedding")
            return None

    def set(self, query_text: str, embedding: list[float]) -> None:
        key = _make_key("embedding", query_text)
        try:
            self._get_client().setex(key, self._ttl, json.dumps(embedding))
        except Exception as e:
            logger.warning(f"EmbeddingCache set failed: {e}")


class ResultCache:
    def __init__(
        self, redis_url: str = config.REDIS_URL, ttl: int = config.RESULT_CACHE_TTL
    ):
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Optional[redis.Redis] = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def get(self, query_text: str, pipeline_name: str) -> Optional["RAGResponse"]:
        key = _make_key("result", query_text, pipeline_name)
        try:
            raw = self._get_client().get(key)
            if raw is None:
                record_cache_miss("result")
                return None
            record_cache_hit("result")
            data = json.loads(raw)
            from evaluator.rag_pipelines.base import RAGResponse

            return RAGResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"ResultCache get failed: {e}")
            record_cache_miss("result")
            return None

    def set(self, query_text: str, pipeline_name: str, response: "RAGResponse") -> None:
        key = _make_key("result", query_text, pipeline_name)
        try:
            self._get_client().setex(key, self._ttl, response.model_dump_json())
        except Exception as e:
            logger.warning(f"ResultCache set failed: {e}")
