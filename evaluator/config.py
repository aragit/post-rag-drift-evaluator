import os
from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluatorConfig(BaseSettings):
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="rag_db")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)

    DATABASE_URL: str | None = Field(default=None)

    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")
    WORKER_BATCH_SIZE: int = Field(default=50)
    WORKER_FLUSH_INTERVAL: float = Field(default=5.0)

    LITELLM_MASTER_KEY: str = Field(default="sk-mock-key-1234")
    DEFAULT_MODEL: str = Field(default="gemma-3n-it")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    LITELLM_TIMEOUT: float = Field(default=30.0)

    DB_POOL_MIN_SIZE: int = Field(default=2)
    DB_POOL_MAX_SIZE: int = Field(default=10)
    DB_POOL_MAX_QUERIES: int = Field(default=50000)
    MAX_REFLECTION_ITERATIONS: int = Field(default=2)
    METRICS_PORT: int = Field(default=8000)

    DRIFT_THRESHOLD: float = Field(default=0.15)
    MMD_THRESHOLD: float = Field(default=0.1)
    PER_COMPONENT_KL_THRESHOLD: float = Field(default=0.5)

    DEFAULT_BASELINE_WINDOW_HOURS: int = Field(default=24)
    DYNAMIC_THRESHOLD_K_SIGMA: float = Field(default=2.0)
    MIN_BASELINE_FRAMES: int = Field(default=20)

    DRIFT_ALERT_WEBHOOK_URL: str | None = Field(default=None)

    REDIS_URL: Optional[str] = Field(default=None)
    REDIS_STREAM_KEY: str = Field(default="telemetry:frames:stream")
    REDIS_CONSUMER_GROUP: str = Field(default="drift_engine_workers")
    REDIS_CONSUMER_NAME: str = Field(default="worker_1")
    EMBEDDING_CACHE_TTL: int = Field(default=3600)
    RESULT_CACHE_TTL: int = Field(default=1800)

    OPENAI_API_KEY: str | None = Field(default=None)
    GEMINI_API_KEY: str | None = Field(default=None)
    ANTHROPIC_API_KEY: str | None = Field(default=None)

    API_KEY_REQUIRED: bool = Field(default=False)
    API_KEYS: List[str] = Field(default_factory=list)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    RATE_LIMIT_PER_MINUTE: int = Field(default=600)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @model_validator(mode="before")
    @classmethod
    def _assemble_database_url(cls, data: Any) -> Any:
        """Construct DATABASE_URL from individual POSTGRES_* vars if unset.

        Preserves backward compatibility: an explicit ``DATABASE_URL``
        environment variable always wins.
        """
        if isinstance(data, dict) and not data.get("DATABASE_URL"):
            user = data.get("POSTGRES_USER", "postgres")
            pw = data.get("POSTGRES_PASSWORD", "postgres")
            db = data.get("POSTGRES_DB", "rag_db")
            host = data.get("POSTGRES_HOST", "localhost")
            port = data.get("POSTGRES_PORT", 5432)
            data["DATABASE_URL"] = (
                f"postgresql://{user}:{pw}@{host}:{port}/{db}"
            )
        return data

    @field_validator("API_KEYS", "CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_list_fields(cls, v: Any) -> Any:
        """Parse comma-separated environment strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if v is None:
            return []
        return v or []

    @property
    def LOG_LEVEL_INT(self) -> int:
        """Resolve ``LOG_LEVEL`` to a numeric ``logging`` constant."""
        levels = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        return levels.get(self.LOG_LEVEL.upper(), 20)


config = EvaluatorConfig()

# Sync resolved provider keys into os.environ so litellm can discover them.
_provider_keys = {
    "OPENAI_API_KEY": config.OPENAI_API_KEY,
    "GEMINI_API_KEY": config.GEMINI_API_KEY,
    "ANTHROPIC_API_KEY": config.ANTHROPIC_API_KEY,
}
for key, value in _provider_keys.items():
    if value is not None:
        os.environ[key] = value

# Fallback: if OPENAI_API_KEY is empty but LITELLM_MASTER_KEY is set, map it over
# so OpenAI-compatible endpoints work out of the box.
if not os.environ.get("OPENAI_API_KEY") and config.LITELLM_MASTER_KEY:
    os.environ["OPENAI_API_KEY"] = config.LITELLM_MASTER_KEY
