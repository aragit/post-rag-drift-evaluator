import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class EvaluatorConfig(BaseSettings):
    DATABASE_URL: str = Field(default="postgresql://postgres:postgres@localhost:5432/rag_db")
    LITELLM_MASTER_KEY: str = Field(default="sk-mock-key-1234")
    DEFAULT_MODEL: str = Field(default="gemma-3n-it")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

    OPENAI_API_KEY: str | None = Field(default=None)
    GEMINI_API_KEY: str | None = Field(default=None)
    ANTHROPIC_API_KEY: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

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
