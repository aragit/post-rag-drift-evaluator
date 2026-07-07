import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class EvaluatorConfig(BaseSettings):
    DATABASE_URL: str = Field(default="postgresql://postgres:postgres@localhost:5432/rag_db")
    LITELLM_MASTER_KEY: str = Field(default="sk-mock-key-1234")
    DEFAULT_MODEL: str = Field(default="gemma-3n-it")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

config = EvaluatorConfig()
