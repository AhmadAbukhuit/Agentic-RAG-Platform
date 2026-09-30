import contextlib
import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "agents-service")
    SERVICE_DESCRIPTION: str = "Agentic RAG & Multi-Agent Orchestration Service"
    SERVICE_VERSION: str = "1.0.0"

    # LangSmith / LangChain Telemetry
    langsmith_tracing: str = "true"
    langsmith_endpoint: str = "https://eu.api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-rag-platform"

    langchain_tracing_v2: str = ""
    langchain_endpoint: str = ""
    langchain_api_key: str = ""
    langchain_project: str = ""

    # Optional External Model Providers (Only needed if switching from local Ollama)
    openai_api_key: str = ""

    @property
    def effective_langsmith_key(self) -> str:
        """Returns the active LangSmith API key from either new or legacy setting."""
        return self.langsmith_api_key or self.langchain_api_key

    @property
    def effective_langsmith_endpoint(self) -> str:
        """Returns the active LangSmith API endpoint (EU or US)."""
        return self.langsmith_endpoint or self.langchain_endpoint or "https://eu.api.smith.langchain.com"

    @property
    def effective_langsmith_project(self) -> str:
        """Returns the target LangSmith project name."""
        return self.langsmith_project or self.langchain_project or "agentic-rag-platform"

    @property
    def is_langsmith_tracing_enabled(self) -> bool:
        """Checks if tracing is explicitly enabled in configuration."""
        flag = (self.langsmith_tracing or self.langchain_tracing_v2 or "true").lower()
        return flag in ["true", "1", "yes"]

    # Ollama / Local Models
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "llama3"
    vision_model: str = "deepseek-ocr"
    embedding_model: str = "zylonai/multilingual-e5-large"
    embedding_dimension: int = 1024

    # Vector Stores
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_name: str = "envision_docs"
    chroma_db_url: str = "http://chromadb:8000"

    # Distributed Redis Cache
    redis_url: str = "redis://redis:6379"
    redis_cache_ttl: int = 86400  # Default TTL: 24 hours in seconds
    enable_redis_cache: bool = True

    # Document / NoSQL Store
    mongodb_url: str = "mongodb://mongodb:27017"
    mongodb_database: str = "rag_db"
    mongodb_collection: str = "raw_documents"

    # Security & API
    app_api_key: str = ""
    allowed_origins: str | list[str] = ["*"]

    @property
    def cors_origins(self) -> list[str]:
        """Returns allowed origins as a parsed list of string patterns."""
        if isinstance(self.allowed_origins, str):
            v_clean = self.allowed_origins.strip()
            if v_clean.startswith("[") and v_clean.endswith("]"):
                with contextlib.suppress(Exception):
                    return json.loads(v_clean)
            return [origin.strip() for origin in v_clean.split(",") if origin.strip()]
        return self.allowed_origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate once to be imported anywhere
settings = Settings()