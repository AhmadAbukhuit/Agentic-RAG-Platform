import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "agents-service")
    SERVICE_DESCRIPTION: str = "Agentic RAG & Multi-Agent Orchestration Service"
    SERVICE_VERSION: str = "1.0.0"

    # LangSmith / LangChain Telemetry
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "agentic-rag-service"

    # Ollama / Local Models
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "llama3"
    vision_model: str = "deepseek-ocr"
    embedding_model: str = "nomic-embed-text"

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
    allowed_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate once to be imported anywhere
settings = Settings()