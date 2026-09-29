from functools import lru_cache

from core.config import settings
from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatOllama:
    """Returns the primary Ollama chat model for reasoning, routing, and synthesis."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.llm_model,
        temperature=temperature,
    )


def get_vision_llm(temperature: float = 0.0, model: str | None = None) -> ChatOllama:
    """Returns the Ollama multimodal model for vision and OCR tasks."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.vision_model,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def get_embeddings(model: str | None = None) -> OllamaEmbeddings:
    """Returns cached Ollama embedding model for dense vector search."""
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=model or settings.embedding_model,
    )