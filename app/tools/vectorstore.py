from functools import lru_cache
from typing import Any

from core.config import settings
from core.llm import get_embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse:
    """Cached singleton for FastEmbed BM25 sparse model to prevent reload overhead."""
    return FastEmbedSparse(model_name="Qdrant/bm25")


def get_qdrant_client() -> QdrantClient:
    """Returns a client connection to Qdrant."""
    return QdrantClient(url=settings.qdrant_url, prefer_grpc=False)


def insert_hybrid_indexes(chunks: list[str], metadata: list[dict[str, Any]]) -> bool:
    """Creates dense and sparse embeddings and stores them in Qdrant with hybrid indexing."""
    if not chunks:
        return True

    dense_embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    target_dim = getattr(settings, "embedding_dimension", 1024)

    # Check if collection exists and whether its vector dimensions match the current model
    if client.collection_exists(collection_name):
        try:
            col_info = client.get_collection(collection_name)
            vectors_cfg = col_info.config.params.vectors
            dense_cfg = None
            if isinstance(vectors_cfg, dict):
                dense_cfg = vectors_cfg.get("dense")
            elif hasattr(vectors_cfg, "dense"):
                dense_cfg = vectors_cfg.dense

            if dense_cfg and getattr(dense_cfg, "size", None) != target_dim:
                print(
                    f"Dimension mismatch in Qdrant collection '{collection_name}' "
                    f"(found {getattr(dense_cfg, 'size', None)}, expected {target_dim}). "
                    f"Re-creating collection for model '{settings.embedding_model}'..."
                )
                client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not check collection dimension: {e}")

    # Create collection if not present
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=target_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )

    # Ingest using hybrid retrieval mode
    QdrantVectorStore.from_texts(
        texts=chunks,
        metadatas=metadata,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=settings.qdrant_url,
        prefer_grpc=False,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    return True


# Alias for backward-compatibility with ingestion nodes referencing chroma
def insert_into_chroma(chunks: list[str], metadata: list[dict[str, Any]]) -> bool:
    """Compatibility bridge directing ingestion to the unified Qdrant store."""
    return insert_hybrid_indexes(chunks=chunks, metadata=metadata)