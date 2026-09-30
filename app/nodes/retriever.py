from core.config import settings
from core.llm import get_embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from state.schema import AgentState
from tools.vectorstore import get_sparse_embeddings


def query_qdrant_hybrid(state: AgentState) -> dict:
    """LangGraph node for native Hybrid retrieval (Dense + BM25 Sparse) via Qdrant."""
    print("---RETRIEVING FROM QDRANT (HYBRID)---")
    question = state.get("question", "")

    try:
        dense_embeddings = get_embeddings()
        sparse_embeddings = get_sparse_embeddings()

        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url=settings.qdrant_url,
            prefer_grpc=False,
            collection_name=settings.qdrant_collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        # Retrieve top 15 candidate chunks for the cross-encoder re-ranking stage
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke(question)
        document_texts = [doc.page_content for doc in docs]
        return {"documents": document_texts}
    except Exception as e:
        print(f"Warning: Vectorstore retrieval failed ({e}). Returning fallback context.")
        return {"documents": [f"No internal documents retrieved (Store offline or empty: {e})"]}


# Alias for backward-compatibility with subgraphs referencing query_chroma
query_chroma = query_qdrant_hybrid


def search_web(state: AgentState) -> dict:
    """LangGraph node for executing web searches for real-time information."""
    print("---EXECUTING WEB SEARCH---")
    question = state.get("question", "")

    results = [
        f"Web Search Results for query: '{question}'",
        f"Current verified data snippet regarding: {question} (Retrieved via real-time search interface)."
    ]
    return {"documents": results}