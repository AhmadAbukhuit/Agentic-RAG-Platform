from state.schema import AgentState
from tools.reranker import rerank_passages


def rerank_documents(state: AgentState) -> dict:
    """LangGraph node that applies FlashRank cross-encoder scoring to filter retrieved context."""
    question = state.get("question", "")
    documents = state.get("documents", [])

    if not documents or not question:
        return {}

    print(f"---CROSS-ENCODER RE-RANKING: Scoring {len(documents)} candidates with FlashRank---")
    
    # Re-rank and keep top 4 most relevant chunks
    ranked_docs = rerank_passages(query=question, passages=documents, top_n=4)
    print(f"---FILTERED DOWN TO TOP {len(ranked_docs)} RE-RANKED CHUNKS---")

    return {"documents": ranked_docs}
