"""Researcher Subgraph: Retrieval, Corrective RAG (CRAG), and Cross-Encoder Re-ranking.

This subgraph encapsulates the entire information-gathering phase:
1. Routing to either internal Qdrant hybrid retrieval or live web search.
2. If vector retrieval is chosen, retrieved candidates are graded for relevance (CRAG).
3. If relevant, documents proceed to cross-encoder re-ranking.
4. If irrelevant, query is rewritten and web search is triggered as a fallback.
5. All candidate passages are distilled by the FlashRank re-ranking node down to
   the top 4 highest-precision chunks before handing off to the master synthesis graph.
"""

from langgraph.graph import END, START, StateGraph
from nodes.crag_nodes import grade_documents, rewrite_query
from nodes.reranker import rerank_documents
from nodes.retriever import query_qdrant_hybrid, search_web
from state.schema import AgentState


def route_retrieval(state: AgentState) -> str:
    """Conditional edge router based on initial datasource decision."""
    datasource = state.get("datasource", "vectorstore")
    if datasource == "web_search":
        return "web_search"
    return "vector_search"


def decide_crag_action(state: AgentState) -> str:
    """Corrective RAG edge: checks document grade to either proceed to re-ranking or trigger web search fallback."""
    grade = state.get("retrieval_grade", "relevant")
    if grade == "irrelevant":
        print("---CRAG ACTION: Internal context irrelevant. Triggering query rewrite & web search---")
        return "rewrite_query"
    print("---CRAG ACTION: Internal context relevant. Proceeding to cross-encoder re-ranking---")
    return "rerank_documents"


# 1. Initialize Subgraph
research_builder = StateGraph(AgentState)

# 2. Add retrieval, CRAG correction, and re-ranking nodes
research_builder.add_node("vector_search", query_qdrant_hybrid)
research_builder.add_node("grade_documents", grade_documents)
research_builder.add_node("rewrite_query", rewrite_query)
research_builder.add_node("web_search", search_web)
research_builder.add_node("reranker", rerank_documents)

# 3. Define Retrieval & CRAG Routing
# Entrypoint: route to initial datasource
research_builder.add_conditional_edges(
    START,
    route_retrieval,
    {
        "vector_search": "vector_search",
        "web_search": "web_search"
    }
)

# Vector search -> Grade retrieved documents
research_builder.add_edge("vector_search", "grade_documents")

# Grade evaluation conditional edge:
# Relevant   -> reranker
# Irrelevant -> rewrite_query -> web_search -> reranker
research_builder.add_conditional_edges(
    "grade_documents",
    decide_crag_action,
    {
        "rerank_documents": "reranker",
        "rewrite_query": "rewrite_query"
    }
)

research_builder.add_edge("rewrite_query", "web_search")
research_builder.add_edge("web_search", "reranker")
research_builder.add_edge("reranker", END)

# 4. Compile the Subgraph
researcher_app = research_builder.compile()