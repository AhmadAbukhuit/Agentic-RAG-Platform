"""Master Agentic RAG Orchestration Graph.

This module coordinates the complete lifecycle of incoming user queries:
1. Long-term memory recall (user preferences and durable facts).
2. Distributed Redis cache lookup (instant sub-5ms response on hits).
3. Query routing (vectorstore vs web search).
4. Subgraph execution (researcher with CRAG grading and FlashRank re-ranking).
5. Actor-Critic synthesis (drafter) and hallucination grading (reviewer).
6. Conditional self-correction loop (max retries bounded).
7. Cache saving and memory persistence before output finalization.
"""

from graphs.subgraphs.researcher import researcher_app
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from nodes.cache_nodes import check_cache, save_cache
from nodes.generator import draft_response, review_response
from nodes.memory_nodes import persist_memory, recall_memory
from nodes.router import route_question
from state.schema import AgentState


def route_cache(state: AgentState) -> str:
    """Routes directly to memory persistence if served from cache; otherwise proceeds to router."""
    if state.get("cache_hit"):
        print("---BYPASSING AGENT PIPELINE (Cache Hit)---")
        return "cache_hit"
    return "cache_miss"


def should_revise(state: AgentState) -> str:
    """Decides whether to route back to drafter for critique-driven revision or proceed to cache and finalize."""
    status = state.get("review_status", "approved")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if status == "revision_needed" and retry_count < max_retries:
        print(f"---CRITIQUE TRIGGERED: Routing back to drafter (Revision #{retry_count})---")
        return "drafter"

    print("---QUALITY APPROVED: Proceeding to cache persistence and final output---")
    return "finalize"


# 1. Initialize Master Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("recall_memory", recall_memory)
workflow.add_node("check_cache", check_cache)
workflow.add_node("router", route_question)
workflow.add_node("research_team", researcher_app)
workflow.add_node("drafter", draft_response)
workflow.add_node("reviewer", review_response)
workflow.add_node("save_cache", save_cache)
workflow.add_node("persist_memory", persist_memory)

# 3. Define Master Graph Flow with Memory & Caching
# START -> recall_memory -> check_cache
workflow.add_edge(START, "recall_memory")
workflow.add_edge("recall_memory", "check_cache")

# Conditional Edge from check_cache:
# Hit  -> persist_memory -> END
# Miss -> router -> research_team -> drafter -> reviewer
workflow.add_conditional_edges(
    "check_cache",
    route_cache,
    {
        "cache_hit": "persist_memory",
        "cache_miss": "router"
    }
)

workflow.add_edge("router", "research_team")
workflow.add_edge("research_team", "drafter")
workflow.add_edge("drafter", "reviewer")

# Conditional Edge from reviewer:
# Revision -> drafter
# Finalize -> save_cache -> persist_memory -> END
workflow.add_conditional_edges(
    "reviewer",
    should_revise,
    {
        "drafter": "drafter",
        "finalize": "save_cache"
    }
)

workflow.add_edge("save_cache", "persist_memory")
workflow.add_edge("persist_memory", END)

# 4. Compile Master Graph with in-memory checkpointer for multi-turn thread support
checkpointer = MemorySaver()
master_app = workflow.compile(checkpointer=checkpointer)