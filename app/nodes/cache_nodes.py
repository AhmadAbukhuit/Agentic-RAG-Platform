from core.cache import cache_manager
from state.schema import AgentState


def check_cache(state: AgentState) -> dict:
    """Checks Redis cache for an existing response to the user query before running the graph."""
    question = state.get("question", "")
    if not question:
        return {"cache_hit": False}

    cached = cache_manager.get(question)
    if cached and "final_answer" in cached:
        print(f"---REDIS CACHE HIT for: '{question[:50]}'---")
        return {
            "cache_hit": True,
            "final_answer": cached["final_answer"],
            "datasource": cached.get("datasource", "cache"),
            "review_status": "approved",
            "draft_answer": cached["final_answer"],
        }

    print(f"---REDIS CACHE MISS for: '{question[:50]}'---")
    return {"cache_hit": False}


def save_cache(state: AgentState) -> dict:
    """Persists newly synthesized and reviewed final response into Redis cache."""
    if state.get("cache_hit"):
        return {}

    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    datasource = state.get("datasource", "vectorstore")

    if question and final_answer:
        print(f"---PERSISTING TO REDIS CACHE for: '{question[:50]}'---")
        cache_manager.set(
            query=question,
            data={
                "final_answer": final_answer,
                "datasource": datasource,
            }
        )

    return {}
