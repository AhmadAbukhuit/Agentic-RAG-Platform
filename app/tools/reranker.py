from functools import lru_cache

try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker = None
    RerankRequest = None


@lru_cache(maxsize=1)
def get_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> object | None:
    """Cached singleton for FlashRank Cross-Encoder model (runs on CPU via ONNX)."""
    if Ranker is None:
        return None
    try:
        return Ranker(model_name=model_name, cache_dir="/tmp/flashrank_cache")
    except Exception as e:
        print(f"Warning: Failed to initialize FlashRank Ranker: {e}")
        return None


def rerank_passages(query: str, passages: list[str], top_n: int = 4) -> list[str]:
    """Re-ranks retrieved passage chunks using a cross-encoder model to maximize relevance.
    
    Args:
        query: The user query string.
        passages: List of retrieved text chunks from the vector store / web search.
        top_n: Number of top re-ranked passages to return.
    """
    if not passages or not query.strip():
        return passages[:top_n]

    ranker = get_ranker()
    if ranker is not None and RerankRequest is not None:
        try:
            # Prepare passages formatted for FlashRank
            formatted_passages = [
                {"id": i, "text": passage}
                for i, passage in enumerate(passages)
            ]
            rerank_req = RerankRequest(query=query, passages=formatted_passages)
            ranked_results = ranker.rerank(rerank_req)

            # Return top_n text bodies ordered by cross-encoder score
            return [res["text"] for res in ranked_results[:top_n]]
        except Exception as e:
            print(f"FlashRank re-ranking failed ({e}). Falling back to original retrieval order.")

    # Graceful fallback: return top_n based on original order
    return passages[:top_n]
