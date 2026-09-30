from datetime import UTC, datetime

from core.config import settings

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

# In-memory fallback dictionary for unit testing or when MongoDB is unreachable
_LOCAL_USER_MEMORIES: dict[str, list[str]] = {}


def get_memory_collection():
    """Returns the MongoDB collection for long-term user memories."""
    if MongoClient is None:
        return None
    try:
        client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        db = client[settings.mongodb_database]
        return db["user_memories"]
    except Exception:
        return None


def recall_user_memories(user_id: str | None, limit: int = 5) -> list[str]:
    """Retrieves long-term preferences and facts associated with the user."""
    if not user_id:
        return []

    # 1. Try MongoDB
    col = get_memory_collection()
    if col is not None:
        try:
            cursor = col.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
            return [doc["memory"] for doc in cursor if "memory" in doc]
        except Exception as e:
            print(f"MongoDB memory retrieval error: {e}")

    # 2. Try in-memory store
    return list(_LOCAL_USER_MEMORIES.get(user_id, []))[:limit]


def save_user_memory(user_id: str | None, memory_text: str) -> bool:
    """Stores a long-term preference or fact for a user."""
    if not user_id or not memory_text.strip():
        return False

    clean_memory = memory_text.strip()

    # 1. Try MongoDB
    col = get_memory_collection()
    if col is not None:
        try:
            col.update_one(
                {"user_id": user_id, "memory": clean_memory},
                {"$setOnInsert": {
                    "user_id": user_id,
                    "memory": clean_memory,
                    "created_at": datetime.now(UTC).isoformat()
                }},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"MongoDB memory save error: {e}")

    # 2. Store in local in-memory fallback
    if user_id not in _LOCAL_USER_MEMORIES:
        _LOCAL_USER_MEMORIES[user_id] = []
    if clean_memory not in _LOCAL_USER_MEMORIES[user_id]:
        _LOCAL_USER_MEMORIES[user_id].append(clean_memory)

    return True


def clear_user_memories(user_id: str | None) -> bool:
    """Clears long-term memories for a user."""
    if not user_id:
        return False

    col = get_memory_collection()
    if col is not None:
        try:
            col.delete_many({"user_id": user_id})
        except Exception as e:
            print(f"MongoDB memory clear error: {e}")

    _LOCAL_USER_MEMORIES.pop(user_id, None)
    return True
