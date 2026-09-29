import hashlib
import json
from typing import Any

from core.config import settings

try:
    import redis
except ImportError:
    redis = None

# In-memory LRU/dict fallback if Redis is unreachable or not installed locally
_LOCAL_FALLBACK_CACHE: dict[str, str] = {}


class RedisCacheManager:
    """Manages exact and semantic distributed caching backed by Redis with local fallback."""

    def __init__(self, redis_url: str = settings.redis_url, ttl: int = settings.redis_cache_ttl):
        self.redis_url = redis_url
        self.ttl = ttl
        self._client: Any | None = None

    @property
    def client(self):
        """Lazy initialization of the Redis client."""
        if not settings.enable_redis_cache or redis is None:
            return None

        if self._client is None:
            try:
                self._client = redis.Redis.from_url(
                    self.redis_url,
                    socket_connect_timeout=1.5,
                    socket_timeout=1.5,
                    decode_responses=True
                )
                # Quick ping to verify connectivity
                self._client.ping()
            except Exception as e:
                # Log warning once and fallback gracefully
                print(f"Warning: Redis cache unreachable at {self.redis_url} ({e}). Using in-memory fallback cache.")
                self._client = None

        return self._client

    @staticmethod
    def hash_key(query: str, namespace: str = "rag:response") -> str:
        """Generates a deterministic SHA256 key from a normalized user query."""
        normalized = query.strip().lower()
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{namespace}:{query_hash}"

    def get(self, query: str) -> dict[str, Any] | None:
        """Retrieves cached response dict for a query if it exists."""
        if not settings.enable_redis_cache:
            return None

        key = self.hash_key(query)

        # 1. Try Redis
        client = self.client
        if client:
            try:
                raw = client.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                print(f"Redis GET failed for key {key}: {e}")

        # 2. Try In-Memory Fallback
        if key in _LOCAL_FALLBACK_CACHE:
            try:
                return json.loads(_LOCAL_FALLBACK_CACHE[key])
            except Exception:
                return None

        return None

    def set(self, query: str, data: dict[str, Any], ttl: int | None = None) -> bool:
        """Stores response dict in cache with TTL."""
        if not settings.enable_redis_cache:
            return False

        key = self.hash_key(query)
        payload = json.dumps(data)
        expiry = ttl or self.ttl

        # 1. Try Redis
        client = self.client
        if client:
            try:
                client.set(key, payload, ex=expiry)
                return True
            except Exception as e:
                print(f"Redis SET failed for key {key}: {e}")

        # 2. Store in local fallback
        _LOCAL_FALLBACK_CACHE[key] = payload
        return True

    def clear(self) -> bool:
        """Flushes the local fallback cache and namespace keys."""
        _LOCAL_FALLBACK_CACHE.clear()
        client = self.client
        if client:
            try:
                keys = client.keys("rag:response:*")
                if keys:
                    client.delete(*keys)
                return True
            except Exception as e:
                print(f"Redis CLEAR failed: {e}")
        return True


# Global singleton instance
cache_manager = RedisCacheManager()
