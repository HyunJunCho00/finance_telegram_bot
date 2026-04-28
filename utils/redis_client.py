import json
import logging
from typing import Any, Optional

import redis
from config.settings import settings

logger = logging.getLogger(__name__)

class RedisManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        self.enabled = False
        self.client = None
        if hasattr(settings, "REDIS_URL") and settings.REDIS_URL:
            try:
                self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis connected successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed (will use memory fallback): {e}")

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ex: Optional[int] = None):
        if not self.enabled:
            return
        try:
            self.client.set(key, value, ex=ex)
        except Exception:
            pass

    def get_json(self, key: str) -> Optional[Any]:
        val = self.get(key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ex: Optional[int] = None):
        self.set(key, json.dumps(value), ex=ex)

redis_client = RedisManager()

def redis_cache(ttl_seconds: int = 60, key_prefix: str = ""):
    """Decorator to cache function results in Redis."""
    def decorator(func):
        import functools
        import hashlib
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not redis_client.enabled:
                return func(*args, **kwargs)
            
            # Create a unique cache key based on function name and arguments
            sig = f"{func.__name__}:{args}:{kwargs}"
            key_hash = hashlib.md5(sig.encode()).hexdigest()
            cache_key = f"cache:{key_prefix or func.__name__}:{key_hash}"
            
            cached = redis_client.get_json(cache_key)
            if cached is not None:
                # If it's a pandas dataframe serialized as json, we need to handle it.
                # But for simple dicts/lists this is perfect.
                return cached
                
            result = func(*args, **kwargs)
            
            # Don't cache empty results or None
            if result is not None:
                # pandas DataFrame needs special handling if used, but for general dict/list it's fine
                try:
                    redis_client.set_json(cache_key, result, ex=ttl_seconds)
                except TypeError:
                    pass # Not JSON serializable
            return result
        return wrapper
    return decorator
