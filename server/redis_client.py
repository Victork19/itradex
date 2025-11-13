# server/redis_client.py
from redis.asyncio import Redis  
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any
from fastapi import Depends
import logging

logger = logging.getLogger(__name__)

# Global pool (managed via lifespan)
_redis_pool: Redis | None = None

async def init_redis(redis_url: str = "redis://localhost:6379") -> None:
    """
    Initialize the Redis connection pool.
    Call this in the app's lifespan startup event.
    """
    global _redis_pool
    if _redis_pool is not None:
        raise ValueError("Redis pool already initialized")
    _redis_pool = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)  # UPDATED: Redis.from_url
    # NEW: Test connection
    try:
        await _redis_pool.ping()
        logger.info(f"Redis connected successfully at {redis_url}")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

async def close_redis() -> None:
    """
    Close the Redis connection pool.
    Call this in the app's lifespan shutdown event.
    """
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None

@asynccontextmanager
async def get_redis() -> AsyncGenerator[Redis, None]:  # UPDATED: Type to Redis
    """
    Async context manager for getting a Redis client.
    Ensures the pool is available and provides a client.
    """
    global _redis_pool
    if _redis_pool is None:
        raise ValueError("Redis not initialized. Call init_redis() first.")
    async with _redis_pool as redis:  # Works the same
        yield redis

# Dependency for FastAPI
async def redis_dependency() -> Redis:  # UPDATED: Type to Redis
    """
    FastAPI dependency to inject Redis client.
    Use as: redis_client: Redis = Depends(redis_dependency)
    """
    global _redis_pool
    if _redis_pool is None:
        raise ValueError("Redis not initialized.")
    return _redis_pool

# Helper functions for common operations (unchanged; API compatible)
async def get_cache(redis: Redis, key: str, default: Any = None) -> Any:  # UPDATED: Type to Redis
    """Get a value from cache, return default if missing."""
    try:
        value = await redis.get(key)
        return value if value is not None else default
    except Exception as e:
        logger.warning(f"Redis get failed for {key}: {e}. Returning default.")
        return default

async def set_cache(redis: Redis, key: str, value: Any, ttl: int = 3600) -> None:  # UPDATED: Type to Redis
    """Set a value in cache with optional TTL (seconds)."""
    try:
        await redis.set(key, value, ex=ttl)
    except Exception as e:
        logger.warning(f"Redis set failed for {key}: {e}")

async def delete_cache(redis: Redis, key: str) -> None:  # UPDATED: Type to Redis
    """Delete a key from cache."""
    try:
        await redis.delete(key)
    except Exception as e:
        logger.warning(f"Redis delete failed for {key}: {e}")

async def incr_cache(redis: Redis, key: str, amount: int = 1) -> int:  # UPDATED: Type to Redis
    """Increment a counter in cache, return new value."""
    try:
        return await redis.incr(key, amount)
    except Exception as e:
        logger.warning(f"Redis incr failed for {key}: {e}")
        return amount  # Fallback to just amount if incr fails