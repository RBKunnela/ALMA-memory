"""
ALMA Consolidation Rate Limiting and Caching.

Implements:
- Fix 1.3: Rate limiting on LLM API calls (prevents budget explosion)
- Fix 1.4: Bounded LRU cache (prevents memory leaks)

This fixes the issues where:
- Unlimited LLM calls could quickly exhaust API budgets
- Unbounded cache could grow to GB+ sizes
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional, TypeVar

logger_available = True
try:
    import logging

    logger = logging.getLogger(__name__)
except ImportError:
    logger_available = False

T = TypeVar("T")


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.

    Prevents more than N calls per M seconds.
    """

    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter.

        Args:
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.calls_made = 0
        self.window_start = time.time()

    def acquire(self) -> bool:
        """
        Check if call is allowed.

        Returns:
            True if call allowed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.window_start

        # Reset window if period has elapsed
        if elapsed >= self.period:
            self.calls_made = 0
            self.window_start = now
            elapsed = 0

        if self.calls_made < self.calls:
            self.calls_made += 1
            return True

        # Calculate wait time
        wait_time = self.period - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
            # Recursive call after waiting
            return self.acquire()

        return False

    async def acquire_async(self) -> bool:
        """
        Check if call is allowed (async version).

        Returns:
            True if call allowed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.window_start

        # Reset window if period has elapsed
        if elapsed >= self.period:
            self.calls_made = 0
            self.window_start = now
            elapsed = 0

        if self.calls_made < self.calls:
            self.calls_made += 1
            return True

        # Calculate wait time
        wait_time = self.period - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            # Recursive call after waiting
            return await self.acquire_async()

        return False


# Global rate limiter for LLM calls
# Configuration: 100 calls per 60 seconds = 1.67 calls/sec
_llm_rate_limiter: Optional[RateLimiter] = None


def init_rate_limiter(calls: int = 100, period: int = 60) -> None:
    """
    Initialize global LLM rate limiter.

    Args:
        calls: Number of calls allowed per period (default: 100)
        period: Time period in seconds (default: 60)
    """
    global _llm_rate_limiter
    _llm_rate_limiter = RateLimiter(calls, period)


def rate_limit_llm_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to rate limit LLM calls.

    Ensures no more than configured calls per period.

    Usage:
        @rate_limit_llm_call
        async def call_llm(prompt: str) -> str:
            return await llm_client.complete(prompt)
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        if _llm_rate_limiter:
            await _llm_rate_limiter.acquire_async()
        return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        if _llm_rate_limiter:
            _llm_rate_limiter.acquire()
        return func(*args, **kwargs)

    # Return appropriate wrapper based on function type
    if inspect.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore


class BoundedCache:
    """
    LRU cache with bounded size to prevent memory leaks.

    Automatically evicts oldest entries when max size reached.
    """

    def __init__(self, maxsize: int = 1000):
        """
        Initialize bounded cache.

        Args:
            maxsize: Maximum number of cached items
        """
        self.maxsize = maxsize
        self.cache: Dict[str, Any] = {}
        self.access_order: list = []

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
            self.access_order.append(key)
            self.cache[key] = value
        else:
            # Add new item
            if len(self.cache) >= self.maxsize:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_order.clear()

    def info(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "utilization": len(self.cache) / self.maxsize,
        }


# Global consolidation result cache
# Stores up to 1000 most recent consolidation results
_consolidation_cache = BoundedCache(maxsize=1000)


@functools.lru_cache(maxsize=1000)
def get_cached_consolidation_result(memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached consolidation result (lru_cache version).

    The @functools.lru_cache decorator automatically:
    - Caches results
    - Evicts oldest entries when full
    - Is thread-safe

    Args:
        memory_id: ID of memory that was consolidated

    Returns:
        Cached consolidation result or None if not cached
    """
    # In production, this would load from database
    # For now, returns None (cache miss)
    return None


def get_cache_info() -> Dict[str, Any]:
    """Get LRU cache statistics."""
    cache_info = get_cached_consolidation_result.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses)
        if (cache_info.hits + cache_info.misses) > 0
        else 0.0,
    }


def clear_cache() -> None:
    """Clear consolidation result cache."""
    get_cached_consolidation_result.cache_clear()
