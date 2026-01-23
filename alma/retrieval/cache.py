"""
ALMA Retrieval Cache.

In-memory caching layer for retrieval results with TTL-based expiration.
"""

import time
import hashlib
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from alma.types import MemorySlice

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached retrieval result with metadata."""
    result: MemorySlice
    created_at: float  # time.time() timestamp
    expires_at: float
    hit_count: int = 0
    query_hash: str = ""


@dataclass
class CacheStats:
    """Statistics about cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.2%}",
            "current_size": self.current_size,
            "max_size": self.max_size,
        }


class RetrievalCache:
    """
    In-memory cache for retrieval results.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Thread-safe operations
    - Cache statistics tracking
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        cleanup_interval: int = 60,
    ):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
            max_entries: Maximum number of cached entries before eviction
            cleanup_interval: Seconds between cleanup cycles for expired entries
        """
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_entries)
        self._last_cleanup = time.time()

    def _generate_key(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        Generate a unique cache key for the query parameters.

        Args:
            query: Task description
            agent: Agent name
            project_id: Project identifier
            user_id: Optional user identifier
            top_k: Number of results requested

        Returns:
            SHA256 hash of the combined parameters
        """
        key_parts = [
            query.lower().strip(),
            agent,
            project_id,
            user_id or "",
            str(top_k),
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[MemorySlice]:
        """
        Get cached result if available and not expired.

        Args:
            query: Task description
            agent: Agent name
            project_id: Project identifier
            user_id: Optional user identifier
            top_k: Number of results requested

        Returns:
            Cached MemorySlice or None if not found/expired
        """
        key = self._generate_key(query, agent, project_id, user_id, top_k)
        now = time.time()

        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()

            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if now > entry.expires_at:
                # Entry expired
                del self._cache[key]
                self._stats.misses += 1
                self._stats.current_size = len(self._cache)
                return None

            # Cache hit
            entry.hit_count += 1
            self._stats.hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry.result

    def set(
        self,
        query: str,
        agent: str,
        project_id: str,
        result: MemorySlice,
        user_id: Optional[str] = None,
        top_k: int = 5,
        ttl_override: Optional[int] = None,
    ):
        """
        Cache a retrieval result.

        Args:
            query: Task description
            agent: Agent name
            project_id: Project identifier
            result: MemorySlice to cache
            user_id: Optional user identifier
            top_k: Number of results requested
            ttl_override: Optional TTL override for this entry
        """
        key = self._generate_key(query, agent, project_id, user_id, top_k)
        now = time.time()
        ttl = ttl_override or self.ttl

        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_entries and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                result=result,
                created_at=now,
                expires_at=now + ttl,
                hit_count=0,
                query_hash=key,
            )
            self._stats.current_size = len(self._cache)
            logger.debug(f"Cached result for query: {query[:50]}...")

    def invalidate(
        self,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Invalidate cache entries matching criteria.

        If no criteria provided, clears entire cache.

        Args:
            agent: Invalidate entries for this agent
            project_id: Invalidate entries for this project
        """
        with self._lock:
            if agent is None and project_id is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._stats.evictions += count
                self._stats.current_size = 0
                logger.info(f"Invalidated entire cache ({count} entries)")
                return

            # Selective invalidation requires scanning all entries
            # Since keys are hashed, we need to check metadata
            # For now, clear all if any filter provided
            # TODO: Store metadata with entries for selective invalidation
            count = len(self._cache)
            self._cache.clear()
            self._stats.evictions += count
            self._stats.current_size = 0
            logger.info(f"Invalidated cache for agent={agent}, project={project_id}")

    def _cleanup_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            self._stats.evictions += len(expired_keys)
            self._stats.current_size = len(self._cache)
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        self._last_cleanup = now

    def _evict_lru(self):
        """Evict least recently used entry (based on hit count and age)."""
        if not self._cache:
            return

        # Find entry with lowest score (hit_count / age)
        # Lower score = less useful = evict first
        now = time.time()
        worst_key = None
        worst_score = float('inf')

        for key, entry in self._cache.items():
            age = now - entry.created_at + 1  # +1 to avoid division by zero
            score = (entry.hit_count + 1) / age  # +1 so new entries aren't immediately evicted
            if score < worst_score:
                worst_score = score
                worst_key = key

        if worst_key:
            del self._cache[worst_key]
            self._stats.evictions += 1
            self._stats.current_size = len(self._cache)
            logger.debug("Evicted LRU cache entry")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.current_size = len(self._cache)
            return self._stats

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats = CacheStats(max_size=self.max_entries)
            logger.info(f"Cleared cache ({count} entries)")


class NullCache(RetrievalCache):
    """
    A no-op cache implementation for testing or when caching is disabled.

    All operations are valid but don't actually cache anything.
    """

    def __init__(self):
        """Initialize null cache."""
        super().__init__()

    def get(self, *args, **kwargs) -> Optional[MemorySlice]:
        """Always returns None (cache miss)."""
        self._stats.misses += 1
        return None

    def set(self, *args, **kwargs):
        """No-op."""
        pass

    def invalidate(self, *args, **kwargs):
        """No-op."""
        pass
