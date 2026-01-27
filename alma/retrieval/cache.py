"""
ALMA Retrieval Cache.

Multi-backend caching layer for retrieval results with TTL-based expiration.
Supports in-memory and Redis backends with performance monitoring.
"""

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from alma.types import MemorySlice

logger = logging.getLogger(__name__)


# ==================== DATA STRUCTURES ====================


@dataclass
class CacheEntry:
    """A cached retrieval result with metadata."""
    result: MemorySlice
    created_at: float  # time.time() timestamp
    expires_at: float
    hit_count: int = 0
    query_hash: str = ""
    # Metadata for selective invalidation
    agent: str = ""
    project_id: str = ""
    user_id: str = ""


@dataclass
class CacheStats:
    """Statistics about cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    # Performance metrics
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    p95_get_time_ms: float = 0.0
    p95_set_time_ms: float = 0.0
    total_get_calls: int = 0
    total_set_calls: int = 0

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
            "avg_get_time_ms": round(self.avg_get_time_ms, 2),
            "avg_set_time_ms": round(self.avg_set_time_ms, 2),
            "p95_get_time_ms": round(self.p95_get_time_ms, 2),
            "p95_set_time_ms": round(self.p95_set_time_ms, 2),
            "total_get_calls": self.total_get_calls,
            "total_set_calls": self.total_set_calls,
        }


@dataclass
class PerformanceMetrics:
    """Tracks timing metrics for performance analysis."""
    get_times: List[float] = field(default_factory=list)
    set_times: List[float] = field(default_factory=list)
    max_samples: int = 1000

    def record_get(self, duration_ms: float):
        """Record a get operation time."""
        self.get_times.append(duration_ms)
        if len(self.get_times) > self.max_samples:
            self.get_times = self.get_times[-self.max_samples:]

    def record_set(self, duration_ms: float):
        """Record a set operation time."""
        self.set_times.append(duration_ms)
        if len(self.set_times) > self.max_samples:
            self.set_times = self.set_times[-self.max_samples:]

    def get_percentile(self, times: List[float], percentile: float) -> float:
        """Calculate percentile from timing data."""
        if not times:
            return 0.0
        sorted_times = sorted(times)
        idx = int(len(sorted_times) * percentile / 100)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def get_avg(self, times: List[float]) -> float:
        """Calculate average from timing data."""
        if not times:
            return 0.0
        return sum(times) / len(times)


# ==================== CACHE INTERFACE ====================


class CacheBackend(ABC):
    """Abstract interface for cache backends."""

    @abstractmethod
    def get(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[MemorySlice]:
        """Get cached result if available."""
        pass

    @abstractmethod
    def set(
        self,
        query: str,
        agent: str,
        project_id: str,
        result: MemorySlice,
        user_id: Optional[str] = None,
        top_k: int = 5,
        ttl_override: Optional[int] = None,
    ) -> None:
        """Cache a retrieval result."""
        pass

    @abstractmethod
    def invalidate(
        self,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries. Returns count of invalidated entries."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


# ==================== IN-MEMORY CACHE ====================


class RetrievalCache(CacheBackend):
    """
    In-memory cache for retrieval results.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Thread-safe operations
    - Selective cache invalidation by agent/project
    - Performance metrics tracking
    - Monitoring hooks
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        cleanup_interval: int = 60,
        enable_metrics: bool = True,
    ):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
            max_entries: Maximum number of cached entries before eviction
            cleanup_interval: Seconds between cleanup cycles for expired entries
            enable_metrics: Whether to track performance metrics
        """
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        self.enable_metrics = enable_metrics

        self._cache: Dict[str, CacheEntry] = {}
        # Index for selective invalidation: agent -> set of cache keys
        self._agent_index: Dict[str, set] = {}
        # Index for selective invalidation: project_id -> set of cache keys
        self._project_index: Dict[str, set] = {}

        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_entries)
        self._metrics = PerformanceMetrics() if enable_metrics else None
        self._last_cleanup = time.time()

        # Monitoring hooks
        self._on_hit: Optional[Callable[[str, float], None]] = None
        self._on_miss: Optional[Callable[[str], None]] = None
        self._on_eviction: Optional[Callable[[int], None]] = None

    def set_hooks(
        self,
        on_hit: Optional[Callable[[str, float], None]] = None,
        on_miss: Optional[Callable[[str], None]] = None,
        on_eviction: Optional[Callable[[int], None]] = None,
    ):
        """
        Set monitoring hooks for cache events.

        Args:
            on_hit: Called on cache hit with (query_hash, latency_ms)
            on_miss: Called on cache miss with (query_hash)
            on_eviction: Called on eviction with (count)
        """
        self._on_hit = on_hit
        self._on_miss = on_miss
        self._on_eviction = on_eviction

    def _generate_key(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Generate a unique cache key for the query parameters."""
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
        """Get cached result if available and not expired."""
        start_time = time.time()
        key = self._generate_key(query, agent, project_id, user_id, top_k)
        now = time.time()

        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()

            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                if self._on_miss:
                    self._on_miss(key)
                self._record_get_time(start_time)
                return None

            if now > entry.expires_at:
                # Entry expired
                self._remove_entry(key, entry)
                self._stats.misses += 1
                if self._on_miss:
                    self._on_miss(key)
                self._record_get_time(start_time)
                return None

            # Cache hit
            entry.hit_count += 1
            self._stats.hits += 1
            latency_ms = (time.time() - start_time) * 1000
            if self._on_hit:
                self._on_hit(key, latency_ms)
            self._record_get_time(start_time)
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
    ) -> None:
        """Cache a retrieval result."""
        start_time = time.time()
        key = self._generate_key(query, agent, project_id, user_id, top_k)
        now = time.time()
        ttl = ttl_override or self.ttl

        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_entries and key not in self._cache:
                self._evict_lru()

            entry = CacheEntry(
                result=result,
                created_at=now,
                expires_at=now + ttl,
                hit_count=0,
                query_hash=key,
                agent=agent,
                project_id=project_id,
                user_id=user_id or "",
            )

            self._cache[key] = entry

            # Update indexes
            if agent not in self._agent_index:
                self._agent_index[agent] = set()
            self._agent_index[agent].add(key)

            if project_id not in self._project_index:
                self._project_index[project_id] = set()
            self._project_index[project_id].add(key)

            self._stats.current_size = len(self._cache)
            self._record_set_time(start_time)
            logger.debug(f"Cached result for query: {query[:50]}...")

    def invalidate(
        self,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        If no criteria provided, clears entire cache.

        Args:
            agent: Invalidate entries for this agent
            project_id: Invalidate entries for this project

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if agent is None and project_id is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._agent_index.clear()
                self._project_index.clear()
                self._stats.evictions += count
                self._stats.current_size = 0
                if self._on_eviction and count > 0:
                    self._on_eviction(count)
                logger.info(f"Invalidated entire cache ({count} entries)")
                return count

            keys_to_remove: set = set()

            # Collect keys matching agent
            if agent and agent in self._agent_index:
                keys_to_remove.update(self._agent_index[agent])

            # Collect keys matching project (intersection if both specified)
            if project_id and project_id in self._project_index:
                project_keys = self._project_index[project_id]
                if agent:
                    # Intersection: both agent AND project must match
                    keys_to_remove = keys_to_remove.intersection(project_keys)
                else:
                    keys_to_remove.update(project_keys)

            # Remove matched entries
            count = 0
            for key in keys_to_remove:
                if key in self._cache:
                    entry = self._cache[key]
                    self._remove_entry(key, entry)
                    count += 1

            self._stats.evictions += count
            if self._on_eviction and count > 0:
                self._on_eviction(count)
            logger.info(
                f"Invalidated {count} cache entries for agent={agent}, project={project_id}"
            )
            return count

    def _remove_entry(self, key: str, entry: CacheEntry) -> None:
        """Remove an entry from cache and indexes."""
        del self._cache[key]

        # Update indexes
        if entry.agent in self._agent_index:
            self._agent_index[entry.agent].discard(key)
            if not self._agent_index[entry.agent]:
                del self._agent_index[entry.agent]

        if entry.project_id in self._project_index:
            self._project_index[entry.project_id].discard(key)
            if not self._project_index[entry.project_id]:
                del self._project_index[entry.project_id]

        self._stats.current_size = len(self._cache)

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired = [
            (key, entry)
            for key, entry in self._cache.items()
            if now > entry.expires_at
        ]

        for key, entry in expired:
            self._remove_entry(key, entry)

        if expired:
            self._stats.evictions += len(expired)
            if self._on_eviction:
                self._on_eviction(len(expired))
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")

        self._last_cleanup = now

    def _evict_lru(self) -> None:
        """Evict least recently used entry (based on hit count and age)."""
        if not self._cache:
            return

        # Find entry with lowest score (hit_count / age)
        now = time.time()
        worst_key = None
        worst_entry = None
        worst_score = float("inf")

        for key, entry in self._cache.items():
            age = now - entry.created_at + 1  # +1 to avoid division by zero
            score = (entry.hit_count + 1) / age
            if score < worst_score:
                worst_score = score
                worst_key = key
                worst_entry = entry

        if worst_key and worst_entry:
            self._remove_entry(worst_key, worst_entry)
            self._stats.evictions += 1
            if self._on_eviction:
                self._on_eviction(1)
            logger.debug("Evicted LRU cache entry")

    def _record_get_time(self, start_time: float) -> None:
        """Record get operation timing."""
        if self._metrics:
            duration_ms = (time.time() - start_time) * 1000
            self._metrics.record_get(duration_ms)
            self._stats.total_get_calls += 1

    def _record_set_time(self, start_time: float) -> None:
        """Record set operation timing."""
        if self._metrics:
            duration_ms = (time.time() - start_time) * 1000
            self._metrics.record_set(duration_ms)
            self._stats.total_set_calls += 1

    def get_stats(self) -> CacheStats:
        """Get cache statistics with performance metrics."""
        with self._lock:
            self._stats.current_size = len(self._cache)

            if self._metrics:
                self._stats.avg_get_time_ms = self._metrics.get_avg(
                    self._metrics.get_times
                )
                self._stats.avg_set_time_ms = self._metrics.get_avg(
                    self._metrics.set_times
                )
                self._stats.p95_get_time_ms = self._metrics.get_percentile(
                    self._metrics.get_times, 95
                )
                self._stats.p95_set_time_ms = self._metrics.get_percentile(
                    self._metrics.set_times, 95
                )

            return self._stats

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._agent_index.clear()
            self._project_index.clear()
            self._stats = CacheStats(max_size=self.max_entries)
            if self._metrics:
                self._metrics = PerformanceMetrics()
            logger.info(f"Cleared cache ({count} entries)")


# ==================== REDIS CACHE ====================


class RedisCache(CacheBackend):
    """
    Redis-based cache for distributed deployments.

    Features:
    - Distributed caching across multiple instances
    - Built-in TTL via Redis EXPIRE
    - Selective invalidation using key prefixes and patterns
    - Performance metrics tracking
    - Automatic reconnection handling
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 300,
        key_prefix: str = "alma:cache:",
        connection_pool_size: int = 10,
        enable_metrics: bool = True,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            ttl_seconds: Default TTL for cache entries
            key_prefix: Prefix for all cache keys
            connection_pool_size: Size of connection pool
            enable_metrics: Whether to track performance metrics
        """
        self.ttl = ttl_seconds
        self.key_prefix = key_prefix
        self.enable_metrics = enable_metrics

        self._stats = CacheStats()
        self._metrics = PerformanceMetrics() if enable_metrics else None
        self._lock = threading.RLock()

        # Monitoring hooks
        self._on_hit: Optional[Callable[[str, float], None]] = None
        self._on_miss: Optional[Callable[[str], None]] = None
        self._on_eviction: Optional[Callable[[int], None]] = None

        # Try to import redis
        try:
            import redis

            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=connection_pool_size,
                decode_responses=False,  # We handle encoding ourselves
            )
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError as err:
            raise ImportError(
                "redis package required for RedisCache. "
                "Install with: pip install redis"
            ) from err
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def set_hooks(
        self,
        on_hit: Optional[Callable[[str, float], None]] = None,
        on_miss: Optional[Callable[[str], None]] = None,
        on_eviction: Optional[Callable[[int], None]] = None,
    ):
        """Set monitoring hooks for cache events."""
        self._on_hit = on_hit
        self._on_miss = on_miss
        self._on_eviction = on_eviction

    def _generate_key(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Generate a unique cache key with prefix for pattern matching."""
        key_parts = [
            query.lower().strip(),
            user_id or "",
            str(top_k),
        ]
        hash_part = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]
        # Structure: prefix:project:agent:hash
        # This enables pattern-based invalidation
        return f"{self.key_prefix}{project_id}:{agent}:{hash_part}"

    def _serialize_result(self, result: MemorySlice) -> bytes:
        """Serialize MemorySlice to bytes."""
        data = {
            "query": result.query,
            "agent": result.agent,
            "retrieval_time_ms": result.retrieval_time_ms,
            "heuristics": [
                {
                    "id": h.id,
                    "agent": h.agent,
                    "project_id": h.project_id,
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                    "occurrence_count": h.occurrence_count,
                    "success_count": h.success_count,
                    "last_validated": h.last_validated.isoformat()
                    if h.last_validated
                    else None,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                }
                for h in result.heuristics
            ],
            "outcomes": [
                {
                    "id": o.id,
                    "agent": o.agent,
                    "project_id": o.project_id,
                    "task_type": o.task_type,
                    "task_description": o.task_description,
                    "success": o.success,
                    "strategy_used": o.strategy_used,
                    "duration_ms": o.duration_ms,
                    "timestamp": o.timestamp.isoformat() if o.timestamp else None,
                }
                for o in result.outcomes
            ],
            "preferences": [
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "category": p.category,
                    "preference": p.preference,
                    "source": p.source,
                    "confidence": p.confidence,
                }
                for p in result.preferences
            ],
            "domain_knowledge": [
                {
                    "id": dk.id,
                    "agent": dk.agent,
                    "project_id": dk.project_id,
                    "domain": dk.domain,
                    "fact": dk.fact,
                    "source": dk.source,
                    "confidence": dk.confidence,
                }
                for dk in result.domain_knowledge
            ],
            "anti_patterns": [
                {
                    "id": ap.id,
                    "agent": ap.agent,
                    "project_id": ap.project_id,
                    "pattern": ap.pattern,
                    "why_bad": ap.why_bad,
                    "better_alternative": ap.better_alternative,
                    "occurrence_count": ap.occurrence_count,
                }
                for ap in result.anti_patterns
            ],
        }
        return json.dumps(data).encode("utf-8")

    def _deserialize_result(self, data: bytes) -> MemorySlice:
        """Deserialize bytes to MemorySlice."""
        from alma.types import (
            AntiPattern,
            DomainKnowledge,
            Heuristic,
            Outcome,
            UserPreference,
        )

        obj = json.loads(data.decode("utf-8"))

        def parse_datetime(s):
            if s is None:
                return datetime.now(timezone.utc)
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        heuristics = [
            Heuristic(
                id=h["id"],
                agent=h["agent"],
                project_id=h["project_id"],
                condition=h["condition"],
                strategy=h["strategy"],
                confidence=h["confidence"],
                occurrence_count=h["occurrence_count"],
                success_count=h["success_count"],
                last_validated=parse_datetime(h.get("last_validated")),
                created_at=parse_datetime(h.get("created_at")),
            )
            for h in obj.get("heuristics", [])
        ]

        outcomes = [
            Outcome(
                id=o["id"],
                agent=o["agent"],
                project_id=o["project_id"],
                task_type=o["task_type"],
                task_description=o["task_description"],
                success=o["success"],
                strategy_used=o["strategy_used"],
                duration_ms=o.get("duration_ms"),
                timestamp=parse_datetime(o.get("timestamp")),
            )
            for o in obj.get("outcomes", [])
        ]

        preferences = [
            UserPreference(
                id=p["id"],
                user_id=p["user_id"],
                category=p["category"],
                preference=p["preference"],
                source=p["source"],
                confidence=p.get("confidence", 1.0),
            )
            for p in obj.get("preferences", [])
        ]

        domain_knowledge = [
            DomainKnowledge(
                id=dk["id"],
                agent=dk["agent"],
                project_id=dk["project_id"],
                domain=dk["domain"],
                fact=dk["fact"],
                source=dk["source"],
                confidence=dk.get("confidence", 1.0),
            )
            for dk in obj.get("domain_knowledge", [])
        ]

        anti_patterns = [
            AntiPattern(
                id=ap["id"],
                agent=ap["agent"],
                project_id=ap["project_id"],
                pattern=ap["pattern"],
                why_bad=ap["why_bad"],
                better_alternative=ap["better_alternative"],
                occurrence_count=ap["occurrence_count"],
                last_seen=datetime.now(timezone.utc),
            )
            for ap in obj.get("anti_patterns", [])
        ]

        return MemorySlice(
            heuristics=heuristics,
            outcomes=outcomes,
            preferences=preferences,
            domain_knowledge=domain_knowledge,
            anti_patterns=anti_patterns,
            query=obj.get("query"),
            agent=obj.get("agent"),
            retrieval_time_ms=obj.get("retrieval_time_ms"),
        )

    def get(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[MemorySlice]:
        """Get cached result from Redis."""
        start_time = time.time()
        key = self._generate_key(query, agent, project_id, user_id, top_k)

        try:
            data = self._redis.get(key)

            if data is None:
                with self._lock:
                    self._stats.misses += 1
                if self._on_miss:
                    self._on_miss(key)
                self._record_get_time(start_time)
                return None

            result = self._deserialize_result(data)
            with self._lock:
                self._stats.hits += 1
            latency_ms = (time.time() - start_time) * 1000
            if self._on_hit:
                self._on_hit(key, latency_ms)
            self._record_get_time(start_time)
            logger.debug(f"Redis cache hit for query: {query[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            with self._lock:
                self._stats.misses += 1
            self._record_get_time(start_time)
            return None

    def set(
        self,
        query: str,
        agent: str,
        project_id: str,
        result: MemorySlice,
        user_id: Optional[str] = None,
        top_k: int = 5,
        ttl_override: Optional[int] = None,
    ) -> None:
        """Cache a retrieval result in Redis."""
        start_time = time.time()
        key = self._generate_key(query, agent, project_id, user_id, top_k)
        ttl = ttl_override or self.ttl

        try:
            data = self._serialize_result(result)
            self._redis.setex(key, ttl, data)
            self._record_set_time(start_time)
            logger.debug(f"Redis cached result for query: {query[:50]}...")

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self._record_set_time(start_time)

    def invalidate(
        self,
        agent: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries using Redis pattern matching.

        Pattern structure: prefix:project:agent:hash
        """
        try:
            if agent is None and project_id is None:
                # Clear all ALMA cache keys
                pattern = f"{self.key_prefix}*"
            elif project_id and agent:
                # Specific project and agent
                pattern = f"{self.key_prefix}{project_id}:{agent}:*"
            elif project_id:
                # All agents for a project
                pattern = f"{self.key_prefix}{project_id}:*"
            elif agent:
                # Specific agent across all projects
                pattern = f"{self.key_prefix}*:{agent}:*"
            else:
                return 0

            # Use SCAN for safe iteration over keys
            count = 0
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    self._redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break

            with self._lock:
                self._stats.evictions += count
            if self._on_eviction and count > 0:
                self._on_eviction(count)
            logger.info(
                f"Invalidated {count} Redis cache entries for agent={agent}, project={project_id}"
            )
            return count

        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
            return 0

    def _record_get_time(self, start_time: float) -> None:
        """Record get operation timing."""
        if self._metrics:
            with self._lock:
                duration_ms = (time.time() - start_time) * 1000
                self._metrics.record_get(duration_ms)
                self._stats.total_get_calls += 1

    def _record_set_time(self, start_time: float) -> None:
        """Record set operation timing."""
        if self._metrics:
            with self._lock:
                duration_ms = (time.time() - start_time) * 1000
                self._metrics.record_set(duration_ms)
                self._stats.total_set_calls += 1

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        try:
            # Get current cache size from Redis
            pattern = f"{self.key_prefix}*"
            cursor = 0
            count = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break

            with self._lock:
                self._stats.current_size = count

                if self._metrics:
                    self._stats.avg_get_time_ms = self._metrics.get_avg(
                        self._metrics.get_times
                    )
                    self._stats.avg_set_time_ms = self._metrics.get_avg(
                        self._metrics.set_times
                    )
                    self._stats.p95_get_time_ms = self._metrics.get_percentile(
                        self._metrics.get_times, 95
                    )
                    self._stats.p95_set_time_ms = self._metrics.get_percentile(
                        self._metrics.set_times, 95
                    )

                return self._stats

        except Exception as e:
            logger.error(f"Redis get_stats error: {e}")
            return self._stats

    def clear(self) -> None:
        """Clear all ALMA cache entries from Redis."""
        try:
            count = self.invalidate()
            with self._lock:
                self._stats = CacheStats()
                if self._metrics:
                    self._metrics = PerformanceMetrics()
            logger.info(f"Cleared Redis cache ({count} entries)")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


# ==================== NULL CACHE ====================


class NullCache(CacheBackend):
    """
    A no-op cache implementation for testing or when caching is disabled.

    All operations are valid but don't actually cache anything.
    """

    def __init__(self):
        """Initialize null cache."""
        self._stats = CacheStats()

    def get(self, *args, **kwargs) -> Optional[MemorySlice]:
        """Always returns None (cache miss)."""
        self._stats.misses += 1
        return None

    def set(self, *args, **kwargs) -> None:
        """No-op."""
        pass

    def invalidate(self, *args, **kwargs) -> int:
        """No-op."""
        return 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def clear(self) -> None:
        """No-op."""
        pass


# ==================== CACHE FACTORY ====================


def create_cache(
    backend: str = "memory",
    ttl_seconds: int = 300,
    max_entries: int = 1000,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    redis_db: int = 0,
    enable_metrics: bool = True,
) -> CacheBackend:
    """
    Factory function to create a cache backend.

    Args:
        backend: "memory", "redis", or "null"
        ttl_seconds: TTL for cache entries
        max_entries: Max entries for memory cache
        redis_host: Redis host (for redis backend)
        redis_port: Redis port (for redis backend)
        redis_password: Redis password (for redis backend)
        redis_db: Redis database number (for redis backend)
        enable_metrics: Whether to track performance metrics

    Returns:
        Configured CacheBackend instance
    """
    if backend == "redis":
        return RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            ttl_seconds=ttl_seconds,
            enable_metrics=enable_metrics,
        )
    elif backend == "null":
        return NullCache()
    else:
        return RetrievalCache(
            ttl_seconds=ttl_seconds,
            max_entries=max_entries,
            enable_metrics=enable_metrics,
        )
