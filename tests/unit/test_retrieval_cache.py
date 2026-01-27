"""
Unit tests for ALMA retrieval cache module.

Tests cover:
- RetrievalCache (in-memory)
- NullCache
- Cache invalidation
- Performance metrics
- Monitoring hooks
"""

import time
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from alma.types import (
    MemorySlice,
    Heuristic,
    Outcome,
    UserPreference,
    DomainKnowledge,
    AntiPattern,
)
from alma.retrieval.cache import (
    CacheBackend,
    RetrievalCache,
    NullCache,
    CacheEntry,
    CacheStats,
    PerformanceMetrics,
    create_cache,
)


# ==================== FIXTURES ====================


@pytest.fixture
def sample_memory_slice():
    """Create a sample MemorySlice for testing."""
    now = datetime.now(timezone.utc)
    return MemorySlice(
        heuristics=[
            Heuristic(
                id="heur_1",
                agent="test_agent",
                project_id="proj_1",
                condition="task type: api_testing",
                strategy="Use structured assertions",
                confidence=0.85,
                occurrence_count=10,
                success_count=8,
                last_validated=now,
                created_at=now,
            )
        ],
        outcomes=[
            Outcome(
                id="out_1",
                agent="test_agent",
                project_id="proj_1",
                task_type="api_testing",
                task_description="Test login endpoint",
                success=True,
                strategy_used="Used status code check",
                duration_ms=150,
                timestamp=now,
            )
        ],
        preferences=[
            UserPreference(
                id="pref_1",
                user_id="user_1",
                category="communication",
                preference="Be concise",
                source="explicit",
                confidence=1.0,
            )
        ],
        domain_knowledge=[
            DomainKnowledge(
                id="dk_1",
                agent="test_agent",
                project_id="proj_1",
                domain="authentication",
                fact="JWT tokens expire in 1 hour",
                source="user_stated",
                confidence=1.0,
            )
        ],
        anti_patterns=[
            AntiPattern(
                id="anti_1",
                agent="test_agent",
                project_id="proj_1",
                pattern="Hard-coded credentials",
                why_bad="Security vulnerability",
                better_alternative="Use environment variables",
                occurrence_count=3,
                last_seen=now,
            )
        ],
        query="test authentication",
        agent="test_agent",
        retrieval_time_ms=50,
    )


@pytest.fixture
def cache():
    """Create a fresh RetrievalCache instance."""
    return RetrievalCache(ttl_seconds=10, max_entries=100)


# ==================== CACHE BACKEND INTERFACE ====================


class TestCacheBackendInterface:
    """Test that implementations conform to CacheBackend interface."""

    def test_retrieval_cache_is_cache_backend(self, cache):
        """RetrievalCache should implement CacheBackend."""
        assert isinstance(cache, CacheBackend)

    def test_null_cache_is_cache_backend(self):
        """NullCache should implement CacheBackend."""
        null_cache = NullCache()
        assert isinstance(null_cache, CacheBackend)


# ==================== RETRIEVAL CACHE TESTS ====================


class TestRetrievalCache:
    """Tests for in-memory RetrievalCache."""

    def test_set_and_get(self, cache, sample_memory_slice):
        """Test basic set and get operations."""
        cache.set(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
            user_id="user_1",
            top_k=5,
        )

        result = cache.get(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
            user_id="user_1",
            top_k=5,
        )

        assert result is not None
        assert result.query == sample_memory_slice.query
        assert len(result.heuristics) == 1
        assert result.heuristics[0].id == "heur_1"

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get(
            query="nonexistent",
            agent="test_agent",
            project_id="proj_1",
        )
        assert result is None

    def test_cache_key_uniqueness(self, cache, sample_memory_slice):
        """Test that different parameters create different cache keys."""
        # Same query, different agents
        cache.set(
            query="test query",
            agent="agent_1",
            project_id="proj_1",
            result=sample_memory_slice,
        )
        cache.set(
            query="test query",
            agent="agent_2",
            project_id="proj_1",
            result=sample_memory_slice,
        )

        # Should be cached separately
        result1 = cache.get(
            query="test query",
            agent="agent_1",
            project_id="proj_1",
        )
        result2 = cache.get(
            query="test query",
            agent="agent_2",
            project_id="proj_1",
        )

        assert result1 is not None
        assert result2 is not None
        assert cache.get_stats().current_size == 2

    def test_ttl_expiration(self, sample_memory_slice):
        """Test that cache entries expire after TTL."""
        cache = RetrievalCache(ttl_seconds=1, max_entries=100)

        cache.set(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
        )

        # Should be available immediately
        result = cache.get(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
        )
        assert result is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        result = cache.get(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
        )
        assert result is None

    def test_ttl_override(self, cache, sample_memory_slice):
        """Test TTL override per entry."""
        cache.set(
            query="short ttl",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
            ttl_override=1,
        )

        cache.set(
            query="long ttl",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
            ttl_override=3600,
        )

        # Wait for short TTL to expire
        time.sleep(1.1)

        short_result = cache.get(
            query="short ttl",
            agent="test_agent",
            project_id="proj_1",
        )
        long_result = cache.get(
            query="long ttl",
            agent="test_agent",
            project_id="proj_1",
        )

        assert short_result is None  # Expired
        assert long_result is not None  # Still valid

    def test_max_entries_eviction(self, sample_memory_slice):
        """Test LRU eviction when max entries reached."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=5)

        # Fill cache
        for i in range(5):
            cache.set(
                query=f"query {i}",
                agent="test_agent",
                project_id="proj_1",
                result=sample_memory_slice,
            )

        assert cache.get_stats().current_size == 5

        # Add one more to trigger eviction
        cache.set(
            query="query 5",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
        )

        # Should still be at max capacity
        assert cache.get_stats().current_size == 5
        assert cache.get_stats().evictions >= 1

    def test_clear(self, cache, sample_memory_slice):
        """Test clearing the cache."""
        for i in range(10):
            cache.set(
                query=f"query {i}",
                agent="test_agent",
                project_id="proj_1",
                result=sample_memory_slice,
            )

        assert cache.get_stats().current_size == 10

        cache.clear()

        assert cache.get_stats().current_size == 0

    def test_hit_count_tracking(self, cache, sample_memory_slice):
        """Test that hit counts are tracked."""
        cache.set(
            query="popular query",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
        )

        # Access multiple times
        for _ in range(5):
            cache.get(
                query="popular query",
                agent="test_agent",
                project_id="proj_1",
            )

        stats = cache.get_stats()
        assert stats.hits == 5

    def test_case_insensitive_query(self, cache, sample_memory_slice):
        """Test that queries are case-insensitive."""
        cache.set(
            query="Test Query",
            agent="test_agent",
            project_id="proj_1",
            result=sample_memory_slice,
        )

        result = cache.get(
            query="test query",
            agent="test_agent",
            project_id="proj_1",
        )

        assert result is not None


# ==================== CACHE INVALIDATION TESTS ====================


class TestCacheInvalidation:
    """Tests for selective cache invalidation."""

    def test_invalidate_all(self, cache, sample_memory_slice):
        """Test invalidating entire cache."""
        for i in range(10):
            cache.set(
                query=f"query {i}",
                agent=f"agent_{i % 3}",
                project_id=f"proj_{i % 2}",
                result=sample_memory_slice,
            )

        assert cache.get_stats().current_size == 10

        count = cache.invalidate()

        assert count == 10
        assert cache.get_stats().current_size == 0

    def test_invalidate_by_agent(self, cache, sample_memory_slice):
        """Test invalidating by agent."""
        # Add entries for multiple agents
        for i in range(6):
            cache.set(
                query=f"query {i}",
                agent=f"agent_{i % 2}",  # agent_0 and agent_1
                project_id="proj_1",
                result=sample_memory_slice,
            )

        assert cache.get_stats().current_size == 6

        # Invalidate only agent_0
        count = cache.invalidate(agent="agent_0")

        assert count == 3
        assert cache.get_stats().current_size == 3

        # agent_1 entries should still be there
        result = cache.get(
            query="query 1",
            agent="agent_1",
            project_id="proj_1",
        )
        assert result is not None

    def test_invalidate_by_project(self, cache, sample_memory_slice):
        """Test invalidating by project."""
        for i in range(6):
            cache.set(
                query=f"query {i}",
                agent="test_agent",
                project_id=f"proj_{i % 2}",  # proj_0 and proj_1
                result=sample_memory_slice,
            )

        count = cache.invalidate(project_id="proj_0")

        assert count == 3
        assert cache.get_stats().current_size == 3

    def test_invalidate_by_agent_and_project(self, cache, sample_memory_slice):
        """Test invalidating by both agent and project."""
        # Use different modulo values to create a matrix of agent/project combinations
        # This ensures we have varied agent/project pairings
        for i in range(9):
            cache.set(
                query=f"query {i}",
                agent=f"agent_{i % 3}",  # agent_0, agent_1, agent_2 (cycles every 3)
                project_id=f"proj_{i // 3}",  # proj_0, proj_0, proj_0, proj_1, proj_1, proj_1, proj_2, proj_2, proj_2
                result=sample_memory_slice,
            )
        # This creates:
        # i=0: agent_0, proj_0
        # i=1: agent_1, proj_0
        # i=2: agent_2, proj_0
        # i=3: agent_0, proj_1
        # i=4: agent_1, proj_1
        # i=5: agent_2, proj_1
        # i=6: agent_0, proj_2
        # i=7: agent_1, proj_2
        # i=8: agent_2, proj_2

        # Invalidate only agent_0 in proj_0
        count = cache.invalidate(agent="agent_0", project_id="proj_0")

        # Only 1 entry matches both criteria (i=0: agent_0, proj_0)
        assert count == 1
        assert cache.get_stats().current_size == 8


# ==================== PERFORMANCE METRICS TESTS ====================


class TestPerformanceMetrics:
    """Tests for performance metrics tracking."""

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        metrics = PerformanceMetrics()
        assert metrics.get_times == []
        assert metrics.set_times == []
        assert metrics.max_samples == 1000

    def test_record_operations(self):
        """Test recording operation times."""
        metrics = PerformanceMetrics()

        metrics.record_get(1.5)
        metrics.record_get(2.0)
        metrics.record_set(0.5)

        assert len(metrics.get_times) == 2
        assert len(metrics.set_times) == 1

    def test_max_samples_limit(self):
        """Test that samples are limited to max_samples."""
        metrics = PerformanceMetrics(max_samples=10)

        for i in range(20):
            metrics.record_get(float(i))

        assert len(metrics.get_times) == 10
        # Should keep the most recent samples
        assert metrics.get_times[0] == 10.0

    def test_percentile_calculation(self):
        """Test percentile calculation."""
        metrics = PerformanceMetrics()

        # Add known values
        for i in range(1, 101):
            metrics.record_get(float(i))

        p50 = metrics.get_percentile(metrics.get_times, 50)
        p95 = metrics.get_percentile(metrics.get_times, 95)
        p99 = metrics.get_percentile(metrics.get_times, 99)

        assert 49 <= p50 <= 51
        assert 94 <= p95 <= 96
        assert 98 <= p99 <= 100

    def test_average_calculation(self):
        """Test average calculation."""
        metrics = PerformanceMetrics()

        metrics.record_get(1.0)
        metrics.record_get(2.0)
        metrics.record_get(3.0)

        avg = metrics.get_avg(metrics.get_times)
        assert avg == 2.0

    def test_empty_metrics(self):
        """Test handling of empty metrics."""
        metrics = PerformanceMetrics()

        assert metrics.get_percentile([], 95) == 0.0
        assert metrics.get_avg([]) == 0.0

    def test_cache_stats_from_metrics(self, sample_memory_slice):
        """Test that cache stats include performance metrics."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=100, enable_metrics=True)

        for i in range(10):
            cache.set(
                query=f"query {i}",
                agent="test_agent",
                project_id="proj_1",
                result=sample_memory_slice,
            )
            cache.get(
                query=f"query {i}",
                agent="test_agent",
                project_id="proj_1",
            )

        stats = cache.get_stats()

        assert stats.total_get_calls == 10
        assert stats.total_set_calls == 10
        assert stats.avg_get_time_ms >= 0
        assert stats.avg_set_time_ms >= 0


# ==================== MONITORING HOOKS TESTS ====================


class TestMonitoringHooks:
    """Tests for monitoring hooks."""

    def test_on_hit_hook(self, cache, sample_memory_slice):
        """Test on_hit hook is called."""
        hit_calls = []

        def on_hit(key: str, latency: float):
            hit_calls.append((key, latency))

        cache.set_hooks(on_hit=on_hit)
        cache.set(
            query="test",
            agent="agent",
            project_id="proj",
            result=sample_memory_slice,
        )
        cache.get(
            query="test",
            agent="agent",
            project_id="proj",
        )

        assert len(hit_calls) == 1
        assert isinstance(hit_calls[0][0], str)
        assert hit_calls[0][1] >= 0

    def test_on_miss_hook(self, cache):
        """Test on_miss hook is called."""
        miss_calls = []

        def on_miss(key: str):
            miss_calls.append(key)

        cache.set_hooks(on_miss=on_miss)
        cache.get(
            query="nonexistent",
            agent="agent",
            project_id="proj",
        )

        assert len(miss_calls) == 1

    def test_on_eviction_hook(self, sample_memory_slice):
        """Test on_eviction hook is called."""
        eviction_calls = []

        def on_eviction(count: int):
            eviction_calls.append(count)

        cache = RetrievalCache(ttl_seconds=60, max_entries=3)
        cache.set_hooks(on_eviction=on_eviction)

        # Fill cache beyond capacity
        for i in range(5):
            cache.set(
                query=f"query {i}",
                agent="agent",
                project_id="proj",
                result=sample_memory_slice,
            )

        assert len(eviction_calls) >= 2  # At least 2 evictions


# ==================== NULL CACHE TESTS ====================


class TestNullCache:
    """Tests for NullCache (no-op implementation)."""

    def test_always_misses(self, sample_memory_slice):
        """Test that NullCache always returns None."""
        cache = NullCache()

        cache.set(
            query="test",
            agent="agent",
            project_id="proj",
            result=sample_memory_slice,
        )

        result = cache.get(
            query="test",
            agent="agent",
            project_id="proj",
        )

        assert result is None

    def test_tracks_misses(self):
        """Test that NullCache tracks miss count."""
        cache = NullCache()

        for _ in range(5):
            cache.get(query="test", agent="agent", project_id="proj")

        stats = cache.get_stats()
        assert stats.misses == 5
        assert stats.hits == 0

    def test_invalidate_returns_zero(self):
        """Test that invalidate returns 0."""
        cache = NullCache()
        assert cache.invalidate() == 0

    def test_clear_is_noop(self):
        """Test that clear is a no-op."""
        cache = NullCache()
        cache.clear()  # Should not raise


# ==================== CACHE FACTORY TESTS ====================


class TestCreateCacheFactory:
    """Tests for create_cache factory function."""

    def test_create_memory_cache(self):
        """Test creating in-memory cache."""
        cache = create_cache(backend="memory", ttl_seconds=120, max_entries=500)
        assert isinstance(cache, RetrievalCache)
        assert cache.ttl == 120
        assert cache.max_entries == 500

    def test_create_null_cache(self):
        """Test creating null cache."""
        cache = create_cache(backend="null")
        assert isinstance(cache, NullCache)

    def test_default_is_memory(self):
        """Test that default backend is memory."""
        cache = create_cache()
        assert isinstance(cache, RetrievalCache)

    def test_metrics_toggle(self, sample_memory_slice):
        """Test enabling/disabling metrics."""
        with_metrics = create_cache(enable_metrics=True)
        without_metrics = create_cache(enable_metrics=False)

        # Both should work
        with_metrics.set(
            query="test",
            agent="agent",
            project_id="proj",
            result=sample_memory_slice,
        )
        without_metrics.set(
            query="test",
            agent="agent",
            project_id="proj",
            result=sample_memory_slice,
        )

        # Only with_metrics should have timing data
        with_stats = with_metrics.get_stats()
        without_stats = without_metrics.get_stats()

        assert with_stats.total_set_calls == 1
        assert without_stats.total_set_calls == 0  # No metrics tracking


# ==================== CACHE STATS TESTS ====================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_division(self):
        """Test hit rate with no accesses."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=100,
            misses=50,
            evictions=10,
            current_size=500,
            max_size=1000,
            avg_get_time_ms=1.5,
            p95_get_time_ms=5.0,
        )

        d = stats.to_dict()

        assert d["hits"] == 100
        assert d["misses"] == 50
        assert d["hit_rate"] == "66.67%"
        assert d["avg_get_time_ms"] == 1.5
        assert d["p95_get_time_ms"] == 5.0


# ==================== THREAD SAFETY TESTS ====================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_access(self, sample_memory_slice):
        """Test concurrent read/write operations."""
        import threading

        cache = RetrievalCache(ttl_seconds=60, max_entries=1000)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(
                        query=f"concurrent {i}",
                        agent="agent",
                        project_id="proj",
                        result=sample_memory_slice,
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(
                        query=f"concurrent {i}",
                        agent="agent",
                        project_id="proj",
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
