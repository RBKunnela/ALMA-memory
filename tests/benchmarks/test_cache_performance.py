"""
Performance benchmarks for ALMA retrieval cache.

Target: <200ms p95 for cache operations.

Run with: pytest tests/benchmarks/test_cache_performance.py -v --benchmark
"""

import statistics
import time
from datetime import datetime, timezone
from typing import List

from alma.retrieval.cache import (
    NullCache,
    RetrievalCache,
    create_cache,
)
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemorySlice,
    Outcome,
    UserPreference,
)

# ==================== FIXTURES ====================


def create_test_memory_slice(size: str = "small") -> MemorySlice:
    """Create a test MemorySlice with configurable size."""
    now = datetime.now(timezone.utc)

    if size == "small":
        n_heuristics = 3
        n_outcomes = 5
        n_knowledge = 2
        n_anti_patterns = 1
    elif size == "medium":
        n_heuristics = 10
        n_outcomes = 20
        n_knowledge = 10
        n_anti_patterns = 5
    else:  # large
        n_heuristics = 50
        n_outcomes = 100
        n_knowledge = 50
        n_anti_patterns = 25

    heuristics = [
        Heuristic(
            id=f"heur_{i}",
            agent="test_agent",
            project_id="proj_1",
            condition=f"When task type is type_{i}",
            strategy=f"Use strategy {i} with detailed explanation " * 10,
            confidence=0.8 + (i % 20) / 100,
            occurrence_count=i + 1,
            success_count=i,
            last_validated=now,
            created_at=now,
        )
        for i in range(n_heuristics)
    ]

    outcomes = [
        Outcome(
            id=f"out_{i}",
            agent="test_agent",
            project_id="proj_1",
            task_type=f"task_type_{i % 5}",
            task_description=f"Test task {i} with description " * 5,
            success=i % 2 == 0,
            strategy_used=f"Strategy for task {i}",
            duration_ms=100 + i * 10,
            timestamp=now,
        )
        for i in range(n_outcomes)
    ]

    preferences = [
        UserPreference(
            id=f"pref_{i}",
            user_id="user_1",
            category=["communication", "code_style", "workflow"][i % 3],
            preference=f"Prefer style {i}",
            source="explicit",
            confidence=0.9,
        )
        for i in range(3)
    ]

    domain_knowledge = [
        DomainKnowledge(
            id=f"dk_{i}",
            agent="test_agent",
            project_id="proj_1",
            domain=f"domain_{i}",
            fact=f"Important fact {i} about the system " * 3,
            source="user_stated",
            confidence=1.0,
        )
        for i in range(n_knowledge)
    ]

    anti_patterns = [
        AntiPattern(
            id=f"anti_{i}",
            agent="test_agent",
            project_id="proj_1",
            pattern=f"Anti-pattern {i}",
            why_bad=f"This causes issue {i}",
            better_alternative=f"Use approach {i} instead",
            occurrence_count=i + 1,
            last_seen=now,
        )
        for i in range(n_anti_patterns)
    ]

    return MemorySlice(
        heuristics=heuristics,
        outcomes=outcomes,
        preferences=preferences,
        domain_knowledge=domain_knowledge,
        anti_patterns=anti_patterns,
        query="test query",
        agent="test_agent",
        retrieval_time_ms=50,
    )


def run_benchmark(
    cache,
    n_iterations: int = 100,
    result_size: str = "small",
) -> dict:
    """
    Run cache benchmark and return timing statistics.

    Args:
        cache: Cache backend to test
        n_iterations: Number of iterations
        result_size: Size of test data

    Returns:
        Dict with timing statistics
    """
    test_slice = create_test_memory_slice(result_size)

    # Benchmark SET operations
    set_times: List[float] = []
    for i in range(n_iterations):
        query = f"benchmark query {i}"
        start = time.perf_counter()
        cache.set(
            query=query,
            agent="test_agent",
            project_id="proj_1",
            result=test_slice,
            user_id="user_1",
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        set_times.append(elapsed_ms)

    # Benchmark GET operations (cache hits)
    get_hit_times: List[float] = []
    for i in range(n_iterations):
        query = f"benchmark query {i % n_iterations}"  # Existing keys
        start = time.perf_counter()
        cache.get(
            query=query,
            agent="test_agent",
            project_id="proj_1",
            user_id="user_1",
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        get_hit_times.append(elapsed_ms)

    # Benchmark GET operations (cache misses)
    get_miss_times: List[float] = []
    for i in range(n_iterations):
        query = f"nonexistent query {i}"
        start = time.perf_counter()
        cache.get(
            query=query,
            agent="other_agent",
            project_id="proj_2",
            user_id="user_1",
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        get_miss_times.append(elapsed_ms)

    # Calculate statistics
    def calc_stats(times: List[float]) -> dict:
        sorted_times = sorted(times)
        n = len(sorted_times)
        return {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0,
            "stdev": statistics.stdev(times) if n > 1 else 0,
        }

    return {
        "set": calc_stats(set_times),
        "get_hit": calc_stats(get_hit_times),
        "get_miss": calc_stats(get_miss_times),
        "n_iterations": n_iterations,
        "result_size": result_size,
    }


# ==================== MEMORY CACHE BENCHMARKS ====================


class TestMemoryCachePerformance:
    """Performance benchmarks for in-memory cache."""

    def test_small_payload_under_200ms_p95(self):
        """Verify p95 latency under 200ms for small payloads."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=1000)
        results = run_benchmark(cache, n_iterations=100, result_size="small")

        # Check p95 targets
        assert results["set"]["p95"] < 200, (
            f"SET p95 ({results['set']['p95']:.2f}ms) exceeds 200ms target"
        )
        assert results["get_hit"]["p95"] < 200, (
            f"GET HIT p95 ({results['get_hit']['p95']:.2f}ms) exceeds 200ms target"
        )
        assert results["get_miss"]["p95"] < 200, (
            f"GET MISS p95 ({results['get_miss']['p95']:.2f}ms) exceeds 200ms target"
        )

        print("\nSmall payload benchmark results:")
        print(f"  SET p95: {results['set']['p95']:.3f}ms")
        print(f"  GET HIT p95: {results['get_hit']['p95']:.3f}ms")
        print(f"  GET MISS p95: {results['get_miss']['p95']:.3f}ms")

    def test_medium_payload_under_200ms_p95(self):
        """Verify p95 latency under 200ms for medium payloads."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=1000)
        results = run_benchmark(cache, n_iterations=100, result_size="medium")

        assert results["set"]["p95"] < 200, (
            f"SET p95 ({results['set']['p95']:.2f}ms) exceeds 200ms target"
        )
        assert results["get_hit"]["p95"] < 200, (
            f"GET HIT p95 ({results['get_hit']['p95']:.2f}ms) exceeds 200ms target"
        )

        print("\nMedium payload benchmark results:")
        print(f"  SET p95: {results['set']['p95']:.3f}ms")
        print(f"  GET HIT p95: {results['get_hit']['p95']:.3f}ms")

    def test_large_payload_under_200ms_p95(self):
        """Verify p95 latency under 200ms for large payloads."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=1000)
        results = run_benchmark(cache, n_iterations=50, result_size="large")

        assert results["set"]["p95"] < 200, (
            f"SET p95 ({results['set']['p95']:.2f}ms) exceeds 200ms target"
        )
        assert results["get_hit"]["p95"] < 200, (
            f"GET HIT p95 ({results['get_hit']['p95']:.2f}ms) exceeds 200ms target"
        )

        print("\nLarge payload benchmark results:")
        print(f"  SET p95: {results['set']['p95']:.3f}ms")
        print(f"  GET HIT p95: {results['get_hit']['p95']:.3f}ms")

    def test_high_volume_concurrent_access(self):
        """Test cache performance under high volume."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=500)
        results = run_benchmark(cache, n_iterations=500, result_size="small")

        # More lenient target for high volume
        assert results["set"]["p95"] < 200, (
            f"High volume SET p95 ({results['set']['p95']:.2f}ms) exceeds target"
        )
        assert results["get_hit"]["p95"] < 200, (
            f"High volume GET p95 ({results['get_hit']['p95']:.2f}ms) exceeds target"
        )

        print("\nHigh volume (500 iterations) benchmark results:")
        print(f"  SET p95: {results['set']['p95']:.3f}ms")
        print(f"  GET HIT p95: {results['get_hit']['p95']:.3f}ms")

    def test_eviction_performance(self):
        """Test performance when cache eviction is triggered."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=50)
        test_slice = create_test_memory_slice("small")

        eviction_times: List[float] = []
        # Fill cache beyond capacity to trigger evictions
        for i in range(100):
            query = f"eviction test query {i}"
            start = time.perf_counter()
            cache.set(
                query=query,
                agent="test_agent",
                project_id="proj_1",
                result=test_slice,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            eviction_times.append(elapsed_ms)

        sorted_times = sorted(eviction_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)]

        assert p95 < 200, f"Eviction p95 ({p95:.2f}ms) exceeds 200ms target"

        print("\nEviction benchmark results:")
        print(f"  SET with eviction p95: {p95:.3f}ms")

    def test_invalidation_performance(self):
        """Test selective cache invalidation performance."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=1000)
        test_slice = create_test_memory_slice("small")

        # Fill cache with entries from multiple agents/projects
        for i in range(100):
            cache.set(
                query=f"query {i}",
                agent=f"agent_{i % 5}",
                project_id=f"proj_{i % 3}",
                result=test_slice,
            )

        # Benchmark invalidation
        invalidation_times: List[float] = []
        for i in range(20):
            # Re-fill after each invalidation
            for j in range(10):
                cache.set(
                    query=f"refill query {i}_{j}",
                    agent="agent_0",
                    project_id="proj_0",
                    result=test_slice,
                )

            start = time.perf_counter()
            cache.invalidate(agent="agent_0", project_id="proj_0")
            elapsed_ms = (time.perf_counter() - start) * 1000
            invalidation_times.append(elapsed_ms)

        sorted_times = sorted(invalidation_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)]

        assert p95 < 200, f"Invalidation p95 ({p95:.2f}ms) exceeds 200ms target"

        print("\nInvalidation benchmark results:")
        print(f"  Selective invalidation p95: {p95:.3f}ms")


# ==================== PERFORMANCE METRICS TRACKING ====================


class TestPerformanceMetrics:
    """Test the built-in performance metrics tracking."""

    def test_metrics_collection(self):
        """Verify metrics are collected correctly."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=100, enable_metrics=True)
        test_slice = create_test_memory_slice("small")

        # Perform operations
        for i in range(50):
            cache.set(
                query=f"metrics test {i}",
                agent="test_agent",
                project_id="proj_1",
                result=test_slice,
            )
            cache.get(
                query=f"metrics test {i}",
                agent="test_agent",
                project_id="proj_1",
            )

        stats = cache.get_stats()

        # Verify metrics are populated
        assert stats.total_get_calls == 50
        assert stats.total_set_calls == 50
        assert stats.avg_get_time_ms >= 0
        assert stats.avg_set_time_ms >= 0
        assert stats.p95_get_time_ms >= 0
        assert stats.p95_set_time_ms >= 0

        print("\nPerformance metrics after 50 ops:")
        print(f"  Avg GET: {stats.avg_get_time_ms:.3f}ms")
        print(f"  Avg SET: {stats.avg_set_time_ms:.3f}ms")
        print(f"  P95 GET: {stats.p95_get_time_ms:.3f}ms")
        print(f"  P95 SET: {stats.p95_set_time_ms:.3f}ms")

        # Verify metrics are within target
        assert stats.p95_get_time_ms < 200, "GET p95 exceeds target"
        assert stats.p95_set_time_ms < 200, "SET p95 exceeds target"

    def test_hit_rate_tracking(self):
        """Verify hit rate is calculated correctly."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=100)
        test_slice = create_test_memory_slice("small")

        # Store 10 items
        for i in range(10):
            cache.set(
                query=f"hit rate test {i}",
                agent="test_agent",
                project_id="proj_1",
                result=test_slice,
            )

        # Access 10 stored items (hits) + 10 new queries (misses)
        for i in range(10):
            cache.get(
                query=f"hit rate test {i}",
                agent="test_agent",
                project_id="proj_1",
            )
        for i in range(10, 20):
            cache.get(
                query=f"hit rate test {i}",
                agent="test_agent",
                project_id="proj_1",
            )

        stats = cache.get_stats()

        assert stats.hits == 10
        assert stats.misses == 10
        assert abs(stats.hit_rate - 0.5) < 0.01, (
            f"Hit rate should be 50%, got {stats.hit_rate}"
        )

        print("\nHit rate tracking:")
        print(f"  Hits: {stats.hits}")
        print(f"  Misses: {stats.misses}")
        print(f"  Hit rate: {stats.hit_rate:.1%}")


# ==================== CACHE FACTORY TESTS ====================


class TestCacheFactory:
    """Test cache factory function."""

    def test_create_memory_cache(self):
        """Test creating in-memory cache."""
        cache = create_cache(backend="memory", ttl_seconds=120, max_entries=500)
        assert isinstance(cache, RetrievalCache)

    def test_create_null_cache(self):
        """Test creating null cache."""
        cache = create_cache(backend="null")
        assert isinstance(cache, NullCache)

    def test_null_cache_no_caching(self):
        """Verify null cache doesn't cache anything."""
        cache = NullCache()
        test_slice = create_test_memory_slice("small")

        cache.set(
            query="test",
            agent="agent",
            project_id="proj",
            result=test_slice,
        )

        result = cache.get(
            query="test",
            agent="agent",
            project_id="proj",
        )

        assert result is None  # Should always miss
        stats = cache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 0


if __name__ == "__main__":
    # Run benchmarks
    print("=" * 60)
    print("ALMA Cache Performance Benchmarks")
    print("Target: <200ms p95 for all cache operations")
    print("=" * 60)

    cache = RetrievalCache(ttl_seconds=60, max_entries=1000)

    for size in ["small", "medium", "large"]:
        print(f"\n{size.upper()} payload benchmark:")
        results = run_benchmark(cache, n_iterations=100, result_size=size)

        print(
            f"  SET - mean: {results['set']['mean']:.3f}ms, "
            f"p95: {results['set']['p95']:.3f}ms, "
            f"p99: {results['set']['p99']:.3f}ms"
        )
        print(
            f"  GET HIT - mean: {results['get_hit']['mean']:.3f}ms, "
            f"p95: {results['get_hit']['p95']:.3f}ms, "
            f"p99: {results['get_hit']['p99']:.3f}ms"
        )
        print(
            f"  GET MISS - mean: {results['get_miss']['mean']:.3f}ms, "
            f"p95: {results['get_miss']['p95']:.3f}ms, "
            f"p99: {results['get_miss']['p99']:.3f}ms"
        )

        # Check targets
        target_met = results["set"]["p95"] < 200 and results["get_hit"]["p95"] < 200
        print(f"  Target (<200ms p95): {'✓ PASS' if target_met else '✗ FAIL'}")

        cache.clear()
