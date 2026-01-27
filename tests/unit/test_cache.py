"""
Unit tests for ALMA Retrieval Cache.
"""

import time

from alma.retrieval.cache import (
    CacheEntry,
    CacheStats,
    NullCache,
    RetrievalCache,
)
from alma.types import MemorySlice


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        now = time.time()
        entry = CacheEntry(
            result=MemorySlice(query="test", agent="helena"),
            created_at=now,
            expires_at=now + 300,
            hit_count=0,
            query_hash="abc123",
        )

        assert entry.hit_count == 0
        assert entry.expires_at > entry.created_at


class TestCacheStats:
    """Tests for CacheStats tracking."""

    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        stats = CacheStats(hits=50, misses=50)
        assert stats.hit_rate == 0.5

        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0

    def test_hit_rate_zero_total(self):
        """Test hit rate with no operations."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test stats serialization."""
        stats = CacheStats(hits=10, misses=5, evictions=2, current_size=100, max_size=1000)
        d = stats.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d


class TestRetrievalCache:
    """Tests for RetrievalCache operations."""

    def test_key_generation_deterministic(self):
        """Same parameters should generate same key."""
        cache = RetrievalCache()

        key1 = cache._generate_key("test query", "helena", "project-1", None, 5)
        key2 = cache._generate_key("test query", "helena", "project-1", None, 5)

        assert key1 == key2

    def test_key_generation_case_insensitive(self):
        """Query should be case-insensitive for caching."""
        cache = RetrievalCache()

        key1 = cache._generate_key("Test Query", "helena", "project-1", None, 5)
        key2 = cache._generate_key("test query", "helena", "project-1", None, 5)

        assert key1 == key2

    def test_different_params_different_keys(self):
        """Different parameters should generate different keys."""
        cache = RetrievalCache()

        key1 = cache._generate_key("query", "helena", "project-1", None, 5)
        key2 = cache._generate_key("query", "victor", "project-1", None, 5)
        key3 = cache._generate_key("query", "helena", "project-2", None, 5)
        key4 = cache._generate_key("query", "helena", "project-1", "user-1", 5)
        key5 = cache._generate_key("query", "helena", "project-1", None, 10)

        keys = [key1, key2, key3, key4, key5]
        assert len(set(keys)) == 5  # All unique

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = RetrievalCache(ttl_seconds=60)
        slice_obj = MemorySlice(query="test", agent="helena")

        cache.set("test query", "helena", "project-1", slice_obj)
        result = cache.get("test query", "helena", "project-1")

        assert result is not None
        assert result.query == "test"
        assert result.agent == "helena"

    def test_get_miss(self):
        """Test cache miss returns None."""
        cache = RetrievalCache()
        result = cache.get("nonexistent", "helena", "project-1")
        assert result is None

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = RetrievalCache(ttl_seconds=1)
        slice_obj = MemorySlice(query="test", agent="helena")

        cache.set("test", "helena", "project-1", slice_obj)
        assert cache.get("test", "helena", "project-1") is not None

        time.sleep(1.5)
        assert cache.get("test", "helena", "project-1") is None

    def test_ttl_override(self):
        """Test TTL can be overridden per entry."""
        cache = RetrievalCache(ttl_seconds=60)
        slice_obj = MemorySlice(query="test", agent="helena")

        # Set with short TTL
        cache.set("test", "helena", "project-1", slice_obj, ttl_override=1)

        assert cache.get("test", "helena", "project-1") is not None
        time.sleep(1.5)
        assert cache.get("test", "helena", "project-1") is None

    def test_invalidate_all(self):
        """Test invalidating entire cache."""
        cache = RetrievalCache(ttl_seconds=60)

        cache.set("q1", "helena", "p1", MemorySlice(query="q1", agent="helena"))
        cache.set("q2", "helena", "p1", MemorySlice(query="q2", agent="helena"))

        cache.invalidate()

        assert cache.get("q1", "helena", "p1") is None
        assert cache.get("q2", "helena", "p1") is None

    def test_clear_resets_stats(self):
        """Test that clear resets statistics."""
        cache = RetrievalCache()

        # Generate some stats
        cache.get("test", "helena", "project-1")  # miss
        cache.set("test", "helena", "project-1", MemorySlice())
        cache.get("test", "helena", "project-1")  # hit

        stats_before = cache.get_stats()
        assert stats_before.hits == 1
        assert stats_before.misses == 1

        cache.clear()

        stats_after = cache.get_stats()
        assert stats_after.hits == 0
        assert stats_after.misses == 0

    def test_lru_eviction(self):
        """Test LRU eviction when max entries reached."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=2)

        # Add two entries
        cache.set("q1", "helena", "p1", MemorySlice(query="q1", agent="helena"))
        cache.set("q2", "helena", "p1", MemorySlice(query="q2", agent="helena"))

        # Access q2 to make it recently used
        cache.get("q2", "helena", "p1")

        # Add third entry - should evict q1 (least recently used)
        cache.set("q3", "helena", "p1", MemorySlice(query="q3", agent="helena"))

        assert cache.get("q2", "helena", "p1") is not None
        assert cache.get("q3", "helena", "p1") is not None
        # q1 should be evicted
        stats = cache.get_stats()
        assert stats.evictions >= 1

    def test_update_existing_entry(self):
        """Test that setting same key updates the entry."""
        cache = RetrievalCache(ttl_seconds=60)

        cache.set("q1", "helena", "p1", MemorySlice(query="v1", agent="helena"))
        cache.set("q1", "helena", "p1", MemorySlice(query="v2", agent="helena"))

        result = cache.get("q1", "helena", "p1")
        assert result.query == "v2"

    def test_hit_count_tracking(self):
        """Test that hit count is tracked per entry."""
        cache = RetrievalCache(ttl_seconds=60)
        cache.set("q1", "helena", "p1", MemorySlice(query="q1", agent="helena"))

        # Multiple hits
        cache.get("q1", "helena", "p1")
        cache.get("q1", "helena", "p1")
        cache.get("q1", "helena", "p1")

        stats = cache.get_stats()
        assert stats.hits == 3


class TestNullCache:
    """Tests for NullCache (disabled caching)."""

    def test_always_misses(self):
        """NullCache should always return None."""
        cache = NullCache()

        cache.set("test", "helena", "project-1", MemorySlice())
        result = cache.get("test", "helena", "project-1")

        assert result is None

    def test_tracks_misses(self):
        """NullCache should track miss statistics."""
        cache = NullCache()

        cache.get("q1", "helena", "p1")
        cache.get("q2", "helena", "p1")

        stats = cache.get_stats()
        assert stats.misses == 2
        assert stats.hits == 0

    def test_invalidate_noop(self):
        """NullCache invalidate should not raise."""
        cache = NullCache()
        cache.invalidate()  # Should not raise
        cache.invalidate(agent="helena")  # Should not raise
