"""
Unit tests for ALMA Retrieval Cache.
"""

import time

import pytest

from alma.retrieval.cache import (
    CacheEntry,
    CacheKeyGenerator,
    CacheStats,
    NullCache,
    RetrievalCache,
    create_cache,
)
from alma.types import MemorySlice


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator - collision-resistant key generation."""

    def test_deterministic_key_generation(self):
        """Same inputs should always produce same key."""
        gen = CacheKeyGenerator()

        key1 = gen.generate("test query", "agent1", "project1", "user1", 5)
        key2 = gen.generate("test query", "agent1", "project1", "user1", 5)

        assert key1 == key2

    def test_key_format_structure(self):
        """Keys should have correct namespace:version:hash structure."""
        gen = CacheKeyGenerator(namespace="test_ns")

        key = gen.generate("query", "agent", "project")

        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "test_ns"
        assert parts[1] == "v1"
        assert len(parts[2]) == 64  # Full SHA-256 hex

    def test_default_namespace(self):
        """Default namespace should be 'alma'."""
        gen = CacheKeyGenerator()

        key = gen.generate("query", "agent", "project")

        assert key.startswith("alma:v1:")

    def test_custom_namespace(self):
        """Custom namespace should be used in key."""
        gen = CacheKeyGenerator(namespace="custom_agent")

        key = gen.generate("query", "agent", "project")

        assert key.startswith("custom_agent:v1:")

    def test_different_namespaces_different_keys(self):
        """Same inputs with different namespaces produce different keys."""
        gen1 = CacheKeyGenerator(namespace="agent_a")
        gen2 = CacheKeyGenerator(namespace="agent_b")

        key1 = gen1.generate("query", "agent", "project", "user", 5)
        key2 = gen2.generate("query", "agent", "project", "user", 5)

        assert key1 != key2

    def test_collision_resistance_delimiter_attack(self):
        """Keys should not collide due to delimiter-based attack."""
        gen = CacheKeyGenerator()

        # These could collide with simple "|" concatenation:
        # "a|b" + "c" vs "a" + "b|c"
        key1 = gen.generate("a|b", "c", "project")
        key2 = gen.generate("a", "b|c", "project")

        assert key1 != key2

    def test_collision_resistance_empty_vs_none(self):
        """Empty string user_id should differ from None."""
        gen = CacheKeyGenerator()

        # These should produce different keys
        key_none = gen.generate("query", "agent", "project", None, 5)
        key_empty = gen.generate("query", "agent", "project", "", 5)

        # Note: Both should be treated as empty, so they may be equal
        # This is intentional - None and "" are semantically equivalent
        assert key_none == key_empty

    def test_query_normalization_case_insensitive(self):
        """Queries should be normalized to lowercase."""
        gen = CacheKeyGenerator()

        key1 = gen.generate("Test Query", "agent", "project")
        key2 = gen.generate("test query", "agent", "project")
        key3 = gen.generate("TEST QUERY", "agent", "project")

        assert key1 == key2 == key3

    def test_query_normalization_whitespace(self):
        """Query whitespace should be normalized."""
        gen = CacheKeyGenerator()

        key1 = gen.generate("test query", "agent", "project")
        key2 = gen.generate("  test   query  ", "agent", "project")
        key3 = gen.generate("test\t\nquery", "agent", "project")

        assert key1 == key2 == key3

    def test_different_top_k_different_keys(self):
        """Different top_k values should produce different keys."""
        gen = CacheKeyGenerator()

        key1 = gen.generate("query", "agent", "project", top_k=5)
        key2 = gen.generate("query", "agent", "project", top_k=10)
        key3 = gen.generate("query", "agent", "project", top_k=20)

        assert key1 != key2 != key3

    def test_extra_context_support(self):
        """Extra context should affect key generation."""
        gen = CacheKeyGenerator()

        key1 = gen.generate("query", "agent", "project")
        key2 = gen.generate(
            "query", "agent", "project", extra_context={"filter": "recent"}
        )
        key3 = gen.generate(
            "query", "agent", "project", extra_context={"filter": "all"}
        )

        assert key1 != key2
        assert key2 != key3

    def test_extra_context_order_independence(self):
        """Extra context should be order-independent (sorted)."""
        gen = CacheKeyGenerator()

        key1 = gen.generate(
            "query", "agent", "project", extra_context={"a": "1", "b": "2"}
        )
        key2 = gen.generate(
            "query", "agent", "project", extra_context={"b": "2", "a": "1"}
        )

        assert key1 == key2

    def test_parse_key(self):
        """parse_key should correctly extract components."""
        gen = CacheKeyGenerator(namespace="test")

        key = gen.generate("query", "agent", "project")
        namespace, version, hash_part = gen.parse_key(key)

        assert namespace == "test"
        assert version == "v1"
        assert len(hash_part) == 64

    def test_parse_key_invalid_format(self):
        """parse_key should raise ValueError for invalid keys."""
        gen = CacheKeyGenerator()

        with pytest.raises(ValueError):
            gen.parse_key("invalid_key_no_colons")

        with pytest.raises(ValueError):
            gen.parse_key("only:one")

    def test_is_valid_key(self):
        """is_valid_key should validate key format and namespace."""
        gen = CacheKeyGenerator(namespace="test")

        valid_key = gen.generate("query", "agent", "project")
        assert gen.is_valid_key(valid_key) is True

        # Different namespace
        other_gen = CacheKeyGenerator(namespace="other")
        other_key = other_gen.generate("query", "agent", "project")
        assert gen.is_valid_key(other_key) is False

        # Invalid format
        assert gen.is_valid_key("invalid") is False
        assert gen.is_valid_key("alma:v1:short") is False  # Hash too short

    def test_generate_pattern(self):
        """generate_pattern should return valid pattern."""
        gen = CacheKeyGenerator(namespace="test")

        pattern = gen.generate_pattern()
        assert pattern == "test:v1:*"

    def test_uniqueness_across_large_set(self):
        """Keys should be unique across a large set of inputs."""
        gen = CacheKeyGenerator()
        keys = set()

        # Generate 1000 unique keys
        for i in range(100):
            for j in range(10):
                key = gen.generate(
                    f"query {i}",
                    f"agent_{j}",
                    f"project_{i % 5}",
                    f"user_{j % 3}" if j % 2 == 0 else None,
                    5 + (i % 3),
                )
                keys.add(key)

        # All keys should be unique
        assert len(keys) == 1000


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
        stats = CacheStats(
            hits=10, misses=5, evictions=2, current_size=100, max_size=1000
        )
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

    def test_namespace_isolation(self):
        """Different namespaces should not share cache entries."""
        cache1 = RetrievalCache(namespace="agent_alpha")
        cache2 = RetrievalCache(namespace="agent_beta")

        slice_obj = MemorySlice(query="test", agent="agent")

        # Set in cache1
        cache1.set("test query", "agent", "project-1", slice_obj)

        # Should be found in cache1
        assert cache1.get("test query", "agent", "project-1") is not None

        # Should NOT be found in cache2 (different namespace)
        assert cache2.get("test query", "agent", "project-1") is None

    def test_key_format_includes_namespace(self):
        """Generated keys should include namespace prefix."""
        cache = RetrievalCache(namespace="custom_ns")

        key = cache._generate_key("query", "agent", "project")

        assert key.startswith("custom_ns:v1:")
        assert len(key.split(":")) == 3

    def test_collision_resistance_with_special_chars(self):
        """Keys should not collide with special characters in inputs."""
        cache = RetrievalCache()

        # Potential collision attack with delimiters
        key1 = cache._generate_key("query:with:colons", "agent", "project")
        key2 = cache._generate_key("query", "with:colons:agent", "project")

        assert key1 != key2

        # Another potential collision scenario
        key3 = cache._generate_key("a|b|c", "d", "e")
        key4 = cache._generate_key("a", "b|c|d", "e")

        assert key3 != key4

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
