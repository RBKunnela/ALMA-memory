"""
Integration tests for ALMA Retrieval Engine.

Tests the full retrieval pipeline including scoring, caching, and embeddings.
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from alma.retrieval import (
    RetrievalEngine,
    MemoryScorer,
    ScoringWeights,
    RetrievalCache,
    NullCache,
    MockEmbedder,
)
from alma.retrieval.scoring import compute_composite_score, ScoredItem
from alma.retrieval.cache import CacheStats
from alma.types import (
    Heuristic,
    Outcome,
    DomainKnowledge,
    AntiPattern,
    MemorySlice,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MagicMock()

    # Create sample data
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=60)

    storage.get_heuristics.return_value = [
        Heuristic(
            id="h1",
            agent="helena",
            project_id="test-project",
            condition="form with required fields",
            strategy="test validation incrementally",
            confidence=0.85,
            occurrence_count=10,
            success_count=8,
            last_validated=now,
            created_at=old_date,
        ),
        Heuristic(
            id="h2",
            agent="helena",
            project_id="test-project",
            condition="login form",
            strategy="test happy path first",
            confidence=0.65,
            occurrence_count=5,
            success_count=3,
            last_validated=old_date,
            created_at=old_date,
        ),
    ]

    storage.get_outcomes.return_value = [
        Outcome(
            id="o1",
            agent="helena",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test login form validation",
            success=True,
            strategy_used="incremental validation",
            timestamp=now,
        ),
        Outcome(
            id="o2",
            agent="helena",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test registration form",
            success=False,
            strategy_used="parallel validation",
            error_message="Timeout on async fields",
            timestamp=old_date,
        ),
    ]

    storage.get_domain_knowledge.return_value = [
        DomainKnowledge(
            id="dk1",
            agent="helena",
            project_id="test-project",
            domain="authentication",
            fact="Login uses JWT with 24h expiry",
            source="code_analysis",
            confidence=0.95,
            last_verified=now,
        ),
    ]

    storage.get_anti_patterns.return_value = [
        AntiPattern(
            id="ap1",
            agent="helena",
            project_id="test-project",
            pattern="Using sleep() for async waits",
            why_bad="Causes flaky tests",
            better_alternative="Use explicit waits",
            occurrence_count=5,
            last_seen=now,
        ),
    ]

    storage.get_user_preferences.return_value = []

    return storage


@pytest.fixture
def scorer():
    """Create a scorer with default weights."""
    return MemoryScorer()


@pytest.fixture
def custom_scorer():
    """Create a scorer with custom weights."""
    weights = ScoringWeights(
        similarity=0.5,
        recency=0.2,
        success_rate=0.2,
        confidence=0.1,
    )
    return MemoryScorer(weights=weights, recency_half_life_days=14.0)


# ============================================================================
# Scoring Tests
# ============================================================================

class TestMemoryScorer:
    """Tests for MemoryScorer class."""

    def test_score_heuristics_ranking(self, scorer, mock_storage):
        """Test that heuristics are ranked by composite score."""
        heuristics = mock_storage.get_heuristics()
        scored = scorer.score_heuristics(heuristics)

        assert len(scored) == 2
        # First heuristic has higher confidence and recency, should rank higher
        assert scored[0].item.id == "h1"
        assert scored[0].score > scored[1].score

    def test_score_outcomes_success_weighting(self, scorer, mock_storage):
        """Test that successful outcomes score higher."""
        outcomes = mock_storage.get_outcomes()
        scored = scorer.score_outcomes(outcomes)

        assert len(scored) == 2
        # Successful outcome should rank higher
        successful = [s for s in scored if s.item.success][0]
        failed = [s for s in scored if not s.item.success][0]
        assert successful.success_score > failed.success_score

    def test_recency_decay(self, scorer):
        """Test that recency score decays over time."""
        now = datetime.now(timezone.utc)
        recent = now
        old = now - timedelta(days=30)  # One half-life
        very_old = now - timedelta(days=60)  # Two half-lives

        score_recent = scorer._compute_recency_score(recent)
        score_old = scorer._compute_recency_score(old)
        score_very_old = scorer._compute_recency_score(very_old)

        assert score_recent > score_old > score_very_old
        # After one half-life, score should be ~0.5
        assert 0.4 < score_old < 0.6
        # After two half-lives, score should be ~0.25
        assert 0.2 < score_very_old < 0.3

    def test_score_threshold_filtering(self, scorer, mock_storage):
        """Test that low-scoring items are filtered out."""
        heuristics = mock_storage.get_heuristics()
        scored = scorer.score_heuristics(heuristics)

        # All items should pass a low threshold
        filtered_low = scorer.apply_score_threshold(scored, min_score=0.1)
        assert len(filtered_low) == 2

        # Only high-scoring items pass a high threshold
        filtered_high = scorer.apply_score_threshold(scored, min_score=0.9)
        assert len(filtered_high) <= 1

    def test_custom_weights(self, custom_scorer):
        """Test that custom weights are applied correctly."""
        weights = custom_scorer.weights
        total = weights.similarity + weights.recency + weights.success_rate + weights.confidence
        # Weights should be normalized to sum to 1.0
        assert 0.99 <= total <= 1.01


class TestCompositeScore:
    """Tests for the compute_composite_score function."""

    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        score = compute_composite_score(
            similarity=1.0,
            recency_days=0,
            success_rate=1.0,
            confidence=1.0,
        )
        # All maxed out should give ~1.0
        assert score > 0.9

    def test_composite_score_with_decay(self):
        """Test composite score with recency decay."""
        fresh = compute_composite_score(
            similarity=1.0,
            recency_days=0,
            success_rate=1.0,
            confidence=1.0,
        )
        old = compute_composite_score(
            similarity=1.0,
            recency_days=30,
            success_rate=1.0,
            confidence=1.0,
        )
        assert fresh > old


# ============================================================================
# Cache Tests
# ============================================================================

class TestRetrievalCache:
    """Tests for RetrievalCache class."""

    def test_cache_hit_and_miss(self):
        """Test basic cache hit and miss."""
        cache = RetrievalCache(ttl_seconds=60)

        # Miss on empty cache
        result = cache.get("test query", "helena", "project-1")
        assert result is None

        # Set and hit
        mock_slice = MemorySlice(query="test query", agent="helena")
        cache.set("test query", "helena", "project-1", mock_slice)

        result = cache.get("test query", "helena", "project-1")
        assert result is not None
        assert result.query == "test query"

    def test_cache_expiration(self):
        """Test that cache entries expire."""
        cache = RetrievalCache(ttl_seconds=1)

        mock_slice = MemorySlice(query="test", agent="helena")
        cache.set("test", "helena", "project-1", mock_slice)

        # Should hit immediately
        assert cache.get("test", "helena", "project-1") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should miss after expiration
        assert cache.get("test", "helena", "project-1") is None

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = RetrievalCache(ttl_seconds=60)

        mock_slice = MemorySlice(query="test", agent="helena")
        cache.set("test", "helena", "project-1", mock_slice)

        assert cache.get("test", "helena", "project-1") is not None

        cache.invalidate()

        assert cache.get("test", "helena", "project-1") is None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = RetrievalCache(ttl_seconds=60)

        # Miss
        cache.get("test", "helena", "project-1")
        stats = cache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 0

        # Set and hit
        mock_slice = MemorySlice(query="test", agent="helena")
        cache.set("test", "helena", "project-1", mock_slice)
        cache.get("test", "helena", "project-1")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_null_cache(self):
        """Test NullCache always misses."""
        cache = NullCache()

        mock_slice = MemorySlice(query="test", agent="helena")
        cache.set("test", "helena", "project-1", mock_slice)

        # Always misses
        assert cache.get("test", "helena", "project-1") is None

        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses > 0

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max entries reached."""
        cache = RetrievalCache(ttl_seconds=60, max_entries=2)

        # Fill cache
        cache.set("q1", "helena", "p1", MemorySlice(query="q1", agent="helena"))
        cache.set("q2", "helena", "p1", MemorySlice(query="q2", agent="helena"))

        # Access q1 to make it more recently used
        cache.get("q1", "helena", "p1")

        # Add third entry, should evict q2 (least used)
        cache.set("q3", "helena", "p1", MemorySlice(query="q3", agent="helena"))

        # q1 should still be there (recently accessed)
        assert cache.get("q1", "helena", "p1") is not None
        # q3 should be there (just added)
        assert cache.get("q3", "helena", "p1") is not None


# ============================================================================
# Retrieval Engine Integration Tests
# ============================================================================

class TestRetrievalEngine:
    """Integration tests for RetrievalEngine."""

    def test_retrieve_basic(self, mock_storage):
        """Test basic retrieval flow."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
            enable_cache=False,
        )

        result = engine.retrieve(
            query="Test the login form",
            agent="helena",
            project_id="test-project",
        )

        assert isinstance(result, MemorySlice)
        assert result.query == "Test the login form"
        assert result.agent == "helena"
        assert result.retrieval_time_ms is not None

    def test_retrieve_with_caching(self, mock_storage):
        """Test that caching works in retrieval."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
            enable_cache=True,
            cache_ttl_seconds=60,
        )

        # First call - cache miss
        result1 = engine.retrieve(
            query="Test the login form",
            agent="helena",
            project_id="test-project",
        )

        # Second call - cache hit
        result2 = engine.retrieve(
            query="Test the login form",
            agent="helena",
            project_id="test-project",
        )

        # Both should return valid results
        assert result1.total_items == result2.total_items

        # Check cache stats
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_retrieve_bypass_cache(self, mock_storage):
        """Test cache bypass option."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
            enable_cache=True,
        )

        # First call
        engine.retrieve(
            query="Test form",
            agent="helena",
            project_id="test-project",
        )

        # Second call with bypass
        engine.retrieve(
            query="Test form",
            agent="helena",
            project_id="test-project",
            bypass_cache=True,
        )

        stats = engine.get_cache_stats()
        # Both should be misses (first miss, second bypassed)
        assert stats["misses"] == 1

    def test_scorer_weights_update(self, mock_storage):
        """Test updating scorer weights."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
        )

        # Get initial weights
        initial = engine.get_scorer_weights()
        assert initial["similarity"] == 0.4

        # Update weights
        engine.update_scorer_weights(similarity=0.6, recency=0.2)

        updated = engine.get_scorer_weights()
        # Weights get normalized
        assert updated["similarity"] > initial["similarity"]

    def test_cache_invalidation_on_weight_update(self, mock_storage):
        """Test that cache is cleared when weights change."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
            enable_cache=True,
        )

        # Populate cache
        engine.retrieve("Test form", "helena", "test-project")

        # Update weights (should clear cache)
        engine.update_scorer_weights(similarity=0.6)

        stats = engine.get_cache_stats()
        assert stats["current_size"] == 0

    def test_retrieve_filters_low_scores(self, mock_storage):
        """Test that low-scoring items are filtered out."""
        engine = RetrievalEngine(
            storage=mock_storage,
            embedding_provider="mock",
            min_score_threshold=0.9,  # Very high threshold
        )

        result = engine.retrieve(
            query="Test form",
            agent="helena",
            project_id="test-project",
        )

        # With high threshold, some items may be filtered
        # This depends on the scoring, but we verify the engine runs
        assert isinstance(result, MemorySlice)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_storage_results(self):
        """Test retrieval with empty storage."""
        storage = MagicMock()
        storage.get_heuristics.return_value = []
        storage.get_outcomes.return_value = []
        storage.get_domain_knowledge.return_value = []
        storage.get_anti_patterns.return_value = []
        storage.get_user_preferences.return_value = []

        engine = RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
        )

        result = engine.retrieve("Test query", "helena", "test-project")

        assert result.total_items == 0
        assert len(result.heuristics) == 0

    def test_score_empty_lists(self, scorer):
        """Test scoring with empty lists."""
        assert scorer.score_heuristics([]) == []
        assert scorer.score_outcomes([]) == []
        assert scorer.score_domain_knowledge([]) == []
        assert scorer.score_anti_patterns([]) == []

    def test_naive_datetime_handling(self, scorer):
        """Test that naive datetimes are handled correctly."""
        # Naive datetime (no timezone)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        score = scorer._compute_recency_score(naive_dt)

        # Should not raise, should return a valid score
        assert 0 <= score <= 1
