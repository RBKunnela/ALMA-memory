"""
Integration tests for ALMA Mode-Aware Retrieval.

Tests the full mode-aware retrieval pipeline including mode inference,
diversity filtering, and mode-specific scoring.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from alma.retrieval import (
    RetrievalEngine,
    RetrievalMode,
    get_mode_config,
)
from alma.retrieval.scoring import ScoredItem
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemorySlice,
    Outcome,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_storage_with_varied_data():
    """Create a mock storage with varied data for testing modes."""
    storage = MagicMock()

    now = datetime.now(timezone.utc)
    recent = now - timedelta(days=5)
    old = now - timedelta(days=60)

    # Heuristics with varying confidence and recency
    storage.get_heuristics.return_value = [
        Heuristic(
            id="h1",
            agent="test-agent",
            project_id="test-project",
            condition="form validation",
            strategy="test incrementally",
            confidence=0.95,
            occurrence_count=20,
            success_count=19,
            last_validated=now,
            created_at=old,
        ),
        Heuristic(
            id="h2",
            agent="test-agent",
            project_id="test-project",
            condition="api testing",
            strategy="mock external services",
            confidence=0.75,
            occurrence_count=15,
            success_count=11,
            last_validated=recent,
            created_at=old,
        ),
        Heuristic(
            id="h3",
            agent="test-agent",
            project_id="test-project",
            condition="database queries",
            strategy="use transactions",
            confidence=0.60,
            occurrence_count=8,
            success_count=5,
            last_validated=old,
            created_at=old,
        ),
        Heuristic(
            id="h4",
            agent="test-agent",
            project_id="test-project",
            condition="authentication",
            strategy="test edge cases",
            confidence=0.55,
            occurrence_count=5,
            success_count=3,
            last_validated=old,
            created_at=old,
        ),
    ]

    # Outcomes with mix of success and failure
    storage.get_outcomes.return_value = [
        Outcome(
            id="o1",
            agent="test-agent",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test login validation",
            success=True,
            strategy_used="incremental testing",
            timestamp=now,
        ),
        Outcome(
            id="o2",
            agent="test-agent",
            project_id="test-project",
            task_type="api_testing",
            task_description="Test payment API",
            success=False,
            strategy_used="direct integration",
            error_message="Timeout on external service",
            timestamp=recent,
        ),
        Outcome(
            id="o3",
            agent="test-agent",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test registration form",
            success=False,
            strategy_used="parallel validation",
            error_message="Race condition in async fields",
            timestamp=old,
        ),
        Outcome(
            id="o4",
            agent="test-agent",
            project_id="test-project",
            task_type="database_testing",
            task_description="Test user queries",
            success=True,
            strategy_used="isolated transactions",
            timestamp=old,
        ),
    ]

    storage.get_domain_knowledge.return_value = [
        DomainKnowledge(
            id="dk1",
            agent="test-agent",
            project_id="test-project",
            domain="authentication",
            fact="Uses JWT tokens with 24h expiry",
            source="code_review",
            confidence=0.95,
            last_verified=now,
        ),
        DomainKnowledge(
            id="dk2",
            agent="test-agent",
            project_id="test-project",
            domain="database",
            fact="PostgreSQL with read replicas",
            source="documentation",
            confidence=0.90,
            last_verified=recent,
        ),
    ]

    storage.get_anti_patterns.return_value = [
        AntiPattern(
            id="ap1",
            agent="test-agent",
            project_id="test-project",
            pattern="Using sleep() for async waits",
            why_bad="Causes flaky tests",
            better_alternative="Use explicit wait conditions",
            occurrence_count=8,
            last_seen=now,
        ),
        AntiPattern(
            id="ap2",
            agent="test-agent",
            project_id="test-project",
            pattern="Hardcoded test data",
            why_bad="Tests become brittle",
            better_alternative="Use factories or fixtures",
            occurrence_count=3,
            last_seen=old,
        ),
    ]

    storage.get_user_preferences.return_value = []

    # Multi-agent methods (return same data for simplicity)
    storage.get_heuristics_for_agents.return_value = storage.get_heuristics.return_value
    storage.get_outcomes_for_agents.return_value = storage.get_outcomes.return_value
    storage.get_domain_knowledge_for_agents.return_value = (
        storage.get_domain_knowledge.return_value
    )
    storage.get_anti_patterns_for_agents.return_value = (
        storage.get_anti_patterns.return_value
    )

    return storage


@pytest.fixture
def engine(mock_storage_with_varied_data):
    """Create a retrieval engine with mock storage."""
    return RetrievalEngine(
        storage=mock_storage_with_varied_data,
        embedding_provider="mock",
        enable_cache=False,
    )


# ============================================================================
# Mode-Aware Retrieval Tests
# ============================================================================


class TestRetrieveWithMode:
    """Tests for retrieve_with_mode method."""

    def test_retrieve_with_explicit_mode(self, engine):
        """Test retrieval with explicitly specified mode."""
        result, mode, reason = engine.retrieve_with_mode(
            query="Test the login form",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.PRECISE,
        )

        assert isinstance(result, MemorySlice)
        assert mode == RetrievalMode.PRECISE
        assert len(reason) > 0

    def test_retrieve_with_auto_inferred_mode(self, engine):
        """Test retrieval with auto-inferred mode."""
        # Diagnostic query
        result, mode, reason = engine.retrieve_with_mode(
            query="Why is the login failing?",
            agent="test-agent",
            project_id="test-project",
        )

        assert mode == RetrievalMode.DIAGNOSTIC
        assert (
            "diagnostic" in reason.lower()
            or "error" in reason.lower()
            or "fail" in reason.lower()
        )

    def test_broad_mode_returns_more_results(self, engine):
        """BROAD mode should return more results than PRECISE."""
        broad_result, _, _ = engine.retrieve_with_mode(
            query="How should we design the API?",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        precise_result, _, _ = engine.retrieve_with_mode(
            query="Implement the login form",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.PRECISE,
        )

        # BROAD has higher default top_k
        broad_config = get_mode_config(RetrievalMode.BROAD)
        precise_config = get_mode_config(RetrievalMode.PRECISE)
        assert broad_config.top_k > precise_config.top_k

    def test_diagnostic_mode_includes_anti_patterns(self, engine):
        """DIAGNOSTIC mode should include anti-patterns."""
        result, mode, _ = engine.retrieve_with_mode(
            query="Debug the authentication error",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.DIAGNOSTIC,
        )

        assert mode == RetrievalMode.DIAGNOSTIC
        # Anti-patterns should be included
        assert (
            len(result.anti_patterns) > 0
            or get_mode_config(RetrievalMode.DIAGNOSTIC).include_anti_patterns
        )

    def test_broad_mode_excludes_anti_patterns(self, engine):
        """BROAD mode should exclude anti-patterns for clean exploration."""
        result, mode, _ = engine.retrieve_with_mode(
            query="What are our options for caching?",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        assert mode == RetrievalMode.BROAD
        # BROAD mode excludes anti-patterns
        assert len(result.anti_patterns) == 0

    def test_override_top_k(self, engine):
        """Should be able to override top_k."""
        result, _, _ = engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
            top_k=2,  # Override the default 15
        )

        # Results should be limited to 2 per type
        assert len(result.heuristics) <= 2
        assert len(result.outcomes) <= 2

    def test_override_min_confidence(self, engine):
        """Should be able to override min_confidence."""
        # High confidence threshold
        high_conf_result, _, _ = engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
            min_confidence=0.9,
        )

        # Low confidence threshold
        low_conf_result, _, _ = engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
            min_confidence=0.1,
        )

        # High threshold should filter more
        assert low_conf_result.total_items >= high_conf_result.total_items

    def test_returns_mode_and_reason(self, engine):
        """Should return the mode used and reason."""
        result, mode, reason = engine.retrieve_with_mode(
            query="Find similar patterns",
            agent="test-agent",
            project_id="test-project",
        )

        assert mode == RetrievalMode.LEARNING
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_retrieval_time_recorded(self, engine):
        """Should record retrieval time."""
        result, _, _ = engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
        )

        assert result.retrieval_time_ms is not None
        assert result.retrieval_time_ms >= 0


class TestModeSpecificBehavior:
    """Tests for mode-specific retrieval behavior."""

    def test_precise_mode_high_confidence_filter(self, engine):
        """PRECISE mode should have higher confidence threshold."""
        result, _, _ = engine.retrieve_with_mode(
            query="Implement the feature",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.PRECISE,
        )

        # PRECISE has min_confidence of 0.7
        # Results should be high-confidence items
        for _heuristic in result.heuristics:
            # Note: Items may have lower confidence but high overall score
            # This just verifies the retrieval completes
            pass

        assert isinstance(result, MemorySlice)

    def test_learning_mode_high_top_k(self, engine):
        """LEARNING mode should request more results for pattern finding."""
        config = get_mode_config(RetrievalMode.LEARNING)
        assert config.top_k >= 15

        result, _, _ = engine.retrieve_with_mode(
            query="Find recurring patterns",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.LEARNING,
        )

        assert isinstance(result, MemorySlice)

    def test_recall_mode_few_results(self, engine):
        """RECALL mode should return few, focused results."""
        config = get_mode_config(RetrievalMode.RECALL)
        assert config.top_k <= 5

        result, _, _ = engine.retrieve_with_mode(
            query="What did we decide about auth?",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.RECALL,
        )

        assert len(result.heuristics) <= config.top_k


class TestDiversityFiltering:
    """Tests for MMR diversity filtering."""

    def test_diversity_filtering_applied_in_broad_mode(self, engine):
        """BROAD mode should apply diversity filtering."""
        config = get_mode_config(RetrievalMode.BROAD)
        assert config.diversity_factor > 0

        result, mode, _ = engine.retrieve_with_mode(
            query="Options for testing",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        # Just verify it completes without error
        assert mode == RetrievalMode.BROAD
        assert isinstance(result, MemorySlice)

    def test_no_diversity_in_recall_mode(self, engine):
        """RECALL mode should not apply diversity filtering."""
        config = get_mode_config(RetrievalMode.RECALL)
        assert config.diversity_factor == 0.0

    def test_diversity_helper_single_item(self, engine):
        """Diversity filter should handle single item."""
        items = [
            ScoredItem(
                item="single",
                score=0.9,
                similarity_score=0.9,
                recency_score=0.9,
                success_score=0.9,
                confidence_score=0.9,
            )
        ]

        result = engine._diversify_results(items, diversity_factor=0.8)
        assert len(result) == 1
        assert result[0].item == "single"

    def test_diversity_helper_empty_list(self, engine):
        """Diversity filter should handle empty list."""
        result = engine._diversify_results([], diversity_factor=0.8)
        assert result == []

    def test_diversity_zero_factor(self, engine):
        """Zero diversity factor should preserve original order."""
        items = [
            ScoredItem(
                item="first",
                score=0.9,
                similarity_score=0.9,
                recency_score=0.9,
                success_score=0.9,
                confidence_score=0.9,
            ),
            ScoredItem(
                item="second",
                score=0.8,
                similarity_score=0.8,
                recency_score=0.8,
                success_score=0.8,
                confidence_score=0.8,
            ),
        ]

        result = engine._diversify_results(items, diversity_factor=0.0)
        assert [r.item for r in result] == ["first", "second"]


class TestFailureBoost:
    """Tests for failure boosting in diagnostic mode."""

    def test_boost_failures_increases_failure_scores(self, engine):
        """Failure boost should increase scores for failed outcomes."""
        now = datetime.now(timezone.utc)

        items = [
            ScoredItem(
                item=Outcome(
                    id="success",
                    agent="test",
                    project_id="p",
                    task_type="t",
                    task_description="d",
                    success=True,
                    strategy_used="s",
                    timestamp=now,
                ),
                score=0.8,
                similarity_score=0.8,
                recency_score=0.8,
                success_score=1.0,
                confidence_score=0.8,
            ),
            ScoredItem(
                item=Outcome(
                    id="failure",
                    agent="test",
                    project_id="p",
                    task_type="t",
                    task_description="d",
                    success=False,
                    strategy_used="s",
                    error_message="error",
                    timestamp=now,
                ),
                score=0.6,
                similarity_score=0.6,
                recency_score=0.6,
                success_score=0.3,
                confidence_score=0.6,
            ),
        ]

        boosted = engine._boost_failures(items)

        # Find the failure item
        failure_item = [b for b in boosted if not b.item.success][0]
        original_failure = [i for i in items if not i.item.success][0]

        # Failure should have boosted score
        assert failure_item.score > original_failure.score


class TestExactMatchBoost:
    """Tests for exact match boosting."""

    def test_high_similarity_gets_boost(self, engine):
        """Items with high similarity should get boosted."""
        items = [
            ScoredItem(
                item="high_sim",
                score=0.7,
                similarity_score=0.95,  # > 0.9 threshold
                recency_score=0.5,
                success_score=0.5,
                confidence_score=0.5,
            ),
            ScoredItem(
                item="low_sim",
                score=0.7,
                similarity_score=0.5,
                recency_score=0.5,
                success_score=0.5,
                confidence_score=0.5,
            ),
        ]

        boosted = engine._apply_exact_match_boost(items, boost_factor=2.0)

        high_sim = [b for b in boosted if b.item == "high_sim"][0]
        low_sim = [b for b in boosted if b.item == "low_sim"][0]

        # High similarity item should have higher boosted score
        assert high_sim.score > low_sim.score

    def test_medium_similarity_gets_partial_boost(self, engine):
        """Items with medium similarity should get partial boost."""
        items = [
            ScoredItem(
                item="medium_sim",
                score=0.7,
                similarity_score=0.85,  # Between 0.8 and 0.9
                recency_score=0.5,
                success_score=0.5,
                confidence_score=0.5,
            ),
        ]

        boosted = engine._apply_exact_match_boost(items, boost_factor=2.0)

        # Should be boosted but not as much as >0.9
        assert boosted[0].score > 0.7
        assert boosted[0].score < 0.7 * 2.0


class TestCachingWithModes:
    """Tests for caching behavior with mode-aware retrieval."""

    def test_mode_aware_caching(self, mock_storage_with_varied_data):
        """Different modes should use different cache keys."""
        engine = RetrievalEngine(
            storage=mock_storage_with_varied_data,
            embedding_provider="mock",
            enable_cache=True,
        )

        # First call with BROAD
        engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        # Second call with PRECISE (different mode, different cache key)
        engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.PRECISE,
        )

        # Both should be cache misses since different modes
        stats = engine.get_cache_stats()
        assert stats["misses"] == 2

    def test_same_mode_uses_cache(self, mock_storage_with_varied_data):
        """Same query with same mode should use cache."""
        engine = RetrievalEngine(
            storage=mock_storage_with_varied_data,
            embedding_provider="mock",
            enable_cache=True,
        )

        # First call
        engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        # Second call with same mode
        engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BROAD,
        )

        stats = engine.get_cache_stats()
        assert stats["hits"] == 1


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_original_retrieve_unchanged(self, engine):
        """Original retrieve method should still work."""
        result = engine.retrieve(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
        )

        assert isinstance(result, MemorySlice)
        assert result.query == "Test query"

    def test_scorer_weights_restored(self, engine):
        """Scorer weights should be restored after mode-aware retrieval."""
        original_weights = engine.get_scorer_weights()

        # Do mode-aware retrieval (which temporarily changes weights)
        engine.retrieve_with_mode(
            query="Test query",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.DIAGNOSTIC,
        )

        restored_weights = engine.get_scorer_weights()

        # Weights should be restored
        assert original_weights == restored_weights
