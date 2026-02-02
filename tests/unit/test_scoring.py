"""
Unit tests for ALMA Memory Scoring.
"""

from datetime import datetime, timedelta, timezone

import pytest

from alma.retrieval.scoring import (
    MemoryScorer,
    ScoredItem,
    ScoringWeights,
    compute_composite_score,
)
from alma.types import AntiPattern, Heuristic, Outcome


class TestScoringWeights:
    """Tests for ScoringWeights configuration."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        weights = ScoringWeights()
        total = (
            weights.similarity
            + weights.recency
            + weights.success_rate
            + weights.confidence
        )
        assert abs(total - 1.0) < 0.01

    def test_custom_weights_normalized(self):
        """Custom weights should be normalized to sum to 1.0."""
        # Intentionally don't sum to 1
        weights = ScoringWeights(
            similarity=0.5,
            recency=0.5,
            success_rate=0.5,
            confidence=0.5,
        )
        total = (
            weights.similarity
            + weights.recency
            + weights.success_rate
            + weights.confidence
        )
        assert abs(total - 1.0) < 0.01

    def test_zero_weights_handled(self):
        """Test that zero weights work correctly."""
        weights = ScoringWeights(
            similarity=1.0,
            recency=0.0,
            success_rate=0.0,
            confidence=0.0,
        )
        assert weights.similarity == 1.0


class TestRecencyScoring:
    """Tests for recency score calculation."""

    def test_recency_score_now_is_one(self):
        """A timestamp of now should give score ~1.0."""
        scorer = MemoryScorer()
        now = datetime.now(timezone.utc)
        score = scorer._compute_recency_score(now)
        assert score > 0.99

    def test_recency_half_life(self):
        """Score should halve after half_life days."""
        half_life = 30.0
        scorer = MemoryScorer(recency_half_life_days=half_life)

        now = datetime.now(timezone.utc)
        after_half_life = now - timedelta(days=half_life)

        score = scorer._compute_recency_score(after_half_life)
        # Should be approximately 0.5
        assert 0.45 < score < 0.55

    def test_recency_double_half_life(self):
        """Score should quarter after 2x half_life days."""
        half_life = 30.0
        scorer = MemoryScorer(recency_half_life_days=half_life)

        now = datetime.now(timezone.utc)
        after_double = now - timedelta(days=half_life * 2)

        score = scorer._compute_recency_score(after_double)
        # Should be approximately 0.25
        assert 0.20 < score < 0.30

    def test_very_old_memory_low_score(self):
        """Very old memories should have very low scores."""
        scorer = MemoryScorer(recency_half_life_days=30.0)

        now = datetime.now(timezone.utc)
        very_old = now - timedelta(days=365)

        score = scorer._compute_recency_score(very_old)
        assert score < 0.01


class TestHeuristicScoring:
    """Tests for heuristic scoring."""

    @pytest.fixture
    def sample_heuristics(self):
        """Create sample heuristics for testing."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)

        return [
            Heuristic(
                id="h1",
                agent="test",
                project_id="p1",
                condition="condition 1",
                strategy="strategy 1",
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=old,
            ),
            Heuristic(
                id="h2",
                agent="test",
                project_id="p1",
                condition="condition 2",
                strategy="strategy 2",
                confidence=0.5,
                occurrence_count=10,
                success_count=5,
                last_validated=old,
                created_at=old,
            ),
        ]

    def test_higher_confidence_scores_higher(self, sample_heuristics):
        """Higher confidence heuristics should score higher."""
        scorer = MemoryScorer()
        scored = scorer.score_heuristics(sample_heuristics)

        # h1 has higher confidence and recency
        assert scored[0].item.id == "h1"
        assert scored[0].confidence_score > scored[1].confidence_score

    def test_scoring_with_similarities(self, sample_heuristics):
        """Test scoring with pre-computed similarities."""
        scorer = MemoryScorer()
        similarities = [0.5, 1.0]  # h2 is more similar

        scored = scorer.score_heuristics(sample_heuristics, similarities)

        # h2 has higher similarity, but h1 might still win due to other factors
        assert scored[0].similarity_score in [0.5, 1.0]


class TestOutcomeScoring:
    """Tests for outcome scoring."""

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes for testing."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        return [
            Outcome(
                id="o1",
                agent="test",
                project_id="p1",
                task_type="test",
                task_description="task 1",
                success=True,
                strategy_used="strategy 1",
                timestamp=now,
            ),
            Outcome(
                id="o2",
                agent="test",
                project_id="p1",
                task_type="test",
                task_description="task 2",
                success=False,
                strategy_used="strategy 2",
                timestamp=old,
            ),
        ]

    def test_success_scores_higher(self, sample_outcomes):
        """Successful outcomes should score higher than failures."""
        scorer = MemoryScorer()
        scored = scorer.score_outcomes(sample_outcomes)

        successful = [s for s in scored if s.item.success][0]
        failed = [s for s in scored if not s.item.success][0]

        assert successful.success_score == 1.0
        assert failed.success_score == 0.3

    def test_failed_outcomes_included(self, sample_outcomes):
        """Failed outcomes should still be included (for learning)."""
        scorer = MemoryScorer()
        scored = scorer.score_outcomes(sample_outcomes)

        assert len(scored) == 2
        assert any(not s.item.success for s in scored)


class TestAntiPatternScoring:
    """Tests for anti-pattern scoring."""

    def test_frequent_anti_patterns_score_higher(self):
        """Anti-patterns seen more often should score higher."""
        now = datetime.now(timezone.utc)

        anti_patterns = [
            AntiPattern(
                id="ap1",
                agent="test",
                project_id="p1",
                pattern="pattern 1",
                why_bad="bad 1",
                better_alternative="better 1",
                occurrence_count=10,
                last_seen=now,
            ),
            AntiPattern(
                id="ap2",
                agent="test",
                project_id="p1",
                pattern="pattern 2",
                why_bad="bad 2",
                better_alternative="better 2",
                occurrence_count=2,
                last_seen=now,
            ),
        ]

        scorer = MemoryScorer()
        scored = scorer.score_anti_patterns(anti_patterns)

        # Higher occurrence count should contribute to higher score
        ap1_score = [s for s in scored if s.item.id == "ap1"][0]
        ap2_score = [s for s in scored if s.item.id == "ap2"][0]

        assert ap1_score.success_score > ap2_score.success_score


class TestCompositeScoreFunction:
    """Tests for the standalone composite score function."""

    def test_perfect_scores(self):
        """Test with all maximum values."""
        score = compute_composite_score(
            similarity=1.0,
            recency_days=0,
            success_rate=1.0,
            confidence=1.0,
        )
        assert score > 0.95

    def test_minimum_scores(self):
        """Test with minimum values."""
        score = compute_composite_score(
            similarity=0.0,
            recency_days=365,
            success_rate=0.0,
            confidence=0.0,
        )
        assert score < 0.1

    def test_custom_weights_applied(self):
        """Test that custom weights change the score."""
        # Default weights
        default_score = compute_composite_score(
            similarity=1.0,
            recency_days=30,
            success_rate=0.5,
            confidence=0.5,
        )

        # Heavy similarity weighting
        custom_weights = ScoringWeights(
            similarity=0.9,
            recency=0.03,
            success_rate=0.03,
            confidence=0.04,
        )
        custom_score = compute_composite_score(
            similarity=1.0,
            recency_days=30,
            success_rate=0.5,
            confidence=0.5,
            weights=custom_weights,
        )

        # Custom should be higher since similarity is maxed
        assert custom_score > default_score


class TestScoreThreshold:
    """Tests for score threshold filtering."""

    def test_filter_low_scores(self):
        """Test that low scores are filtered out."""
        scorer = MemoryScorer()

        items = [
            ScoredItem(
                item="high",
                score=0.9,
                similarity_score=0.9,
                recency_score=0.9,
                success_score=0.9,
                confidence_score=0.9,
            ),
            ScoredItem(
                item="medium",
                score=0.5,
                similarity_score=0.5,
                recency_score=0.5,
                success_score=0.5,
                confidence_score=0.5,
            ),
            ScoredItem(
                item="low",
                score=0.1,
                similarity_score=0.1,
                recency_score=0.1,
                success_score=0.1,
                confidence_score=0.1,
            ),
        ]

        filtered = scorer.apply_score_threshold(items, min_score=0.4)

        assert len(filtered) == 2
        assert all(item.score >= 0.4 for item in filtered)
