"""
Unit tests for ALMA Forgetting Mechanism.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from alma.learning.forgetting import (
    ForgettingEngine,
    PrunePolicy,
    PruneResult,
    PruneSummary,
    PruneReason,
)
from alma.types import Heuristic, Outcome, DomainKnowledge, AntiPattern


class TestPrunePolicy:
    """Tests for PrunePolicy configuration."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        policy = PrunePolicy()

        assert policy.outcome_max_age_days == 90
        assert policy.heuristic_min_confidence == 0.3
        assert policy.max_heuristics_per_agent == 100

    def test_custom_values(self):
        """Test custom policy values."""
        policy = PrunePolicy(
            outcome_max_age_days=30,
            heuristic_min_confidence=0.5,
            max_heuristics_per_agent=50,
        )

        assert policy.outcome_max_age_days == 30
        assert policy.heuristic_min_confidence == 0.5
        assert policy.max_heuristics_per_agent == 50


class TestPruneSummary:
    """Tests for PruneSummary."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        summary = PruneSummary(
            outcomes_pruned=10,
            heuristics_pruned=5,
            knowledge_pruned=2,
            total_pruned=17,
        )

        d = summary.to_dict()

        assert d["outcomes_pruned"] == 10
        assert d["heuristics_pruned"] == 5
        assert d["total_pruned"] == 17


class TestForgettingEngine:
    """Tests for ForgettingEngine."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage with sample data."""
        storage = MagicMock()
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=120)  # Older than default 90 days

        storage.get_outcomes.return_value = [
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Recent task",
                success=True,
                strategy_used="strategy1",
                timestamp=now,
            ),
            Outcome(
                id="o2",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Old task",
                success=True,
                strategy_used="strategy2",
                timestamp=old,
            ),
        ]

        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="helena",
                project_id="test",
                condition="condition1",
                strategy="strategy1",
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="h2",
                agent="helena",
                project_id="test",
                condition="condition2",
                strategy="strategy2",
                confidence=0.2,  # Below threshold
                occurrence_count=10,
                success_count=2,
                last_validated=old,
                created_at=old,
            ),
        ]

        storage.get_domain_knowledge.return_value = []
        storage.get_anti_patterns.return_value = []
        storage.get_stats.return_value = {"agents": ["helena"]}
        storage.delete_outcomes_older_than.return_value = 1
        storage.delete_heuristic.return_value = True

        return storage

    def test_prune_identifies_stale_outcomes(self, mock_storage):
        """Test that stale outcomes are identified for pruning."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", dry_run=True)

        # One old outcome should be identified
        assert summary.outcomes_pruned >= 1

    def test_prune_identifies_low_confidence_heuristics(self, mock_storage):
        """Test that low confidence heuristics are identified."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", dry_run=True)

        # h2 has confidence 0.2, below threshold 0.3
        low_conf = [
            r for r in summary.pruned_items
            if r.reason == PruneReason.LOW_CONFIDENCE
        ]
        assert len(low_conf) >= 1

    def test_dry_run_no_deletion(self, mock_storage):
        """Test that dry run doesn't delete anything."""
        engine = ForgettingEngine(mock_storage)

        engine.prune("test", dry_run=True)

        # delete_heuristic should not be called in dry run
        mock_storage.delete_heuristic.assert_not_called()

    def test_actual_prune_deletes(self, mock_storage):
        """Test that actual prune deletes items."""
        engine = ForgettingEngine(mock_storage)

        engine.prune("test", dry_run=False)

        # delete methods should be called
        mock_storage.delete_outcomes_older_than.assert_called()

    def test_compute_decay_score(self, mock_storage):
        """Test decay score calculation."""
        engine = ForgettingEngine(mock_storage)

        # Perfect item
        score_perfect = engine.compute_decay_score(
            item_age_days=0,
            confidence=1.0,
            success_rate=1.0,
            occurrence_count=20,
        )

        # Old, low confidence item
        score_poor = engine.compute_decay_score(
            item_age_days=90,
            confidence=0.2,
            success_rate=0.3,
            occurrence_count=2,
        )

        assert score_perfect > score_poor
        assert score_perfect > 0.8
        assert score_poor < 0.4

    def test_identify_candidates(self, mock_storage):
        """Test candidate identification."""
        engine = ForgettingEngine(mock_storage)

        candidates = engine.identify_candidates("test", max_candidates=10)

        assert len(candidates) > 0
        # Candidates should be sorted by score (lowest first)
        if len(candidates) > 1:
            assert candidates[0]["score"] <= candidates[-1]["score"]

    def test_custom_policy(self, mock_storage):
        """Test using custom prune policy."""
        policy = PrunePolicy(
            outcome_max_age_days=30,
            heuristic_min_confidence=0.5,
        )
        engine = ForgettingEngine(mock_storage, policy=policy)

        # With stricter policy, more items should be identified
        summary = engine.prune("test", dry_run=True)

        # With higher confidence threshold, both heuristics might be flagged
        assert summary.heuristics_pruned >= 1

    def test_agent_specific_prune(self, mock_storage):
        """Test pruning for specific agent."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", agent="helena", dry_run=True)

        # Should only affect helena's items
        for item in summary.pruned_items:
            assert item.agent == "helena"


class TestPruneReason:
    """Tests for PruneReason enum."""

    def test_enum_values(self):
        """Test that all expected values exist."""
        assert PruneReason.STALE.value == "stale"
        assert PruneReason.LOW_CONFIDENCE.value == "low_confidence"
        assert PruneReason.LOW_SUCCESS_RATE.value == "low_success"
        assert PruneReason.QUOTA_EXCEEDED.value == "quota"


class TestPruneResult:
    """Tests for PruneResult dataclass."""

    def test_creation(self):
        """Test creating a prune result."""
        result = PruneResult(
            reason=PruneReason.STALE,
            item_type="outcome",
            item_id="o1",
            agent="helena",
            project_id="test",
            details="Older than 90 days",
        )

        assert result.reason == PruneReason.STALE
        assert result.item_type == "outcome"
        assert result.agent == "helena"
