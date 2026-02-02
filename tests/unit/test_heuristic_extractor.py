"""
Unit tests for ALMA Heuristic Extraction.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from alma.learning.heuristic_extractor import (
    ExtractionResult,
    HeuristicExtractor,
    PatternCandidate,
    extract_heuristics_from_outcome,
)
from alma.types import Heuristic, MemoryScope, Outcome


class TestPatternCandidate:
    """Tests for PatternCandidate dataclass."""

    def test_success_rate(self):
        """Test success rate calculation."""
        pattern = PatternCandidate(
            task_type="testing",
            strategy="test strategy",
            occurrence_count=10,
            success_count=8,
            failure_count=2,
        )

        assert pattern.success_rate == 0.8

    def test_success_rate_zero_occurrences(self):
        """Test success rate with zero occurrences."""
        pattern = PatternCandidate(
            task_type="testing",
            strategy="test strategy",
            occurrence_count=0,
            success_count=0,
            failure_count=0,
        )

        assert pattern.success_rate == 0.0

    def test_confidence_with_sample_size(self):
        """Test that confidence accounts for sample size."""
        # Small sample
        small = PatternCandidate(
            task_type="testing",
            strategy="strategy",
            occurrence_count=3,
            success_count=3,
            failure_count=0,
        )

        # Large sample
        large = PatternCandidate(
            task_type="testing",
            strategy="strategy",
            occurrence_count=20,
            success_count=20,
            failure_count=0,
        )

        # Both have 100% success rate, but large sample should have higher confidence
        assert large.confidence > small.confidence
        assert small.confidence > 0.5  # Still decent
        assert large.confidence > 0.9  # High confidence


class TestHeuristicExtractor:
    """Tests for HeuristicExtractor."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock()
        storage.get_heuristics.return_value = []
        storage.save_heuristic.return_value = True
        return storage

    @pytest.fixture
    def scopes(self):
        """Create sample scopes."""
        return {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }

    @pytest.fixture
    def sample_outcomes(self):
        """Create sample outcomes with patterns."""
        now = datetime.now(timezone.utc)
        return [
            # Pattern 1: successful testing strategy
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="form_testing",
                task_description="Test login",
                success=True,
                strategy_used="test validation first",
                timestamp=now,
            ),
            Outcome(
                id="o2",
                agent="helena",
                project_id="test",
                task_type="form_testing",
                task_description="Test signup",
                success=True,
                strategy_used="test validation first, then submit",
                timestamp=now,
            ),
            Outcome(
                id="o3",
                agent="helena",
                project_id="test",
                task_type="form_testing",
                task_description="Test profile",
                success=True,
                strategy_used="test validation then submit",
                timestamp=now,
            ),
            # Pattern 2: failed strategy
            Outcome(
                id="o4",
                agent="helena",
                project_id="test",
                task_type="form_testing",
                task_description="Test payment",
                success=False,
                strategy_used="submit immediately",
                timestamp=now,
            ),
        ]

    def test_extract_creates_heuristic(self, mock_storage, scopes, sample_outcomes):
        """Test that extraction creates heuristics from patterns."""
        mock_storage.get_outcomes.return_value = sample_outcomes

        extractor = HeuristicExtractor(
            storage=mock_storage,
            scopes=scopes,
            min_occurrences=3,
            min_confidence=0.5,
        )

        result = extractor.extract("test")

        assert result.heuristics_created >= 1
        mock_storage.save_heuristic.assert_called()

    def test_extract_respects_min_occurrences(self, mock_storage, scopes):
        """Test that extraction respects minimum occurrences."""
        # Only 2 similar outcomes, below threshold of 3
        outcomes = [
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Test 1",
                success=True,
                strategy_used="unique strategy",
                timestamp=datetime.now(timezone.utc),
            ),
            Outcome(
                id="o2",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Test 2",
                success=True,
                strategy_used="unique strategy too",
                timestamp=datetime.now(timezone.utc),
            ),
        ]
        mock_storage.get_outcomes.return_value = outcomes

        extractor = HeuristicExtractor(
            storage=mock_storage,
            scopes=scopes,
            min_occurrences=3,
        )

        result = extractor.extract("test")

        assert result.heuristics_created == 0
        assert "insufficient_occurrences" in str(result.rejected_reasons)

    def test_extract_respects_min_confidence(self, mock_storage, scopes):
        """Test that extraction respects minimum confidence."""
        # Mixed results = low confidence
        outcomes = [
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Test 1",
                success=True,
                strategy_used="same strategy",
                timestamp=datetime.now(timezone.utc),
            ),
            Outcome(
                id="o2",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Test 2",
                success=False,
                strategy_used="same strategy",
                timestamp=datetime.now(timezone.utc),
            ),
            Outcome(
                id="o3",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Test 3",
                success=False,
                strategy_used="same strategy",
                timestamp=datetime.now(timezone.utc),
            ),
        ]
        mock_storage.get_outcomes.return_value = outcomes

        extractor = HeuristicExtractor(
            storage=mock_storage,
            scopes=scopes,
            min_occurrences=3,
            min_confidence=0.6,  # Requires >60% success
        )

        result = extractor.extract("test")

        assert result.heuristics_created == 0
        assert "low_confidence" in str(result.rejected_reasons)

    def test_strategies_similar(self, mock_storage, scopes):
        """Test strategy similarity detection."""
        extractor = HeuristicExtractor(
            storage=mock_storage,
            scopes=scopes,
            strategy_similarity_threshold=0.5,
        )

        # Similar strategies
        assert extractor._strategies_similar(
            "test validation first then submit", "test validation, then submit form"
        )

        # Different strategies
        assert not extractor._strategies_similar(
            "use database query", "click the button"
        )

    def test_normalize_strategy(self, mock_storage, scopes):
        """Test strategy normalization."""
        extractor = HeuristicExtractor(mock_storage, scopes)

        words = extractor._normalize_strategy(
            "First, test the validation. Then, submit the form."
        )

        # Stop words should be removed
        assert "the" not in words
        assert "first" not in words
        assert "test" in words or "validation" in words

    def test_empty_outcomes(self, mock_storage, scopes):
        """Test extraction with no outcomes."""
        mock_storage.get_outcomes.return_value = []

        extractor = HeuristicExtractor(mock_storage, scopes)
        result = extractor.extract("test")

        assert result.heuristics_created == 0
        assert result.patterns_analyzed == 0


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ExtractionResult(
            heuristics_created=3,
            heuristics_updated=1,
            patterns_analyzed=10,
            patterns_rejected=6,
            rejected_reasons={"low_confidence": 4, "insufficient": 2},
        )

        d = result.to_dict()

        assert d["heuristics_created"] == 3
        assert d["patterns_analyzed"] == 10
        assert d["rejected_reasons"]["low_confidence"] == 4


class TestConvenienceFunction:
    """Tests for extract_heuristics_from_outcome function."""

    def test_matches_existing_heuristic(self):
        """Test matching an outcome to existing heuristic."""
        outcome = Outcome(
            id="o1",
            agent="helena",
            project_id="test",
            task_type="form_testing",
            task_description="Test",
            success=True,
            strategy_used="test first",
            timestamp=datetime.now(timezone.utc),
        )

        heuristics = [
            Heuristic(
                id="h1",
                agent="helena",
                project_id="test",
                condition="task type: form_testing",
                strategy="test first",
                confidence=0.8,
                occurrence_count=5,
                success_count=4,
                last_validated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            ),
        ]

        result = extract_heuristics_from_outcome(outcome, heuristics)

        assert result is not None
        assert result["heuristic_id"] == "h1"
        assert result["action"] == "validate"

    def test_no_match(self):
        """Test when no heuristic matches."""
        outcome = Outcome(
            id="o1",
            agent="helena",
            project_id="test",
            task_type="api_testing",
            task_description="Test",
            success=True,
            strategy_used="test first",
            timestamp=datetime.now(timezone.utc),
        )

        heuristics = [
            Heuristic(
                id="h1",
                agent="victor",
                project_id="test",  # Different agent
                condition="task type: api_testing",
                strategy="test first",
                confidence=0.8,
                occurrence_count=5,
                success_count=4,
                last_validated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            ),
        ]

        result = extract_heuristics_from_outcome(outcome, heuristics)

        assert result is None
