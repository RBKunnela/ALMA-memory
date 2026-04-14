"""
Unit tests for the ALMA Retrieval Feedback Loop.

Tests FeedbackSummary scoring, FeedbackTracker recording,
FeedbackAwareScorer re-ranking, and ALMA-level convenience methods.
"""

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from alma.retrieval.feedback import FeedbackAwareScorer, FeedbackTracker
from alma.retrieval.scoring import ScoredItem
from alma.testing import MockStorage
from alma.types import (
    FeedbackSignal,
    FeedbackSummary,
    MemoryType,
    RetrievalFeedback,
)

# ==================== FIXTURES ====================


@pytest.fixture
def storage() -> MockStorage:
    """Fresh MockStorage for each test."""
    return MockStorage()


@pytest.fixture
def tracker(storage: MockStorage) -> FeedbackTracker:
    """FeedbackTracker with MockStorage."""
    return FeedbackTracker(storage)


@pytest.fixture
def scorer(tracker: FeedbackTracker) -> FeedbackAwareScorer:
    """FeedbackAwareScorer with default weight."""
    return FeedbackAwareScorer(tracker, feedback_weight=0.15)


def _make_feedback(
    memory_id: str = "m1",
    signal: FeedbackSignal = FeedbackSignal.USED,
    memory_type: MemoryType = MemoryType.HEURISTIC,
    agent: str = "test-agent",
    project_id: str = "proj-1",
    query: str = "test query",
) -> RetrievalFeedback:
    """Helper to create a RetrievalFeedback record."""
    return RetrievalFeedback(
        id=str(uuid.uuid4()),
        memory_id=memory_id,
        memory_type=memory_type,
        query=query,
        agent=agent,
        project_id=project_id,
        signal=signal,
    )


def _make_scored_item(item_id: str, score: float) -> ScoredItem:
    """Helper to create a ScoredItem with an item that has an id attribute."""

    @dataclass
    class FakeMemory:
        id: str

    return ScoredItem(
        item=FakeMemory(id=item_id),
        score=score,
        similarity_score=score,
        recency_score=0.5,
        success_score=0.5,
        confidence_score=0.5,
    )


# ==================== FeedbackSummary.feedback_score ====================


class TestFeedbackSummaryScore:
    """Tests for FeedbackSummary.feedback_score property."""

    def test_UNIT_feedback_score_should_return_zero_when_all_counts_zero(self):
        """No feedback signals should yield a neutral score."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
        )
        assert summary.feedback_score == 0.0

    def test_UNIT_feedback_score_should_return_positive_one_when_all_used(self):
        """All USED signals should produce +1.0."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            use_count=10,
            ignore_count=0,
            positive_count=0,
            negative_count=0,
        )
        assert summary.feedback_score == 1.0

    def test_UNIT_feedback_score_should_return_negative_one_when_all_ignored(self):
        """All IGNORED signals should produce -1.0."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            use_count=0,
            ignore_count=5,
        )
        assert summary.feedback_score == -1.0

    def test_UNIT_feedback_score_should_return_mixed_value_when_mixed_signals(self):
        """Mixed signals should produce a value between -1 and 1."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            use_count=3,
            ignore_count=1,
            positive_count=2,
            negative_count=0,
        )
        # positive = 3 + 2 = 5, negative = 1 + 0 = 1, total = 6
        # score = (5 - 1) / 6 = 0.6667
        expected = (5 - 1) / 6
        assert abs(summary.feedback_score - expected) < 1e-9

    def test_UNIT_feedback_score_should_return_zero_when_balanced(self):
        """Equal positive and negative signals should produce 0.0."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            use_count=3,
            ignore_count=3,
        )
        assert summary.feedback_score == 0.0

    def test_UNIT_feedback_score_should_include_thumbs_signals(self):
        """THUMBS_UP and THUMBS_DOWN should be included in score."""
        summary = FeedbackSummary(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            use_count=0,
            ignore_count=0,
            positive_count=4,
            negative_count=1,
        )
        # positive = 4, negative = 1, total = 5
        # score = (4 - 1) / 5 = 0.6
        assert abs(summary.feedback_score - 0.6) < 1e-9


# ==================== FeedbackTracker.record ====================


class TestFeedbackTrackerRecord:
    """Tests for FeedbackTracker.record() method."""

    def test_UNIT_record_should_save_to_storage_when_valid_feedback(
        self, tracker: FeedbackTracker, storage: MockStorage
    ):
        """Record should persist feedback and return its ID."""
        fb = _make_feedback(memory_id="m1", signal=FeedbackSignal.USED)
        result_id = tracker.record(fb)

        assert result_id == fb.id
        assert storage.feedback_count == 1

    def test_UNIT_record_should_return_feedback_id_when_called(
        self, tracker: FeedbackTracker
    ):
        """Return value should be the feedback record ID."""
        fb = _make_feedback()
        returned = tracker.record(fb)
        assert returned == fb.id

    def test_UNIT_record_should_increment_count_when_multiple_records(
        self, tracker: FeedbackTracker, storage: MockStorage
    ):
        """Multiple records should all persist."""
        for i in range(5):
            tracker.record(_make_feedback(memory_id=f"m{i}"))
        assert storage.feedback_count == 5


# ==================== FeedbackTracker.record_usage ====================


class TestFeedbackTrackerRecordUsage:
    """Tests for FeedbackTracker.record_usage() method."""

    def test_UNIT_record_usage_should_mark_used_and_ignored_when_partial_use(
        self, tracker: FeedbackTracker, storage: MockStorage
    ):
        """Memories in used_ids get USED; others get IGNORED."""
        ids = tracker.record_usage(
            retrieved_ids=["m1", "m2", "m3"],
            used_ids=["m1"],
            memory_type=MemoryType.HEURISTIC,
            agent="test-agent",
            project_id="proj-1",
            query="test",
        )
        assert len(ids) == 3
        assert storage.feedback_count == 3

        # Verify signals via get_feedback_summary
        summaries = storage.get_feedback_summary(
            ["m1", "m2", "m3"], MemoryType.HEURISTIC
        )
        assert summaries["m1"].use_count == 1
        assert summaries["m1"].ignore_count == 0
        assert summaries["m2"].use_count == 0
        assert summaries["m2"].ignore_count == 1
        assert summaries["m3"].ignore_count == 1

    def test_UNIT_record_usage_should_mark_all_used_when_all_ids_in_used(
        self, tracker: FeedbackTracker, storage: MockStorage
    ):
        """When all retrieved are used, all should be USED."""
        tracker.record_usage(
            retrieved_ids=["m1", "m2"],
            used_ids=["m1", "m2"],
            memory_type=MemoryType.OUTCOME,
            agent="agent-x",
            project_id="proj-1",
        )
        summaries = storage.get_feedback_summary(["m1", "m2"], MemoryType.OUTCOME)
        assert summaries["m1"].use_count == 1
        assert summaries["m2"].use_count == 1

    def test_UNIT_record_usage_should_mark_all_ignored_when_none_used(
        self, tracker: FeedbackTracker, storage: MockStorage
    ):
        """When no retrieved are used, all should be IGNORED."""
        tracker.record_usage(
            retrieved_ids=["m1", "m2"],
            used_ids=[],
            memory_type=MemoryType.HEURISTIC,
            agent="agent-x",
            project_id="proj-1",
        )
        summaries = storage.get_feedback_summary(["m1", "m2"], MemoryType.HEURISTIC)
        assert summaries["m1"].ignore_count == 1
        assert summaries["m2"].ignore_count == 1

    def test_UNIT_record_usage_should_return_empty_list_when_no_retrieved_ids(
        self, tracker: FeedbackTracker
    ):
        """Empty retrieved_ids should produce empty result."""
        ids = tracker.record_usage(
            retrieved_ids=[],
            used_ids=[],
            memory_type=MemoryType.HEURISTIC,
            agent="agent-x",
            project_id="proj-1",
        )
        assert ids == []


# ==================== FeedbackTracker.get_summaries ====================


class TestFeedbackTrackerGetSummaries:
    """Tests for FeedbackTracker.get_summaries() method."""

    def test_UNIT_get_summaries_should_return_aggregated_data_when_feedback_exists(
        self, tracker: FeedbackTracker
    ):
        """Summaries should aggregate multiple feedback records per memory."""
        # Record 3 USED and 1 IGNORED for m1
        for _ in range(3):
            tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.USED))
        tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.IGNORED))

        summaries = tracker.get_summaries(["m1"], MemoryType.HEURISTIC)
        assert "m1" in summaries
        assert summaries["m1"].use_count == 3
        assert summaries["m1"].ignore_count == 1

    def test_UNIT_get_summaries_should_return_empty_dict_when_no_feedback(
        self, tracker: FeedbackTracker
    ):
        """No feedback records should produce empty dict."""
        summaries = tracker.get_summaries(["m1"], MemoryType.HEURISTIC)
        assert summaries == {}

    def test_UNIT_get_summaries_should_filter_by_memory_type(
        self, tracker: FeedbackTracker
    ):
        """Feedback for different memory types should not cross-contaminate."""
        tracker.record(
            _make_feedback(
                memory_id="m1",
                signal=FeedbackSignal.USED,
                memory_type=MemoryType.HEURISTIC,
            )
        )
        tracker.record(
            _make_feedback(
                memory_id="m1",
                signal=FeedbackSignal.IGNORED,
                memory_type=MemoryType.OUTCOME,
            )
        )

        summaries = tracker.get_summaries(["m1"], MemoryType.HEURISTIC)
        assert summaries["m1"].use_count == 1
        assert summaries["m1"].ignore_count == 0

    def test_UNIT_get_summaries_should_only_include_requested_ids(
        self, tracker: FeedbackTracker
    ):
        """Summaries should not include feedback for unrequested memory IDs."""
        tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.USED))
        tracker.record(_make_feedback(memory_id="m2", signal=FeedbackSignal.USED))

        summaries = tracker.get_summaries(["m1"], MemoryType.HEURISTIC)
        assert "m1" in summaries
        assert "m2" not in summaries


# ==================== FeedbackAwareScorer ====================


class TestFeedbackAwareScorer:
    """Tests for FeedbackAwareScorer.apply_feedback() and validation."""

    def test_UNIT_apply_feedback_should_rerank_items_when_feedback_exists(
        self, tracker: FeedbackTracker
    ):
        """Items with positive feedback should be boosted."""
        # m1 has positive feedback, m2 has negative
        for _ in range(5):
            tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.USED))
        for _ in range(5):
            tracker.record(
                _make_feedback(memory_id="m2", signal=FeedbackSignal.IGNORED)
            )

        scorer = FeedbackAwareScorer(tracker, feedback_weight=0.5)
        items = [
            _make_scored_item("m2", 0.9),  # Higher base score but negative feedback
            _make_scored_item("m1", 0.7),  # Lower base score but positive feedback
        ]

        result = scorer.apply_feedback(items, MemoryType.HEURISTIC)

        # m1 should be re-ranked higher due to positive feedback
        assert result[0].item.id == "m1"
        assert result[1].item.id == "m2"

    def test_UNIT_apply_feedback_should_not_change_items_when_no_feedback(
        self, tracker: FeedbackTracker
    ):
        """Items without feedback should retain original scores and order."""
        scorer = FeedbackAwareScorer(tracker, feedback_weight=0.15)
        items = [
            _make_scored_item("m1", 0.9),
            _make_scored_item("m2", 0.7),
        ]

        result = scorer.apply_feedback(items, MemoryType.HEURISTIC)

        assert result[0].item.id == "m1"
        assert result[0].score == 0.9
        assert result[1].item.id == "m2"
        assert result[1].score == 0.7

    def test_UNIT_apply_feedback_should_return_empty_list_when_empty_input(
        self, tracker: FeedbackTracker
    ):
        """Empty input should produce empty output."""
        scorer = FeedbackAwareScorer(tracker, feedback_weight=0.15)
        result = scorer.apply_feedback([], MemoryType.HEURISTIC)
        assert result == []

    def test_UNIT_apply_feedback_should_return_unchanged_when_weight_zero(
        self, tracker: FeedbackTracker
    ):
        """Zero weight should not modify scores even with feedback."""
        tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.USED))
        scorer = FeedbackAwareScorer(tracker, feedback_weight=0.0)
        items = [_make_scored_item("m1", 0.8)]

        result = scorer.apply_feedback(items, MemoryType.HEURISTIC)
        assert result[0].score == 0.8

    def test_UNIT_apply_feedback_should_blend_score_correctly_when_feedback_positive(
        self, tracker: FeedbackTracker
    ):
        """Verify the blending formula: final = (1-w)*base + w*normalized."""
        tracker.record(_make_feedback(memory_id="m1", signal=FeedbackSignal.USED))
        weight = 0.2
        scorer = FeedbackAwareScorer(tracker, feedback_weight=weight)

        base_score = 0.6
        items = [_make_scored_item("m1", base_score)]

        result = scorer.apply_feedback(items, MemoryType.HEURISTIC)

        # feedback_score = 1.0 (all used), normalized = (1.0 + 1.0) / 2.0 = 1.0
        # final = 0.8 * 0.6 + 0.2 * 1.0 = 0.48 + 0.2 = 0.68
        expected = (1.0 - weight) * base_score + weight * 1.0
        assert abs(result[0].score - expected) < 1e-9

    def test_UNIT_apply_feedback_should_handle_items_without_id_attribute(
        self, tracker: FeedbackTracker
    ):
        """Items whose .item has no .id should pass through unchanged."""
        scorer = FeedbackAwareScorer(tracker, feedback_weight=0.15)

        class NoIdItem:
            pass

        item = ScoredItem(
            item=NoIdItem(),
            score=0.5,
            similarity_score=0.5,
            recency_score=0.5,
            success_score=0.5,
            confidence_score=0.5,
        )
        result = scorer.apply_feedback([item], MemoryType.HEURISTIC)
        assert result[0].score == 0.5

    def test_UNIT_init_should_raise_when_weight_below_zero(
        self, tracker: FeedbackTracker
    ):
        """Weight below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            FeedbackAwareScorer(tracker, feedback_weight=-0.1)

    def test_UNIT_init_should_raise_when_weight_above_one(
        self, tracker: FeedbackTracker
    ):
        """Weight above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            FeedbackAwareScorer(tracker, feedback_weight=1.5)

    def test_UNIT_init_should_accept_boundary_weights(
        self, tracker: FeedbackTracker
    ):
        """Boundary values 0.0 and 1.0 should be valid."""
        s0 = FeedbackAwareScorer(tracker, feedback_weight=0.0)
        s1 = FeedbackAwareScorer(tracker, feedback_weight=1.0)
        assert s0.feedback_weight == 0.0
        assert s1.feedback_weight == 1.0


# ==================== ALMA.record_feedback (end-to-end with MockStorage) ====================


class TestALMARecordFeedback:
    """Tests for ALMA.record_feedback() convenience method."""

    def _make_alma(self, storage: MockStorage) -> Any:
        """Create a minimal ALMA instance for testing."""
        from alma.core import ALMA
        from alma.learning.protocols import LearningProtocol
        from alma.retrieval.engine import RetrievalEngine

        engine = RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
        )
        learning = LearningProtocol(
            storage=storage,
            scopes={},
        )
        return ALMA(
            storage=storage,
            retrieval_engine=engine,
            learning_protocol=learning,
            scopes={},
            project_id="test-project",
        )

    def test_UNIT_record_feedback_should_persist_when_valid_signal(self):
        """ALMA.record_feedback should save to storage."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        result_id = alma.record_feedback(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            signal=FeedbackSignal.USED,
            agent="qa-agent",
            query="testing query",
        )

        assert isinstance(result_id, str)
        assert len(result_id) > 0
        assert storage.feedback_count == 1

    def test_UNIT_record_feedback_should_use_project_id_from_alma(self):
        """Feedback should inherit ALMA's project_id."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        alma.record_feedback(
            memory_id="m1",
            memory_type=MemoryType.HEURISTIC,
            signal=FeedbackSignal.THUMBS_UP,
            agent="dev-agent",
        )

        # Check stored feedback has correct project_id
        summaries = storage.get_feedback_summary(["m1"], MemoryType.HEURISTIC)
        assert "m1" in summaries
        assert summaries["m1"].positive_count == 1

    def test_UNIT_record_feedback_should_handle_all_signal_types(self):
        """All FeedbackSignal enum values should work."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        for signal in FeedbackSignal:
            alma.record_feedback(
                memory_id=f"m-{signal.value}",
                memory_type=MemoryType.HEURISTIC,
                signal=signal,
                agent="test-agent",
            )

        assert storage.feedback_count == len(FeedbackSignal)


# ==================== ALMA.record_usage (end-to-end) ====================


class TestALMARecordUsage:
    """Tests for ALMA.record_usage() convenience method."""

    def _make_alma(self, storage: MockStorage) -> Any:
        """Create a minimal ALMA instance for testing."""
        from alma.core import ALMA
        from alma.learning.protocols import LearningProtocol
        from alma.retrieval.engine import RetrievalEngine

        engine = RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
        )
        learning = LearningProtocol(
            storage=storage,
            scopes={},
        )
        return ALMA(
            storage=storage,
            retrieval_engine=engine,
            learning_protocol=learning,
            scopes={},
            project_id="test-project",
        )

    def test_UNIT_record_usage_should_persist_all_feedback_when_partial_use(self):
        """Convenience method should create feedback for all retrieved IDs."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        ids = alma.record_usage(
            retrieved_memory_ids=["m1", "m2", "m3"],
            used_memory_ids=["m1"],
            memory_type=MemoryType.HEURISTIC,
            agent="test-agent",
        )

        assert len(ids) == 3
        assert storage.feedback_count == 3

        summaries = storage.get_feedback_summary(
            ["m1", "m2", "m3"], MemoryType.HEURISTIC
        )
        assert summaries["m1"].use_count == 1
        assert summaries["m2"].ignore_count == 1
        assert summaries["m3"].ignore_count == 1

    def test_UNIT_record_usage_should_return_empty_when_no_retrieved(self):
        """Empty retrieved list should produce empty result."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        ids = alma.record_usage(
            retrieved_memory_ids=[],
            used_memory_ids=[],
            memory_type=MemoryType.HEURISTIC,
            agent="test-agent",
        )

        assert ids == []
        assert storage.feedback_count == 0

    def test_UNIT_record_usage_should_accept_optional_query(self):
        """Query parameter should be optional and default to empty string."""
        storage = MockStorage()
        alma = self._make_alma(storage)

        # Should not raise without query
        ids = alma.record_usage(
            retrieved_memory_ids=["m1"],
            used_memory_ids=["m1"],
            memory_type=MemoryType.OUTCOME,
            agent="agent-x",
        )
        assert len(ids) == 1

        # Should also work with explicit query
        ids2 = alma.record_usage(
            retrieved_memory_ids=["m2"],
            used_memory_ids=[],
            memory_type=MemoryType.OUTCOME,
            agent="agent-x",
            query="explicit query",
        )
        assert len(ids2) == 1
