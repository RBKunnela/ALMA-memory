"""Tests for retrieval feedback data structures."""

from alma.rag.feedback_types import (
    RetrievalEffectiveness,
    RetrievalFeedback,
    RetrievalRecord,
)


class TestRetrievalRecord:
    def test_basic_construction(self):
        record = RetrievalRecord(
            id="rr_001",
            query="how to deploy",
            agent="test-agent",
            project_id="proj-1",
            memory_ids=["h1", "o1"],
        )
        assert record.id == "rr_001"
        assert len(record.memory_ids) == 2
        assert record.mode == "default"

    def test_with_scores(self):
        record = RetrievalRecord(
            id="rr_002",
            query="test",
            agent="agent",
            project_id="proj-1",
            memory_ids=["h1"],
            scores={"h1": 0.95},
        )
        assert record.scores["h1"] == 0.95


class TestRetrievalFeedback:
    def test_basic_construction(self):
        fb = RetrievalFeedback(
            id="rf_001",
            retrieval_record_id="rr_001",
            outcome_id="out_001",
            success=True,
        )
        assert fb.success is True
        assert fb.helpful_memory_ids == []

    def test_with_explicit_feedback(self):
        fb = RetrievalFeedback(
            id="rf_002",
            retrieval_record_id="rr_001",
            outcome_id="out_002",
            success=False,
            helpful_memory_ids=["h1"],
            unhelpful_memory_ids=["o1"],
        )
        assert len(fb.helpful_memory_ids) == 1
        assert len(fb.unhelpful_memory_ids) == 1


class TestRetrievalEffectiveness:
    def test_compute_correlation_no_data(self):
        eff = RetrievalEffectiveness(memory_id="h1")
        eff.compute_correlation()
        assert eff.success_correlation == 0.5  # Neutral default

    def test_compute_correlation_all_success(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=5,
            times_in_success=5,
            times_in_failure=0,
        )
        eff.compute_correlation()
        assert eff.success_correlation == 1.0

    def test_compute_correlation_all_failure(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=4,
            times_in_success=0,
            times_in_failure=4,
        )
        eff.compute_correlation()
        assert eff.success_correlation == 0.0

    def test_compute_correlation_mixed(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=10,
            times_in_success=7,
            times_in_failure=3,
        )
        eff.compute_correlation()
        assert abs(eff.success_correlation - 0.7) < 0.01

    def test_weight_adjustment_not_enough_data(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=2,
            times_in_success=2,
        )
        eff.compute_weight_adjustment()
        assert eff.weight_adjustment == 1.0  # Neutral, not enough data

    def test_weight_adjustment_high_success(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=10,
            times_in_success=9,
            times_in_failure=1,
        )
        eff.compute_weight_adjustment()
        assert eff.weight_adjustment > 1.0  # Should boost

    def test_weight_adjustment_high_failure(self):
        eff = RetrievalEffectiveness(
            memory_id="h1",
            times_retrieved=10,
            times_in_success=1,
            times_in_failure=9,
        )
        eff.compute_weight_adjustment()
        assert eff.weight_adjustment < 1.0  # Should demote
        assert eff.weight_adjustment >= 0.5  # But not below floor
