"""Tests for the RetrievalFeedbackTracker -- full feedback loop."""

import json
from unittest.mock import MagicMock

from alma.rag.feedback import FEEDBACK_DOMAIN, FEEDBACK_SOURCE, RetrievalFeedbackTracker
from alma.storage.base import StorageBackend
from alma.types import DomainKnowledge


def _make_mock_storage():
    storage = MagicMock(spec=StorageBackend)
    storage.get_domain_knowledge.return_value = []
    return storage


class TestRecordRetrieval:
    def test_basic_record(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        record = tracker.record_retrieval(
            query="how to deploy auth",
            agent="test-agent",
            memory_ids=["h1", "o1", "dk1"],
        )

        assert record.id.startswith("rr_")
        assert record.query == "how to deploy auth"
        assert len(record.memory_ids) == 3
        storage.save_domain_knowledge.assert_called_once()

        # Verify persisted DK
        saved_dk = storage.save_domain_knowledge.call_args[0][0]
        assert saved_dk.domain == FEEDBACK_DOMAIN
        data = json.loads(saved_dk.fact)
        assert data["type"] == "retrieval_record"

    def test_record_with_scores(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        record = tracker.record_retrieval(
            query="test",
            agent="agent",
            memory_ids=["h1"],
            scores={"h1": 0.95},
        )

        assert record.scores["h1"] == 0.95


class TestRecordFeedback:
    def test_basic_feedback(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        # First record a retrieval
        record = tracker.record_retrieval(
            query="test",
            agent="agent",
            memory_ids=["h1", "o1"],
        )

        # Then record feedback
        feedback = tracker.record_feedback(
            retrieval_record_id=record.id,
            outcome_id="out_001",
            success=True,
        )

        assert feedback is not None
        assert feedback.success is True
        assert feedback.retrieval_record_id == record.id

        # Should have saved 2 DK records (1 retrieval + 1 feedback)
        assert storage.save_domain_knowledge.call_count == 2

    def test_feedback_for_unknown_record(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        feedback = tracker.record_feedback(
            retrieval_record_id="nonexistent",
            outcome_id="out_001",
            success=True,
        )

        assert feedback is None

    def test_feedback_persisted_correctly(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        record = tracker.record_retrieval(
            query="test",
            agent="agent",
            memory_ids=["h1"],
        )
        tracker.record_feedback(
            retrieval_record_id=record.id,
            outcome_id="out_001",
            success=False,
        )

        # Check the feedback DK
        last_call = storage.save_domain_knowledge.call_args_list[-1]
        saved_dk = last_call[0][0]
        data = json.loads(saved_dk.fact)
        assert data["type"] == "retrieval_feedback"
        assert data["success"] is False
        assert data["memory_ids"] == ["h1"]


class TestGetEffectiveness:
    def _make_feedback_dk(self, memory_ids, success):
        """Create a DomainKnowledge record simulating stored feedback."""
        return DomainKnowledge(
            id=f"rf_{id(memory_ids)}",
            agent="agent",
            project_id="proj-1",
            domain=FEEDBACK_DOMAIN,
            fact=json.dumps({
                "type": "retrieval_feedback",
                "retrieval_record_id": "rr_001",
                "outcome_id": "out_001",
                "success": success,
                "memory_ids": memory_ids,
                "helpful_memory_ids": [],
                "unhelpful_memory_ids": [],
            }),
            source=FEEDBACK_SOURCE,
        )

    def test_empty_feedback(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        effectiveness = tracker.get_memory_effectiveness()
        assert effectiveness == {}

    def test_single_memory_all_success(self):
        storage = _make_mock_storage()
        storage.get_domain_knowledge.return_value = [
            self._make_feedback_dk(["h1"], True),
            self._make_feedback_dk(["h1"], True),
            self._make_feedback_dk(["h1"], True),
        ]

        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")
        effectiveness = tracker.get_memory_effectiveness()

        assert "h1" in effectiveness
        eff = effectiveness["h1"]
        assert eff.times_retrieved == 3
        assert eff.times_in_success == 3
        assert eff.success_correlation == 1.0
        assert eff.weight_adjustment > 1.0  # Should be boosted

    def test_single_memory_mixed(self):
        storage = _make_mock_storage()
        storage.get_domain_knowledge.return_value = [
            self._make_feedback_dk(["h1"], True),
            self._make_feedback_dk(["h1"], False),
            self._make_feedback_dk(["h1"], True),
            self._make_feedback_dk(["h1"], False),
        ]

        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")
        effectiveness = tracker.get_memory_effectiveness()

        eff = effectiveness["h1"]
        assert eff.times_retrieved == 4
        assert abs(eff.success_correlation - 0.5) < 0.01
        assert abs(eff.weight_adjustment - 1.0) < 0.01  # Neutral

    def test_multiple_memories(self):
        storage = _make_mock_storage()
        storage.get_domain_knowledge.return_value = [
            self._make_feedback_dk(["h1", "h2"], True),
            self._make_feedback_dk(["h1", "h2"], True),
            self._make_feedback_dk(["h1", "h2"], True),
        ]

        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")
        effectiveness = tracker.get_memory_effectiveness()

        assert "h1" in effectiveness
        assert "h2" in effectiveness
        assert effectiveness["h1"].times_retrieved == 3
        assert effectiveness["h2"].times_retrieved == 3


class TestComputeWeightAdjustments:
    def test_returns_non_neutral_only(self):
        storage = _make_mock_storage()
        feedback_records = []
        for i in range(4):
            feedback_records.append(
                DomainKnowledge(
                    id=f"rf_{i}",
                    agent="agent",
                    project_id="proj-1",
                    domain=FEEDBACK_DOMAIN,
                    fact=json.dumps({
                        "type": "retrieval_feedback",
                        "retrieval_record_id": "rr",
                        "outcome_id": "out",
                        "success": i % 2 == 0,
                        "memory_ids": ["h_neutral"],
                        "helpful_memory_ids": [],
                        "unhelpful_memory_ids": [],
                    }),
                    source=FEEDBACK_SOURCE,
                )
            )
        storage.get_domain_knowledge.return_value = feedback_records

        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")
        adjustments = tracker.compute_weight_adjustments()

        # h_neutral has 50% success rate -- should not appear
        assert "h_neutral" not in adjustments


class TestGetLastRetrievalRecord:
    def test_returns_most_recent(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        tracker.record_retrieval(query="first", agent="agent-a", memory_ids=["h1"])
        tracker.record_retrieval(query="second", agent="agent-a", memory_ids=["h2"])
        tracker.record_retrieval(query="other", agent="agent-b", memory_ids=["h3"])

        last = tracker.get_last_retrieval_record("agent-a")
        assert last is not None
        assert last.query == "second"

    def test_returns_none_if_no_records(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        assert tracker.get_last_retrieval_record("agent-a") is None


class TestFullFeedbackLoop:
    """Integration test: retrieve -> learn -> compute adjustments."""

    def test_full_loop(self):
        storage = _make_mock_storage()
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        # Step 1: Record retrievals
        r1 = tracker.record_retrieval(
            query="deploy auth",
            agent="agent",
            memory_ids=["h1", "h2"],
        )
        r2 = tracker.record_retrieval(
            query="test login",
            agent="agent",
            memory_ids=["h1", "h3"],
        )

        # Step 2: Record feedback (h1 always present, succeeds both times)
        tracker.record_feedback(r1.id, "out_1", success=True)
        tracker.record_feedback(r2.id, "out_2", success=True)

        # Step 3: Set up storage to return the feedback we just created
        all_saved = [
            call[0][0]
            for call in storage.save_domain_knowledge.call_args_list
        ]
        feedback_dks = [dk for dk in all_saved if dk.source == FEEDBACK_SOURCE]
        storage.get_domain_knowledge.return_value = feedback_dks

        # Step 4: Compute effectiveness
        effectiveness = tracker.get_memory_effectiveness()

        assert "h1" in effectiveness
        assert effectiveness["h1"].times_retrieved == 2
        assert effectiveness["h1"].times_in_success == 2
