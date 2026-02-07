"""
Unit tests for ALMA Trust-Integrated Scoring.

Tests TrustLevel, TrustWeights, AgentTrustProfile, AgentTrustContext,
TrustAwareScorer, and TrustScoredItem.
"""

from datetime import datetime, timedelta, timezone

import pytest

from alma.retrieval.scoring import ScoringWeights
from alma.retrieval.trust_scoring import (
    AgentTrustContext,
    AgentTrustProfile,
    TrustAwareScorer,
    TrustLevel,
    TrustScoredItem,
    TrustWeights,
)
from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_outcome,
)
from alma.types import MemorySlice


class TestTrustLevel:
    """Tests for TrustLevel constants and label()."""

    def test_constants_ascending(self):
        """Trust level constants should increase monotonically."""
        levels = [
            TrustLevel.UNTRUSTED,
            TrustLevel.MINIMAL,
            TrustLevel.LOW,
            TrustLevel.MODERATE,
            TrustLevel.GOOD,
            TrustLevel.HIGH,
            TrustLevel.FULL,
        ]
        for i in range(len(levels) - 1):
            assert levels[i] < levels[i + 1]

    def test_label_full(self):
        assert TrustLevel.label(1.0) == "FULL"

    def test_label_high(self):
        assert TrustLevel.label(0.9) == "HIGH"

    def test_label_good(self):
        assert TrustLevel.label(0.75) == "GOOD"

    def test_label_moderate(self):
        assert TrustLevel.label(0.5) == "MODERATE"

    def test_label_low(self):
        assert TrustLevel.label(0.4) == "LOW"

    def test_label_minimal(self):
        assert TrustLevel.label(0.2) == "MINIMAL"

    def test_label_untrusted(self):
        assert TrustLevel.label(0.0) == "UNTRUSTED"
        assert TrustLevel.label(0.1) == "UNTRUSTED"

    def test_label_boundary_values(self):
        """Boundary values should map to the correct tier."""
        assert TrustLevel.label(0.85) == "HIGH"
        assert TrustLevel.label(0.7) == "GOOD"
        assert TrustLevel.label(0.19) == "UNTRUSTED"


class TestTrustWeights:
    """Tests for TrustWeights dataclass."""

    def test_default_weights_sum_to_one(self):
        """Default weights including trust should sum to 1.0."""
        tw = TrustWeights()
        total = tw.similarity + tw.recency + tw.success_rate + tw.confidence + tw.trust
        assert abs(total - 1.0) < 0.01

    def test_normalization_when_sum_not_one(self):
        """Weights that don't sum to 1.0 should be normalized."""
        tw = TrustWeights(
            similarity=1.0,
            recency=1.0,
            success_rate=1.0,
            confidence=1.0,
            trust=1.0,
        )
        total = tw.similarity + tw.recency + tw.success_rate + tw.confidence + tw.trust
        assert abs(total - 1.0) < 0.01
        assert abs(tw.similarity - 0.2) < 0.01

    def test_no_normalization_when_already_one(self):
        """Weights already summing to 1.0 should stay unchanged."""
        tw = TrustWeights(
            similarity=0.3,
            recency=0.2,
            success_rate=0.2,
            confidence=0.1,
            trust=0.2,
        )
        assert abs(tw.similarity - 0.3) < 0.01
        assert abs(tw.trust - 0.2) < 0.01

    def test_extends_scoring_weights(self):
        """TrustWeights should be a subclass of ScoringWeights."""
        tw = TrustWeights()
        assert isinstance(tw, ScoringWeights)


class TestAgentTrustProfile:
    """Tests for AgentTrustProfile."""

    def test_default_trust_is_moderate(self):
        profile = AgentTrustProfile(agent_id="test-agent")
        assert profile.current_trust == TrustLevel.MODERATE

    def test_calculate_trust_no_sessions(self):
        """With no sessions, trust should stay at MODERATE."""
        profile = AgentTrustProfile(agent_id="test-agent", sessions_completed=0)
        trust = profile.calculate_trust()
        assert trust == TrustLevel.MODERATE

    def test_calculate_trust_perfect_performance(self):
        """Perfect performance should yield high trust."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            sessions_completed=10,
            total_actions=100,
            total_violations=0,
            consecutive_clean_sessions=10,
            last_session=datetime.now(timezone.utc),
        )
        trust = profile.calculate_trust()
        assert trust > TrustLevel.GOOD

    def test_calculate_trust_many_violations(self):
        """Many violations should lower trust significantly."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            sessions_completed=10,
            total_actions=100,
            total_violations=50,
            consecutive_clean_sessions=0,
            last_session=datetime.now(timezone.utc),
        )
        trust = profile.calculate_trust()
        assert trust < TrustLevel.GOOD

    def test_streak_bonus(self):
        """Consecutive clean sessions should add a bonus."""
        base = AgentTrustProfile(
            agent_id="test-agent",
            sessions_completed=10,
            total_actions=50,
            total_violations=0,
            consecutive_clean_sessions=0,
            last_session=datetime.now(timezone.utc),
        )
        streaky = AgentTrustProfile(
            agent_id="test-agent",
            sessions_completed=10,
            total_actions=50,
            total_violations=0,
            consecutive_clean_sessions=5,
            last_session=datetime.now(timezone.utc),
        )
        base_trust = base.calculate_trust()
        streaky_trust = streaky.calculate_trust()
        assert streaky_trust >= base_trust

    def test_streak_bonus_caps_at_point_one(self):
        """Streak bonus should cap at 0.1."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            sessions_completed=10,
            total_actions=50,
            total_violations=0,
            consecutive_clean_sessions=100,
            last_session=datetime.now(timezone.utc),
        )
        trust = profile.calculate_trust()
        assert trust <= 1.0

    def test_decay_no_last_session(self):
        """No last_session should give decay factor of 1.0."""
        profile = AgentTrustProfile(agent_id="test-agent", last_session=None)
        decay = profile._calculate_decay()
        assert decay == 1.0

    def test_decay_recent_session(self):
        """Recent session should give decay near 1.0."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            last_session=datetime.now(timezone.utc),
        )
        decay = profile._calculate_decay()
        assert decay > 0.99

    def test_decay_old_session(self):
        """Old session should reduce decay but not below 0.5."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            last_session=datetime.now(timezone.utc) - timedelta(days=365),
            trust_half_life_days=30,
        )
        decay = profile._calculate_decay()
        assert decay == 0.5  # Minimum floor

    def test_decay_half_life(self):
        """After half_life days, decay should be around 0.5."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            last_session=datetime.now(timezone.utc) - timedelta(days=30),
            trust_half_life_days=30,
        )
        decay = profile._calculate_decay()
        assert 0.49 < decay < 0.52

    def test_decay_naive_datetime_handled(self):
        """Naive datetimes (no tzinfo) should be handled gracefully."""
        profile = AgentTrustProfile(
            agent_id="test-agent",
            last_session=datetime.now() - timedelta(days=1),
        )
        decay = profile._calculate_decay()
        assert 0.5 <= decay <= 1.0

    def test_to_dict_roundtrip(self):
        """to_dict/from_dict should preserve all fields."""
        now = datetime.now(timezone.utc)
        original = AgentTrustProfile(
            agent_id="helena",
            current_trust=0.85,
            sessions_completed=20,
            total_actions=200,
            total_violations=3,
            consecutive_clean_sessions=7,
            last_session=now,
            trust_half_life_days=45,
            behavior_trust={"verification_before_claim": 0.9, "loud_failure": 0.8},
        )

        data = original.to_dict()
        restored = AgentTrustProfile.from_dict(data)

        assert restored.agent_id == original.agent_id
        assert restored.current_trust == original.current_trust
        assert restored.sessions_completed == original.sessions_completed
        assert restored.total_actions == original.total_actions
        assert restored.total_violations == original.total_violations
        assert (
            restored.consecutive_clean_sessions == original.consecutive_clean_sessions
        )
        assert restored.trust_half_life_days == original.trust_half_life_days
        assert restored.behavior_trust == original.behavior_trust

    def test_to_dict_none_last_session(self):
        """to_dict with None last_session should serialize as None."""
        profile = AgentTrustProfile(agent_id="test", last_session=None)
        data = profile.to_dict()
        assert data["last_session"] is None

    def test_from_dict_missing_fields_use_defaults(self):
        """from_dict should use defaults for missing optional fields."""
        data = {"agent_id": "minimal"}
        profile = AgentTrustProfile.from_dict(data)
        assert profile.agent_id == "minimal"
        assert profile.current_trust == TrustLevel.MODERATE
        assert profile.sessions_completed == 0

    def test_trust_clamped_to_zero_one(self):
        """Trust should be clamped between 0.0 and 1.0."""
        profile = AgentTrustProfile(
            agent_id="test",
            sessions_completed=10,
            total_actions=10,
            total_violations=0,
            consecutive_clean_sessions=100,
            last_session=datetime.now(timezone.utc),
            behavior_trust={"a": 1.0, "b": 1.0, "c": 1.0},
        )
        trust = profile.calculate_trust()
        assert 0.0 <= trust <= 1.0


class TestTrustScoredItem:
    """Tests for TrustScoredItem dataclass."""

    def test_default_values(self):
        item = TrustScoredItem(
            item="test",
            score=0.8,
            similarity_score=0.7,
            recency_score=0.6,
            success_score=0.5,
            confidence_score=0.4,
        )
        assert item.trust_score == 1.0
        assert item.source_agent is None
        assert item.trust_level == "MODERATE"

    def test_custom_trust_fields(self):
        item = TrustScoredItem(
            item="test",
            score=0.8,
            similarity_score=0.7,
            recency_score=0.6,
            success_score=0.5,
            confidence_score=0.4,
            trust_score=0.9,
            source_agent="helena",
            trust_level="HIGH",
        )
        assert item.trust_score == 0.9
        assert item.source_agent == "helena"
        assert item.trust_level == "HIGH"


class TestAgentTrustContext:
    """Tests for AgentTrustContext."""

    def test_from_profile(self):
        """Should create context from full profile."""
        profile = AgentTrustProfile(
            agent_id="helena",
            current_trust=0.85,
            behavior_trust={"verification_before_claim": 0.9},
        )
        ctx = AgentTrustContext.from_profile(profile)
        assert ctx.agent_id == "helena"
        assert ctx.trust_score == 0.85
        assert ctx.trust_behaviors == {"verification_before_claim": 0.9}

    def test_trust_level_property(self):
        ctx = AgentTrustContext(agent_id="test", trust_score=0.9)
        assert ctx.trust_level == "HIGH"

    def test_trust_behaviors_isolated(self):
        """Modifying context behaviors shouldn't affect the source profile."""
        profile = AgentTrustProfile(
            agent_id="test",
            behavior_trust={"a": 1.0},
        )
        ctx = AgentTrustContext.from_profile(profile)
        ctx.trust_behaviors["b"] = 0.5
        assert "b" not in profile.behavior_trust


class TestTrustAwareScorer:
    """Tests for TrustAwareScorer."""

    @pytest.fixture
    def scorer(self):
        """Scorer with a known trust profile."""
        profiles = {
            "trusted-agent": AgentTrustProfile(
                agent_id="trusted-agent",
                sessions_completed=10,
                total_actions=100,
                total_violations=0,
                consecutive_clean_sessions=10,
                last_session=datetime.now(timezone.utc),
            ),
            "untrusted-agent": AgentTrustProfile(
                agent_id="untrusted-agent",
                sessions_completed=10,
                total_actions=100,
                total_violations=80,
                consecutive_clean_sessions=0,
                last_session=datetime.now(timezone.utc),
            ),
        }
        return TrustAwareScorer(trust_profiles=profiles)

    def test_get_agent_trust_known(self, scorer):
        trust = scorer.get_agent_trust("trusted-agent")
        assert trust > TrustLevel.GOOD

    def test_get_agent_trust_unknown_returns_default(self, scorer):
        trust = scorer.get_agent_trust("unknown-agent")
        assert trust == TrustLevel.MODERATE

    def test_set_trust_profile(self, scorer):
        new_profile = AgentTrustProfile(
            agent_id="new-agent",
            sessions_completed=5,
            total_actions=50,
            total_violations=0,
            last_session=datetime.now(timezone.utc),
        )
        scorer.set_trust_profile("new-agent", new_profile)
        assert "new-agent" in scorer.trust_profiles

    # --- Heuristic scoring ---

    def test_score_heuristics_empty_list(self, scorer):
        result = scorer.score_heuristics_with_trust([])
        assert result == []

    def test_score_heuristics_returns_trust_scored_items(self, scorer):
        h = create_test_heuristic(agent="trusted-agent")
        result = scorer.score_heuristics_with_trust([h])
        assert len(result) == 1
        assert isinstance(result[0], TrustScoredItem)
        assert result[0].source_agent == "trusted-agent"

    def test_score_heuristics_sorted_descending(self, scorer):
        h1 = create_test_heuristic(agent="trusted-agent", confidence=0.9)
        h2 = create_test_heuristic(agent="untrusted-agent", confidence=0.9)
        result = scorer.score_heuristics_with_trust([h1, h2])
        assert result[0].score >= result[1].score

    def test_score_heuristics_verified_boost(self, scorer):
        """Verified heuristics should get a trust boost."""
        h_normal = create_test_heuristic(agent="trusted-agent", metadata={})
        h_verified = create_test_heuristic(
            agent="trusted-agent", metadata={"verified": True}
        )
        normal_result = scorer.score_heuristics_with_trust([h_normal])
        verified_result = scorer.score_heuristics_with_trust([h_verified])
        assert verified_result[0].trust_score >= normal_result[0].trust_score

    def test_score_heuristics_default_similarities(self, scorer):
        """Without similarities, all should default to 1.0."""
        h = create_test_heuristic(agent="trusted-agent")
        result = scorer.score_heuristics_with_trust([h])
        assert result[0].similarity_score == 1.0

    def test_score_heuristics_custom_similarities(self, scorer):
        h = create_test_heuristic(agent="trusted-agent")
        result = scorer.score_heuristics_with_trust([h], similarities=[0.3])
        assert result[0].similarity_score == 0.3

    # --- Outcome scoring ---

    def test_score_outcomes_empty_list(self, scorer):
        result = scorer.score_outcomes_with_trust([])
        assert result == []

    def test_score_outcomes_success_vs_failure(self, scorer):
        o_success = create_test_outcome(agent="trusted-agent", success=True)
        o_failure = create_test_outcome(agent="trusted-agent", success=False)
        result = scorer.score_outcomes_with_trust([o_success, o_failure])
        success_item = [r for r in result if r.item.success][0]
        failure_item = [r for r in result if not r.item.success][0]
        assert success_item.success_score == 1.0
        assert failure_item.success_score == 0.3

    def test_score_outcomes_user_feedback_boost(self, scorer):
        """Outcomes with user_feedback should get a trust boost."""
        o_no_fb = create_test_outcome(agent="trusted-agent", user_feedback=None)
        o_fb = create_test_outcome(agent="trusted-agent", user_feedback="great work")
        no_fb = scorer.score_outcomes_with_trust([o_no_fb])
        fb = scorer.score_outcomes_with_trust([o_fb])
        assert fb[0].trust_score >= no_fb[0].trust_score

    # --- Anti-pattern scoring ---

    def test_score_anti_patterns_empty_list(self, scorer):
        result = scorer.score_anti_patterns_with_trust([])
        assert result == []

    def test_score_anti_patterns_occurrence_score(self, scorer):
        """Higher occurrence count should yield higher occurrence score."""
        ap_high = create_test_anti_pattern(agent="trusted-agent", occurrence_count=10)
        ap_low = create_test_anti_pattern(agent="trusted-agent", occurrence_count=1)
        result = scorer.score_anti_patterns_with_trust([ap_high, ap_low])
        high_item = [r for r in result if r.item.occurrence_count == 10][0]
        low_item = [r for r in result if r.item.occurrence_count == 1][0]
        assert high_item.success_score > low_item.success_score

    def test_score_anti_patterns_violation_boost(self, scorer):
        """Anti-patterns from trust violations get a 20% trust boost."""
        ap_normal = create_test_anti_pattern(agent="trusted-agent", metadata={})
        ap_violation = create_test_anti_pattern(
            agent="trusted-agent", metadata={"from_trust_violation": True}
        )
        normal = scorer.score_anti_patterns_with_trust([ap_normal])
        violation = scorer.score_anti_patterns_with_trust([ap_violation])
        assert violation[0].trust_score >= normal[0].trust_score

    def test_score_anti_patterns_occurrence_capped(self, scorer):
        """Occurrence score should cap at 1.0 (count/10)."""
        ap = create_test_anti_pattern(agent="trusted-agent", occurrence_count=100)
        result = scorer.score_anti_patterns_with_trust([ap])
        assert result[0].success_score == 1.0

    # --- Full slice scoring ---

    def test_score_with_trust_full_slice(self, scorer):
        """score_with_trust should return dict with all three types."""
        memory_slice = MemorySlice(
            heuristics=[create_test_heuristic(agent="trusted-agent")],
            outcomes=[create_test_outcome(agent="trusted-agent")],
            anti_patterns=[create_test_anti_pattern(agent="trusted-agent")],
        )
        result = scorer.score_with_trust(memory_slice)
        assert "heuristics" in result
        assert "outcomes" in result
        assert "anti_patterns" in result
        assert len(result["heuristics"]) == 1
        assert len(result["outcomes"]) == 1
        assert len(result["anti_patterns"]) == 1

    def test_score_with_trust_empty_slice(self, scorer):
        result = scorer.score_with_trust(MemorySlice())
        assert result["heuristics"] == []
        assert result["outcomes"] == []
        assert result["anti_patterns"] == []

    def test_score_with_trust_custom_similarities(self, scorer):
        memory_slice = MemorySlice(
            heuristics=[create_test_heuristic(agent="trusted-agent")],
            outcomes=[create_test_outcome(agent="trusted-agent")],
        )
        sims = {"heuristics": [0.5], "outcomes": [0.8]}
        result = scorer.score_with_trust(memory_slice, similarities=sims)
        assert result["heuristics"][0].similarity_score == 0.5
        assert result["outcomes"][0].similarity_score == 0.8

    def test_trust_level_label_in_scored_items(self, scorer):
        """Scored items should have correct trust_level labels."""
        h = create_test_heuristic(agent="trusted-agent")
        result = scorer.score_heuristics_with_trust([h])
        # trusted-agent should have good+ trust
        assert result[0].trust_level in ("GOOD", "HIGH", "FULL")

    def test_custom_default_trust(self):
        """Custom default_trust should be used for unknown agents."""
        scorer = TrustAwareScorer(default_trust=0.9)
        trust = scorer.get_agent_trust("nobody")
        assert trust == 0.9

    def test_custom_weights(self):
        """Custom TrustWeights should be applied."""
        tw = TrustWeights(
            similarity=0.1,
            recency=0.1,
            success_rate=0.1,
            confidence=0.1,
            trust=0.6,
        )
        scorer = TrustAwareScorer(weights=tw)
        assert abs(scorer.trust_weights.trust - 0.6) < 0.01
