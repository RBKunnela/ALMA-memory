"""
Tests for the Confidence Engine module.
"""

import pytest

from alma.confidence import (
    ConfidenceEngine,
    ConfidenceSignal,
    OpportunitySignal,
    RiskSignal,
)


class TestRiskSignal:
    """Tests for RiskSignal dataclass."""

    def test_basic_creation(self):
        """Test basic risk signal creation."""
        risk = RiskSignal(
            signal_type="similar_to_failure",
            description="Similar to a known failure pattern",
            severity=0.7,
            source="anti_pattern:ap123",
        )

        assert risk.id is not None
        assert risk.signal_type == "similar_to_failure"
        assert risk.severity == 0.7

    def test_to_dict(self):
        """Test serialization."""
        risk = RiskSignal(
            signal_type="high_complexity",
            description="Too complex",
            severity=0.5,
            source="analysis",
        )

        data = risk.to_dict()

        assert data["signal_type"] == "high_complexity"
        assert data["severity"] == 0.5
        assert "detected_at" in data


class TestOpportunitySignal:
    """Tests for OpportunitySignal dataclass."""

    def test_basic_creation(self):
        """Test basic opportunity signal creation."""
        opp = OpportunitySignal(
            signal_type="proven_pattern",
            description="90% success rate over 10 uses",
            strength=0.9,
            source="heuristic:h123",
        )

        assert opp.id is not None
        assert opp.signal_type == "proven_pattern"
        assert opp.strength == 0.9

    def test_to_dict(self):
        """Test serialization."""
        opp = OpportunitySignal(
            signal_type="recent_success",
            description="Worked yesterday",
            strength=0.6,
            source="outcome:o456",
        )

        data = opp.to_dict()

        assert data["signal_type"] == "recent_success"
        assert data["strength"] == 0.6


class TestConfidenceSignal:
    """Tests for ConfidenceSignal dataclass."""

    def test_create(self):
        """Test creating a confidence signal."""
        signal = ConfidenceSignal.create(
            strategy="Use incremental validation",
            context="Testing a 5-field form",
            agent="Helena",
        )

        assert signal.id is not None
        assert signal.strategy == "Use incremental validation"
        assert signal.agent == "Helena"
        assert signal.recommendation == "neutral"  # Default

    def test_add_risk(self):
        """Test adding risk signals."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")

        risk = signal.add_risk(
            signal_type="high_complexity",
            description="Very complex strategy",
            severity=0.8,
            source="analysis",
        )

        assert len(signal.risk_signals) == 1
        assert signal.total_risk_score == 0.8
        assert risk.signal_type == "high_complexity"

    def test_add_opportunity(self):
        """Test adding opportunity signals."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")

        opp = signal.add_opportunity(
            signal_type="proven_pattern",
            description="Works every time",
            strength=0.9,
            source="heuristic:h1",
        )

        assert len(signal.opportunity_signals) == 1
        assert signal.total_opportunity_score == 0.9
        assert opp.signal_type == "proven_pattern"

    def test_recalculate_scores(self):
        """Test score recalculation."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.historical_success_rate = 0.8
        signal.predicted_success = 0.7
        signal.context_similarity = 0.6
        signal.uncertainty = 0.3

        signal._recalculate_scores()

        # Should have positive confidence
        assert signal.confidence_score > 0.5
        assert signal.recommendation in ["yes", "strong_yes", "neutral"]

    def test_recommendation_high_risk_override(self):
        """Test that high risk overrides confidence."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.historical_success_rate = 0.9
        signal.predicted_success = 0.9

        # Add critical risk
        signal.add_risk(
            signal_type="critical",
            description="Critical issue",
            severity=0.9,
        )

        # Should recommend avoid despite good history
        assert signal.recommendation == "avoid"

    def test_to_prompt(self):
        """Test prompt formatting."""
        signal = ConfidenceSignal.create("Test strategy", "Test context", "Helena")
        signal.historical_success_rate = 0.75
        signal.occurrence_count = 10
        signal.predicted_success = 0.8
        signal.context_similarity = 0.7
        signal.uncertainty = 0.2
        signal._recalculate_scores()

        prompt = signal.to_prompt()

        assert "Confidence Assessment" in prompt
        assert "Helena" in prompt or "Test strategy" in prompt
        assert "75%" in prompt  # Historical success
        assert "Metrics" in prompt

    def test_to_prompt_with_signals(self):
        """Test prompt with risk and opportunity signals."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.add_risk("risk1", "A risk", 0.7)
        signal.add_opportunity("opp1", "An opportunity", 0.8)

        prompt = signal.to_prompt()

        assert "Risks" in prompt
        assert "A risk" in prompt
        assert "Opportunities" in prompt
        assert "An opportunity" in prompt

    def test_to_dict(self):
        """Test serialization."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.add_risk("risk", "desc", 0.5)
        signal.add_opportunity("opp", "desc", 0.6)

        data = signal.to_dict()

        assert data["strategy"] == "Strategy"
        assert data["agent"] == "Agent"
        assert len(data["risk_signals"]) == 1
        assert len(data["opportunity_signals"]) == 1
        assert "assessed_at" in data


class TestConfidenceEngine:
    """Tests for ConfidenceEngine."""

    @pytest.fixture
    def engine(self):
        """Create a basic engine without ALMA."""
        return ConfidenceEngine()

    def test_assess_strategy_basic(self, engine):
        """Test basic strategy assessment."""
        signal = engine.assess_strategy(
            strategy="Use incremental testing",
            context="Testing a form",
            agent="Helena",
        )

        assert signal is not None
        assert signal.strategy == "Use incremental testing"
        assert signal.agent == "Helena"
        assert signal.recommendation in [
            "strong_yes",
            "yes",
            "neutral",
            "caution",
            "avoid",
        ]

    def test_assess_strategy_detects_complexity(self, engine):
        """Test that complex strategies get risk signals."""
        signal = engine.assess_strategy(
            strategy="Test all fields in every form across the entire application completely",
            context="Testing forms",
            agent="Helena",
        )

        # Should detect complexity
        complexity_risks = [
            r for r in signal.risk_signals if r.signal_type == "high_complexity"
        ]
        assert len(complexity_risks) >= 1

    def test_assess_strategy_detects_risky_patterns(self, engine):
        """Test detection of risky patterns."""
        signal = engine.assess_strategy(
            strategy="Use sleep(5) to wait for element",
            context="Testing async UI",
            agent="Helena",
        )

        risky_pattern_risks = [
            r for r in signal.risk_signals if r.signal_type == "risky_pattern"
        ]
        assert len(risky_pattern_risks) >= 1

    def test_assess_strategy_detects_best_practices(self, engine):
        """Test detection of best practice patterns."""
        signal = engine.assess_strategy(
            strategy="Use incremental validation to test first",
            context="Testing forms",
            agent="Helena",
        )

        best_practice_opps = [
            o for o in signal.opportunity_signals if o.signal_type == "best_practice"
        ]
        assert len(best_practice_opps) >= 1

    def test_rank_strategies(self, engine):
        """Test ranking multiple strategies."""
        rankings = engine.rank_strategies(
            strategies=[
                "Use sleep(10) and force delete all",  # Bad
                "Use incremental validation",  # Good
                "Test random things",  # Neutral
            ],
            context="Testing forms",
            agent="Helena",
        )

        assert len(rankings) == 3

        # First should be the incremental one (best practice)
        # Rankings are sorted by confidence score
        strategies_in_order = [s for s, _ in rankings]
        assert "incremental" in strategies_in_order[0].lower()

    def test_detect_risks_empty(self, engine):
        """Test risk detection with safe strategy."""
        risks = engine.detect_risks(
            strategy="Click submit button",
            context="Testing form",
            agent="Helena",
        )

        # Simple safe strategy shouldn't have many risks
        assert isinstance(risks, list)

    def test_detect_opportunities_with_best_practice(self, engine):
        """Test opportunity detection."""
        opportunities = engine.detect_opportunities(
            strategy="Validate input incrementally",
            context="Testing form",
            agent="Helena",
        )

        assert len(opportunities) >= 1
        types = [o.signal_type for o in opportunities]
        assert "best_practice" in types

    def test_compute_uncertainty_few_occurrences(self, engine):
        """Test uncertainty is high with few occurrences."""
        uncertainty = engine._compute_uncertainty(
            occurrence_count=1,
            context_similarity=0.5,
        )

        assert uncertainty >= 0.5  # High uncertainty

    def test_compute_uncertainty_many_occurrences(self, engine):
        """Test uncertainty is low with many occurrences."""
        uncertainty = engine._compute_uncertainty(
            occurrence_count=20,
            context_similarity=0.8,
        )

        assert uncertainty <= 0.3  # Low uncertainty

    def test_is_similar_exact_match(self, engine):
        """Test similarity detection - exact match."""
        assert engine._is_similar("Test login", "Test login") is True

    def test_is_similar_substring(self, engine):
        """Test similarity detection - substring."""
        assert engine._is_similar("login", "Test login form") is True

    def test_is_similar_word_overlap(self, engine):
        """Test similarity detection - word overlap."""
        assert engine._is_similar("test login form", "login form validation") is True

    def test_is_similar_no_match(self, engine):
        """Test similarity detection - no match."""
        assert engine._is_similar("test login", "check database") is False

    def test_generate_reasoning_with_history(self, engine):
        """Test reasoning generation."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.occurrence_count = 10
        signal.historical_success_rate = 0.9
        signal.context_similarity = 0.8

        reasoning = engine._generate_reasoning(signal)

        assert "90%" in reasoning
        assert "10 uses" in reasoning
        assert "similar" in reasoning.lower()

    def test_generate_reasoning_novel(self, engine):
        """Test reasoning for novel strategy."""
        signal = ConfidenceSignal.create("Strategy", "Context", "Agent")
        signal.occurrence_count = 0
        signal.context_similarity = 0.2

        reasoning = engine._generate_reasoning(signal)

        assert "novel" in reasoning.lower() or "no historical" in reasoning.lower()
