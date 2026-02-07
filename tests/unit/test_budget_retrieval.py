"""
Unit tests for ALMA Token Budget Management.

Tests PriorityTier, BudgetConfig, BudgetReport, TokenEstimator,
RetrievalBudget, and BudgetAwareRetrieval.
"""

import pytest

from alma.retrieval.budget import (
    BudgetConfig,
    BudgetedItem,
    BudgetReport,
    PriorityTier,
    RetrievalBudget,
    TokenEstimator,
)
from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
    create_test_outcome,
    create_test_preference,
)
from alma.types import MemorySlice


class TestPriorityTier:
    """Tests for PriorityTier enum."""

    def test_tier_ordering(self):
        """MUST_SEE < SHOULD_SEE < FETCH_ON_DEMAND < EXCLUDE."""
        assert PriorityTier.MUST_SEE.value < PriorityTier.SHOULD_SEE.value
        assert PriorityTier.SHOULD_SEE.value < PriorityTier.FETCH_ON_DEMAND.value
        assert PriorityTier.FETCH_ON_DEMAND.value < PriorityTier.EXCLUDE.value

    def test_all_four_tiers_exist(self):
        assert len(PriorityTier) == 4


class TestBudgetConfig:
    """Tests for BudgetConfig."""

    def test_default_values(self):
        config = BudgetConfig()
        assert config.max_tokens == 4000
        assert config.must_see_pct == 0.4
        assert config.should_see_pct == 0.35
        assert config.fetch_on_demand_pct == 0.25

    def test_get_tier_budget_must_see(self):
        config = BudgetConfig(max_tokens=1000, must_see_pct=0.4)
        assert config.get_tier_budget(PriorityTier.MUST_SEE) == 400

    def test_get_tier_budget_should_see(self):
        config = BudgetConfig(max_tokens=1000, should_see_pct=0.35)
        assert config.get_tier_budget(PriorityTier.SHOULD_SEE) == 350

    def test_get_tier_budget_fetch_on_demand(self):
        config = BudgetConfig(max_tokens=1000, fetch_on_demand_pct=0.25)
        assert config.get_tier_budget(PriorityTier.FETCH_ON_DEMAND) == 250

    def test_get_tier_budget_exclude_returns_zero(self):
        config = BudgetConfig()
        assert config.get_tier_budget(PriorityTier.EXCLUDE) == 0

    def test_percentages_sum_to_one(self):
        config = BudgetConfig()
        total = config.must_see_pct + config.should_see_pct + config.fetch_on_demand_pct
        assert abs(total - 1.0) < 0.01


class TestBudgetReport:
    """Tests for BudgetReport."""

    def test_utilization_pct_half_used(self):
        report = BudgetReport(total_budget=1000, used_tokens=500, remaining_tokens=500)
        assert report.utilization_pct == 50.0

    def test_utilization_pct_zero_budget(self):
        report = BudgetReport(total_budget=0, used_tokens=0, remaining_tokens=0)
        assert report.utilization_pct == 0.0

    def test_utilization_pct_full(self):
        report = BudgetReport(total_budget=1000, used_tokens=1000, remaining_tokens=0)
        assert report.utilization_pct == 100.0

    def test_default_items_dropped_empty(self):
        report = BudgetReport(total_budget=1000, used_tokens=0, remaining_tokens=1000)
        assert report.items_dropped == []
        assert report.budget_exceeded is False


class TestTokenEstimator:
    """Tests for TokenEstimator."""

    def test_estimate_heuristic(self):
        estimator = TokenEstimator(chars_per_token=4)
        h = create_test_heuristic(condition="when X", strategy="do Y")
        tokens = estimator.estimate(h)
        # "when X do Y" = 11 chars / 4 = 2 + 20 metadata = 22
        assert tokens > 0

    def test_estimate_outcome(self):
        estimator = TokenEstimator(chars_per_token=4)
        o = create_test_outcome(
            task_type="test", task_description="desc", strategy_used="strat"
        )
        tokens = estimator.estimate(o)
        assert tokens > 0

    def test_estimate_outcome_with_error(self):
        estimator = TokenEstimator(chars_per_token=4)
        o_no_err = create_test_outcome(error_message=None)
        o_err = create_test_outcome(error_message="Something failed badly")
        t_no_err = estimator.estimate(o_no_err)
        t_err = estimator.estimate(o_err)
        assert t_err > t_no_err

    def test_estimate_knowledge(self):
        estimator = TokenEstimator(chars_per_token=4)
        k = create_test_knowledge(domain="auth", fact="JWT tokens expire")
        tokens = estimator.estimate(k)
        assert tokens > 0

    def test_estimate_anti_pattern(self):
        estimator = TokenEstimator(chars_per_token=4)
        ap = create_test_anti_pattern(
            pattern="sleep waits", why_bad="flaky", better_alternative="explicit waits"
        )
        tokens = estimator.estimate(ap)
        assert tokens > 0

    def test_estimate_preference(self):
        estimator = TokenEstimator(chars_per_token=4)
        p = create_test_preference(category="style", preference="no emojis")
        tokens = estimator.estimate(p)
        assert tokens > 0

    def test_estimate_string(self):
        estimator = TokenEstimator(chars_per_token=4)
        tokens = estimator.estimate("hello world")  # 11 chars / 4 = 2
        assert tokens == 2

    def test_estimate_unknown_type(self):
        estimator = TokenEstimator()
        tokens = estimator.estimate(42)
        assert tokens == 50  # Default estimate

    def test_estimate_slice(self):
        estimator = TokenEstimator(chars_per_token=4)
        memory_slice = MemorySlice(
            heuristics=[create_test_heuristic()],
            outcomes=[create_test_outcome()],
            domain_knowledge=[create_test_knowledge()],
            anti_patterns=[create_test_anti_pattern()],
            preferences=[create_test_preference()],
        )
        total = estimator.estimate_slice(memory_slice)
        assert total > 0
        # Should be sum of individual estimates
        individual_sum = (
            estimator.estimate(memory_slice.heuristics[0])
            + estimator.estimate(memory_slice.outcomes[0])
            + estimator.estimate(memory_slice.domain_knowledge[0])
            + estimator.estimate(memory_slice.anti_patterns[0])
            + estimator.estimate(memory_slice.preferences[0])
        )
        assert total == individual_sum

    def test_estimate_empty_slice(self):
        estimator = TokenEstimator()
        assert estimator.estimate_slice(MemorySlice()) == 0

    def test_custom_chars_per_token(self):
        est_4 = TokenEstimator(chars_per_token=4)
        est_2 = TokenEstimator(chars_per_token=2)
        text = "abcdefgh"  # 8 chars
        assert est_4.estimate(text) == 2  # 8/4
        assert est_2.estimate(text) == 4  # 8/2


class TestRetrievalBudget:
    """Tests for RetrievalBudget."""

    @pytest.fixture
    def budget(self):
        config = BudgetConfig(max_tokens=1000)
        return RetrievalBudget(config=config)

    def test_initial_state(self, budget):
        assert budget.used_tokens == 0
        assert budget.remaining_tokens == 1000

    def test_can_include_within_budget(self, budget):
        h = create_test_heuristic(condition="x", strategy="y")
        assert budget.can_include(h, PriorityTier.MUST_SEE) is True

    def test_can_include_exclude_tier(self, budget):
        h = create_test_heuristic()
        assert budget.can_include(h, PriorityTier.EXCLUDE) is False

    def test_include_tracks_tokens(self, budget):
        h = create_test_heuristic(condition="x", strategy="y")
        budgeted = budget.include(h, "heuristic", PriorityTier.MUST_SEE)
        assert budgeted.included is True
        assert budget.used_tokens > 0
        assert budget.remaining_tokens < 1000

    def test_include_over_budget_excluded(self):
        """Items that don't fit should be excluded."""
        config = BudgetConfig(max_tokens=10, must_see_pct=1.0)
        budget = RetrievalBudget(config=config)

        h = create_test_heuristic(
            condition="a very long condition that exceeds the tiny budget",
            strategy="with an equally verbose strategy description",
        )
        budgeted = budget.include(h, "heuristic", PriorityTier.MUST_SEE)
        assert budgeted.included is False

    def test_include_force_over_budget(self):
        """Force=True should include even when over budget."""
        config = BudgetConfig(max_tokens=10, must_see_pct=1.0)
        budget = RetrievalBudget(config=config)

        h = create_test_heuristic(
            condition="a very long condition that exceeds the tiny budget",
            strategy="with an equally verbose strategy description",
        )
        budgeted = budget.include(h, "heuristic", PriorityTier.MUST_SEE, force=True)
        assert budgeted.included is True

    def test_reset(self, budget):
        h = create_test_heuristic()
        budget.include(h, "heuristic", PriorityTier.MUST_SEE)
        assert budget.used_tokens > 0

        budget.reset()
        assert budget.used_tokens == 0
        assert budget.remaining_tokens == 1000

    def test_tier_budget_enforcement(self):
        """Items should be rejected when their tier budget is exhausted."""
        config = BudgetConfig(
            max_tokens=10000,
            must_see_pct=0.01,  # Very small MUST_SEE budget (100 tokens)
        )
        budget = RetrievalBudget(config=config)

        # Fill up MUST_SEE tier
        for i in range(20):
            h = create_test_heuristic(
                condition=f"condition {i} with enough text to use tokens",
                strategy=f"strategy {i} with enough text",
            )
            budget.include(h, "heuristic", PriorityTier.MUST_SEE)

        # Next item should exceed tier budget
        h_extra = create_test_heuristic(
            condition="one more condition",
            strategy="one more strategy",
        )
        can_fit = budget.can_include(h_extra, PriorityTier.MUST_SEE)
        assert can_fit is False

    def test_apply_budget_basic(self, budget):
        """apply_budget should return filtered slice and report."""
        memory_slice = MemorySlice(
            heuristics=[create_test_heuristic()],
            outcomes=[create_test_outcome()],
            preferences=[create_test_preference()],
            domain_knowledge=[create_test_knowledge()],
            anti_patterns=[create_test_anti_pattern()],
        )
        result_slice, report = budget.apply_budget(memory_slice)

        assert isinstance(result_slice, MemorySlice)
        assert isinstance(report, BudgetReport)
        assert report.included_count > 0
        assert report.total_budget == 1000

    def test_apply_budget_respects_type_limits(self):
        """apply_budget should respect max_heuristics etc."""
        config = BudgetConfig(max_tokens=50000, max_heuristics=2)
        budget = RetrievalBudget(config=config)

        heuristics = [create_test_heuristic() for _ in range(5)]
        memory_slice = MemorySlice(heuristics=heuristics)

        result_slice, report = budget.apply_budget(memory_slice)
        assert len(result_slice.heuristics) <= 2

    def test_apply_budget_adds_metadata(self, budget):
        memory_slice = MemorySlice(heuristics=[create_test_heuristic()])
        result_slice, _ = budget.apply_budget(memory_slice)
        assert "budget_report" in result_slice.metadata

    def test_apply_budget_empty_slice(self, budget):
        result_slice, report = budget.apply_budget(MemorySlice())
        assert report.used_tokens == 0
        assert report.included_count == 0

    def test_apply_budget_custom_priorities(self, budget):
        """Custom type_priorities should override defaults."""
        memory_slice = MemorySlice(
            heuristics=[create_test_heuristic()],
        )
        priorities = {"heuristic": PriorityTier.FETCH_ON_DEMAND}
        result_slice, report = budget.apply_budget(memory_slice, priorities)
        # Heuristics processed with FETCH_ON_DEMAND priority
        assert isinstance(result_slice, MemorySlice)

    def test_apply_budget_resets_between_calls(self, budget):
        """Each apply_budget call should start fresh."""
        memory_slice = MemorySlice(heuristics=[create_test_heuristic()])
        _, report1 = budget.apply_budget(memory_slice)
        _, report2 = budget.apply_budget(memory_slice)
        assert report1.used_tokens == report2.used_tokens

    def test_default_classifier_anti_pattern_is_must_see(self, budget):
        ap = create_test_anti_pattern()
        tier = budget._default_classifier(ap, "anti_pattern")
        assert tier == PriorityTier.MUST_SEE

    def test_default_classifier_preference_is_must_see(self, budget):
        p = create_test_preference()
        tier = budget._default_classifier(p, "preference")
        assert tier == PriorityTier.MUST_SEE

    def test_default_classifier_high_confidence_heuristic(self, budget):
        h = create_test_heuristic(confidence=0.9)
        tier = budget._default_classifier(h, "heuristic")
        assert tier == PriorityTier.MUST_SEE

    def test_default_classifier_low_confidence_heuristic(self, budget):
        h = create_test_heuristic(confidence=0.5)
        tier = budget._default_classifier(h, "heuristic")
        assert tier == PriorityTier.SHOULD_SEE

    def test_default_classifier_successful_outcome(self, budget):
        o = create_test_outcome(success=True)
        tier = budget._default_classifier(o, "outcome")
        assert tier == PriorityTier.SHOULD_SEE

    def test_default_classifier_failed_outcome(self, budget):
        o = create_test_outcome(success=False)
        tier = budget._default_classifier(o, "outcome")
        assert tier == PriorityTier.FETCH_ON_DEMAND

    def test_default_classifier_high_confidence_knowledge(self, budget):
        k = create_test_knowledge(confidence=0.8)
        tier = budget._default_classifier(k, "domain_knowledge")
        assert tier == PriorityTier.SHOULD_SEE

    def test_default_classifier_low_confidence_knowledge(self, budget):
        k = create_test_knowledge(confidence=0.3)
        tier = budget._default_classifier(k, "domain_knowledge")
        assert tier == PriorityTier.FETCH_ON_DEMAND

    def test_default_classifier_unknown_type(self, budget):
        tier = budget._default_classifier("something", "unknown")
        assert tier == PriorityTier.SHOULD_SEE

    def test_get_fetch_on_demand_ids(self, budget):
        h = create_test_heuristic(id="h-123")
        budget.include(h, "heuristic", PriorityTier.FETCH_ON_DEMAND)
        ids = budget.get_fetch_on_demand_ids()
        assert "h-123" in ids

    def test_get_fetch_on_demand_ids_excludes_other_tiers(self, budget):
        h1 = create_test_heuristic(id="h-must")
        h2 = create_test_heuristic(id="h-fod")
        budget.include(h1, "heuristic", PriorityTier.MUST_SEE)
        budget.include(h2, "heuristic", PriorityTier.FETCH_ON_DEMAND)
        ids = budget.get_fetch_on_demand_ids()
        assert "h-must" not in ids
        assert "h-fod" in ids

    def test_excluded_items_tracked(self):
        """Excluded items should be tracked in _excluded list."""
        config = BudgetConfig(max_tokens=10, must_see_pct=1.0)
        budget = RetrievalBudget(config=config)

        h = create_test_heuristic(
            id="big-h",
            condition="a very long condition text that will not fit",
            strategy="equally long strategy text",
        )
        budget.include(h, "heuristic", PriorityTier.MUST_SEE)
        # If excluded, it should be tracked
        if not budget._items[-1].included:
            assert len(budget._excluded) > 0


class TestBudgetedItem:
    """Tests for BudgetedItem dataclass."""

    def test_default_values(self):
        item = BudgetedItem(
            item="test",
            memory_type="heuristic",
            priority=PriorityTier.MUST_SEE,
            estimated_tokens=50,
        )
        assert item.included is True
        assert item.truncated is False
        assert item.summary_only is False
