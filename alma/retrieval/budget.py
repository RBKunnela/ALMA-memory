"""
ALMA Token Budget Management.

Implements attention budget tracking for retrieval to prevent context overflow.
Based on context engineering principles: "Context window space is finite and expensive."

Features:
- Token counting with configurable estimator
- Priority-based inclusion (must-see → should-see → fetch-on-demand)
- Budget enforcement with graceful degradation
- Tracking and metrics for optimization
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemorySlice,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)


class PriorityTier(Enum):
    """
    Priority tiers for attention budget allocation.

    Based on context engineering guidance:
    - MUST_SEE: In the window for every step, no exceptions
    - SHOULD_SEE: Important but can be summarized if needed
    - FETCH_ON_DEMAND: Referenced but not included; agent can request
    - EXCLUDE: Never read again; processed and compressed
    """

    MUST_SEE = 1
    SHOULD_SEE = 2
    FETCH_ON_DEMAND = 3
    EXCLUDE = 4


@dataclass
class BudgetConfig:
    """Configuration for token budget management."""

    # Total budget in tokens
    max_tokens: int = 4000

    # Per-tier allocation (percentages of max_tokens)
    must_see_pct: float = 0.4  # 40% for critical memories
    should_see_pct: float = 0.35  # 35% for important memories
    fetch_on_demand_pct: float = 0.25  # 25% reserved for on-demand

    # Per-type limits (within tier allocation)
    max_heuristics: int = 10
    max_outcomes: int = 10
    max_knowledge: int = 5
    max_anti_patterns: int = 5
    max_preferences: int = 5

    # Token estimation
    chars_per_token: int = 4  # Rough estimate: 4 chars = 1 token

    # Truncation settings
    truncate_long_content: bool = True
    max_content_chars: int = 500  # Truncate individual items

    def get_tier_budget(self, tier: PriorityTier) -> int:
        """Get token budget for a priority tier."""
        if tier == PriorityTier.MUST_SEE:
            return int(self.max_tokens * self.must_see_pct)
        elif tier == PriorityTier.SHOULD_SEE:
            return int(self.max_tokens * self.should_see_pct)
        elif tier == PriorityTier.FETCH_ON_DEMAND:
            return int(self.max_tokens * self.fetch_on_demand_pct)
        return 0


@dataclass
class BudgetedItem:
    """A memory item with budget metadata."""

    item: Any
    memory_type: str
    priority: PriorityTier
    estimated_tokens: int
    included: bool = True
    truncated: bool = False
    summary_only: bool = False


@dataclass
class BudgetReport:
    """Report on budget usage after retrieval."""

    total_budget: int
    used_tokens: int
    remaining_tokens: int

    # Per-tier breakdown
    must_see_used: int = 0
    must_see_budget: int = 0
    should_see_used: int = 0
    should_see_budget: int = 0
    fetch_on_demand_count: int = 0

    # Items by status
    included_count: int = 0
    excluded_count: int = 0
    truncated_count: int = 0
    summary_only_count: int = 0

    # Warnings
    budget_exceeded: bool = False
    items_dropped: List[str] = field(default_factory=list)

    @property
    def utilization_pct(self) -> float:
        """Budget utilization percentage."""
        if self.total_budget == 0:
            return 0.0
        return (self.used_tokens / self.total_budget) * 100


class TokenEstimator:
    """Estimates token count for memory items."""

    def __init__(self, chars_per_token: int = 4):
        self.chars_per_token = chars_per_token

    def estimate(self, item: Any) -> int:
        """Estimate tokens for any memory item."""
        if isinstance(item, Heuristic):
            return self._estimate_heuristic(item)
        elif isinstance(item, Outcome):
            return self._estimate_outcome(item)
        elif isinstance(item, DomainKnowledge):
            return self._estimate_knowledge(item)
        elif isinstance(item, AntiPattern):
            return self._estimate_anti_pattern(item)
        elif isinstance(item, UserPreference):
            return self._estimate_preference(item)
        elif isinstance(item, str):
            return len(item) // self.chars_per_token
        return 50  # Default estimate

    def _estimate_heuristic(self, h: Heuristic) -> int:
        """Estimate tokens for a heuristic."""
        text = f"{h.condition} {h.strategy}"
        return len(text) // self.chars_per_token + 20  # +20 for metadata

    def _estimate_outcome(self, o: Outcome) -> int:
        """Estimate tokens for an outcome."""
        text = f"{o.task_type} {o.task_description} {o.strategy_used}"
        if o.error_message:
            text += f" {o.error_message}"
        return len(text) // self.chars_per_token + 15

    def _estimate_knowledge(self, k: DomainKnowledge) -> int:
        """Estimate tokens for domain knowledge."""
        # Knowledge can have complex 'fact' structures
        fact_str = str(k.fact) if k.fact else ""
        text = f"{k.domain} {fact_str}"
        return len(text) // self.chars_per_token + 10

    def _estimate_anti_pattern(self, ap: AntiPattern) -> int:
        """Estimate tokens for an anti-pattern."""
        text = f"{ap.pattern} {ap.why_bad} {ap.better_alternative}"
        return len(text) // self.chars_per_token + 15

    def _estimate_preference(self, p: UserPreference) -> int:
        """Estimate tokens for a preference."""
        text = f"{p.category} {p.preference} {p.source or ''}"
        return len(text) // self.chars_per_token + 10

    def estimate_slice(self, memory_slice: MemorySlice) -> int:
        """Estimate total tokens for a MemorySlice."""
        total = 0
        for h in memory_slice.heuristics:
            total += self._estimate_heuristic(h)
        for o in memory_slice.outcomes:
            total += self._estimate_outcome(o)
        for k in memory_slice.domain_knowledge:
            total += self._estimate_knowledge(k)
        for ap in memory_slice.anti_patterns:
            total += self._estimate_anti_pattern(ap)
        for p in memory_slice.preferences:
            total += self._estimate_preference(p)
        return total


class RetrievalBudget:
    """
    Manages token budget for memory retrieval.

    Ensures retrieval results fit within context window limits
    while prioritizing the most important memories.

    Usage:
        budget = RetrievalBudget(config=BudgetConfig(max_tokens=4000))

        # Check before including
        if budget.can_include(memory, priority=PriorityTier.MUST_SEE):
            budget.include(memory, "heuristic", PriorityTier.MUST_SEE)

        # Or process entire slice
        budgeted_slice, report = budget.apply_budget(memory_slice, priorities)
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        estimator: Optional[TokenEstimator] = None,
        priority_classifier: Optional[Callable[[Any, str], PriorityTier]] = None,
    ):
        self.config = config or BudgetConfig()
        self.estimator = estimator or TokenEstimator(self.config.chars_per_token)
        self.priority_classifier = priority_classifier or self._default_classifier

        # Tracking
        self._used_tokens = 0
        self._tier_usage: Dict[PriorityTier, int] = {
            PriorityTier.MUST_SEE: 0,
            PriorityTier.SHOULD_SEE: 0,
            PriorityTier.FETCH_ON_DEMAND: 0,
        }
        self._items: List[BudgetedItem] = []
        self._excluded: List[str] = []

    def reset(self) -> None:
        """Reset budget tracking for new retrieval."""
        self._used_tokens = 0
        self._tier_usage = {
            PriorityTier.MUST_SEE: 0,
            PriorityTier.SHOULD_SEE: 0,
            PriorityTier.FETCH_ON_DEMAND: 0,
        }
        self._items = []
        self._excluded = []

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining in total budget."""
        return max(0, self.config.max_tokens - self._used_tokens)

    @property
    def used_tokens(self) -> int:
        """Tokens used so far."""
        return self._used_tokens

    def can_include(
        self,
        item: Any,
        priority: PriorityTier = PriorityTier.SHOULD_SEE,
    ) -> bool:
        """Check if an item can be included within budget."""
        if priority == PriorityTier.EXCLUDE:
            return False

        estimated = self.estimator.estimate(item)
        tier_budget = self.config.get_tier_budget(priority)
        tier_used = self._tier_usage.get(priority, 0)

        # Check tier budget
        if tier_used + estimated > tier_budget:
            return False

        # Check total budget
        if self._used_tokens + estimated > self.config.max_tokens:
            return False

        return True

    def include(
        self,
        item: Any,
        memory_type: str,
        priority: PriorityTier = PriorityTier.SHOULD_SEE,
        force: bool = False,
    ) -> BudgetedItem:
        """
        Include an item in the budget.

        Args:
            item: Memory item to include
            memory_type: Type name (heuristic, outcome, etc.)
            priority: Priority tier for allocation
            force: Include even if over budget (for MUST_SEE items)

        Returns:
            BudgetedItem with inclusion status
        """
        estimated = self.estimator.estimate(item)
        can_fit = self.can_include(item, priority)

        budgeted = BudgetedItem(
            item=item,
            memory_type=memory_type,
            priority=priority,
            estimated_tokens=estimated,
            included=can_fit or force,
        )

        if budgeted.included:
            self._used_tokens += estimated
            self._tier_usage[priority] = self._tier_usage.get(priority, 0) + estimated
        else:
            item_desc = f"{memory_type}:{getattr(item, 'id', 'unknown')}"
            self._excluded.append(item_desc)

        self._items.append(budgeted)
        return budgeted

    def apply_budget(
        self,
        memory_slice: MemorySlice,
        type_priorities: Optional[Dict[str, PriorityTier]] = None,
    ) -> Tuple[MemorySlice, BudgetReport]:
        """
        Apply budget constraints to a MemorySlice.

        Args:
            memory_slice: Raw retrieval results
            type_priorities: Optional priority overrides per type

        Returns:
            Tuple of (budgeted MemorySlice, BudgetReport)
        """
        self.reset()

        # Default priorities
        priorities = type_priorities or {
            "heuristic": PriorityTier.MUST_SEE,
            "outcome": PriorityTier.SHOULD_SEE,
            "domain_knowledge": PriorityTier.SHOULD_SEE,
            "anti_pattern": PriorityTier.MUST_SEE,  # Important for avoiding mistakes
            "preference": PriorityTier.MUST_SEE,  # User prefs are critical
        }

        # Process each type in priority order
        included_heuristics = []
        included_outcomes = []
        included_knowledge = []
        included_anti_patterns = []
        included_preferences = []

        # MUST_SEE first (preferences, anti-patterns, heuristics)
        for pref in memory_slice.preferences[: self.config.max_preferences]:
            budgeted = self.include(
                pref, "preference", priorities.get("preference", PriorityTier.MUST_SEE)
            )
            if budgeted.included:
                included_preferences.append(pref)

        for ap in memory_slice.anti_patterns[: self.config.max_anti_patterns]:
            budgeted = self.include(
                ap,
                "anti_pattern",
                priorities.get("anti_pattern", PriorityTier.MUST_SEE),
            )
            if budgeted.included:
                included_anti_patterns.append(ap)

        for h in memory_slice.heuristics[: self.config.max_heuristics]:
            budgeted = self.include(
                h, "heuristic", priorities.get("heuristic", PriorityTier.MUST_SEE)
            )
            if budgeted.included:
                included_heuristics.append(h)

        # SHOULD_SEE next (outcomes, knowledge)
        for o in memory_slice.outcomes[: self.config.max_outcomes]:
            budgeted = self.include(
                o, "outcome", priorities.get("outcome", PriorityTier.SHOULD_SEE)
            )
            if budgeted.included:
                included_outcomes.append(o)

        for k in memory_slice.domain_knowledge[: self.config.max_knowledge]:
            budgeted = self.include(
                k,
                "domain_knowledge",
                priorities.get("domain_knowledge", PriorityTier.SHOULD_SEE),
            )
            if budgeted.included:
                included_knowledge.append(k)

        # Build report
        report = BudgetReport(
            total_budget=self.config.max_tokens,
            used_tokens=self._used_tokens,
            remaining_tokens=self.remaining_tokens,
            must_see_used=self._tier_usage.get(PriorityTier.MUST_SEE, 0),
            must_see_budget=self.config.get_tier_budget(PriorityTier.MUST_SEE),
            should_see_used=self._tier_usage.get(PriorityTier.SHOULD_SEE, 0),
            should_see_budget=self.config.get_tier_budget(PriorityTier.SHOULD_SEE),
            included_count=len([i for i in self._items if i.included]),
            excluded_count=len([i for i in self._items if not i.included]),
            truncated_count=len([i for i in self._items if i.truncated]),
            summary_only_count=len([i for i in self._items if i.summary_only]),
            budget_exceeded=self._used_tokens > self.config.max_tokens,
            items_dropped=self._excluded,
        )

        # Build budgeted slice
        budgeted_slice = MemorySlice(
            heuristics=included_heuristics,
            outcomes=included_outcomes,
            preferences=included_preferences,
            domain_knowledge=included_knowledge,
            anti_patterns=included_anti_patterns,
            query=memory_slice.query,
            agent=memory_slice.agent,
            retrieval_time_ms=memory_slice.retrieval_time_ms,
        )

        # Add budget metadata
        budgeted_slice.metadata["budget_report"] = {
            "total_budget": report.total_budget,
            "used_tokens": report.used_tokens,
            "utilization_pct": report.utilization_pct,
            "items_dropped": len(report.items_dropped),
        }

        logger.info(
            f"Budget applied: {report.used_tokens}/{report.total_budget} tokens "
            f"({report.utilization_pct:.1f}%), "
            f"{report.included_count} included, {report.excluded_count} excluded"
        )

        return budgeted_slice, report

    def _default_classifier(self, item: Any, memory_type: str) -> PriorityTier:
        """Default priority classification based on memory type and attributes."""
        # Anti-patterns and preferences are always high priority
        if memory_type in ("anti_pattern", "preference"):
            return PriorityTier.MUST_SEE

        # High-confidence heuristics are must-see
        if memory_type == "heuristic":
            if hasattr(item, "confidence") and item.confidence >= 0.8:
                return PriorityTier.MUST_SEE
            return PriorityTier.SHOULD_SEE

        # Recent successful outcomes
        if memory_type == "outcome":
            if hasattr(item, "success") and item.success:
                return PriorityTier.SHOULD_SEE
            return PriorityTier.FETCH_ON_DEMAND

        # Domain knowledge by confidence
        if memory_type == "domain_knowledge":
            if hasattr(item, "confidence") and item.confidence >= 0.7:
                return PriorityTier.SHOULD_SEE
            return PriorityTier.FETCH_ON_DEMAND

        return PriorityTier.SHOULD_SEE

    def get_fetch_on_demand_ids(self) -> List[str]:
        """Get IDs of items marked for fetch-on-demand."""
        return [
            getattr(i.item, "id", None)
            for i in self._items
            if i.priority == PriorityTier.FETCH_ON_DEMAND and hasattr(i.item, "id")
        ]


class BudgetAwareRetrieval:
    """
    Wrapper that adds budget management to retrieval operations.

    Usage:
        budget_retrieval = BudgetAwareRetrieval(
            retrieval_engine,
            budget_config=BudgetConfig(max_tokens=4000)
        )

        result, report = budget_retrieval.retrieve_with_budget(
            query="...",
            agent="helena",
            project_id="my-project"
        )
    """

    def __init__(
        self,
        retrieval_engine: Any,  # RetrievalEngine
        budget_config: Optional[BudgetConfig] = None,
    ):
        self.engine = retrieval_engine
        self.budget = RetrievalBudget(config=budget_config)

    def retrieve_with_budget(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> Tuple[MemorySlice, BudgetReport]:
        """
        Retrieve memories with budget enforcement.

        Returns:
            Tuple of (budgeted MemorySlice, BudgetReport)
        """
        # Get raw results (request more than needed for budget filtering)
        raw_slice = self.engine.retrieve(
            query=query,
            agent=agent,
            project_id=project_id,
            user_id=user_id,
            top_k=top_k * 2,  # Get extra for filtering
            **kwargs,
        )

        # Apply budget
        return self.budget.apply_budget(raw_slice)
