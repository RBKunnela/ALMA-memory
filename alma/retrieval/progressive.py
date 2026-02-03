"""
ALMA Progressive Disclosure.

Implements summary → detail retrieval pattern to optimize context usage.
Based on context engineering principle: "Must know exists" before "must see."

Features:
- Summary extraction for memory items
- Lazy-loading of full details on demand
- Reference IDs for fetch-on-demand pattern
- Tiered disclosure (summary → key details → full content)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemorySlice,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)


class DisclosureLevel(Enum):
    """Levels of detail for progressive disclosure."""

    REFERENCE = 1  # Just ID and type - agent knows it exists
    SUMMARY = 2  # Brief summary - enough to decide if needed
    KEY_DETAILS = 3  # Important fields only
    FULL = 4  # Complete memory with all details


@dataclass
class MemorySummary:
    """Compact summary of a memory item."""

    id: str
    memory_type: str
    summary: str
    relevance_hint: str  # Why this might be relevant
    estimated_tokens: int
    disclosure_level: DisclosureLevel = DisclosureLevel.SUMMARY

    # Optional key details (for KEY_DETAILS level)
    key_fields: Dict[str, Any] = field(default_factory=dict)

    # Reference to full item (for lazy loading)
    _full_item: Optional[Any] = field(default=None, repr=False)

    def get_full(self) -> Optional[Any]:
        """Get full item if available."""
        return self._full_item


@dataclass
class ProgressiveSlice:
    """
    Memory slice with progressive disclosure support.

    Contains summaries by default, with ability to fetch full details
    on demand.
    """

    # Summaries by type
    heuristic_summaries: List[MemorySummary] = field(default_factory=list)
    outcome_summaries: List[MemorySummary] = field(default_factory=list)
    knowledge_summaries: List[MemorySummary] = field(default_factory=list)
    anti_pattern_summaries: List[MemorySummary] = field(default_factory=list)
    preference_summaries: List[MemorySummary] = field(default_factory=list)

    # Full items (populated on demand)
    _full_items: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Metadata
    query: str = ""
    agent: str = ""
    total_available: int = 0
    summaries_included: int = 0
    estimated_summary_tokens: int = 0

    @property
    def all_summaries(self) -> List[MemorySummary]:
        """Get all summaries across types."""
        return (
            self.heuristic_summaries
            + self.outcome_summaries
            + self.knowledge_summaries
            + self.anti_pattern_summaries
            + self.preference_summaries
        )

    def get_full_item(self, memory_id: str) -> Optional[Any]:
        """Get full item by ID (lazy load if needed)."""
        # Check cache first
        if memory_id in self._full_items:
            return self._full_items[memory_id]

        # Check summaries for attached full items
        for summary in self.all_summaries:
            if summary.id == memory_id and summary._full_item:
                self._full_items[memory_id] = summary._full_item
                return summary._full_item

        return None

    def get_ids_by_type(self, memory_type: str) -> List[str]:
        """Get all IDs for a memory type."""
        type_map = {
            "heuristic": self.heuristic_summaries,
            "outcome": self.outcome_summaries,
            "domain_knowledge": self.knowledge_summaries,
            "anti_pattern": self.anti_pattern_summaries,
            "preference": self.preference_summaries,
        }
        summaries = type_map.get(memory_type, [])
        return [s.id for s in summaries]


class SummaryExtractor:
    """Extracts summaries from memory items."""

    def __init__(
        self,
        max_summary_length: int = 100,
        chars_per_token: int = 4,
    ):
        self.max_summary_length = max_summary_length
        self.chars_per_token = chars_per_token

    def extract_heuristic_summary(
        self,
        h: Heuristic,
        level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> MemorySummary:
        """Extract summary from a heuristic."""
        if level == DisclosureLevel.REFERENCE:
            summary = f"Heuristic: {h.condition[:30]}..."
            relevance = "Learned pattern"
        elif level == DisclosureLevel.SUMMARY:
            summary = self._truncate(
                f"When {h.condition}, {h.strategy}",
                self.max_summary_length,
            )
            relevance = f"Success rate: {h.success_rate:.0%}, Confidence: {h.confidence:.0%}"
        else:  # KEY_DETAILS or FULL
            summary = f"When {h.condition}, {h.strategy}"
            relevance = f"Used {h.occurrence_count}x, {h.success_rate:.0%} success"

        key_fields = {}
        if level >= DisclosureLevel.KEY_DETAILS:
            key_fields = {
                "confidence": h.confidence,
                "success_rate": h.success_rate,
                "occurrence_count": h.occurrence_count,
            }

        return MemorySummary(
            id=h.id,
            memory_type="heuristic",
            summary=summary,
            relevance_hint=relevance,
            estimated_tokens=len(summary) // self.chars_per_token + 10,
            disclosure_level=level,
            key_fields=key_fields,
            _full_item=h if level == DisclosureLevel.FULL else None,
        )

    def extract_outcome_summary(
        self,
        o: Outcome,
        level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> MemorySummary:
        """Extract summary from an outcome."""
        status = "Success" if o.success else "Failed"

        if level == DisclosureLevel.REFERENCE:
            summary = f"Outcome: {o.task_type} ({status})"
            relevance = f"{o.task_type} task"
        elif level == DisclosureLevel.SUMMARY:
            summary = self._truncate(
                f"{status}: {o.task_description} using {o.strategy_used}",
                self.max_summary_length,
            )
            relevance = f"{o.task_type} - {status}"
        else:
            summary = f"{status}: {o.task_description}\nStrategy: {o.strategy_used}"
            if o.error_message:
                summary += f"\nError: {o.error_message}"
            relevance = f"{o.task_type} task outcome"

        key_fields = {}
        if level >= DisclosureLevel.KEY_DETAILS:
            key_fields = {
                "success": o.success,
                "task_type": o.task_type,
                "strategy_used": o.strategy_used,
            }
            if o.error_message:
                key_fields["error"] = o.error_message[:100]

        return MemorySummary(
            id=o.id,
            memory_type="outcome",
            summary=summary,
            relevance_hint=relevance,
            estimated_tokens=len(summary) // self.chars_per_token + 10,
            disclosure_level=level,
            key_fields=key_fields,
            _full_item=o if level == DisclosureLevel.FULL else None,
        )

    def extract_knowledge_summary(
        self,
        k: DomainKnowledge,
        level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> MemorySummary:
        """Extract summary from domain knowledge."""
        fact_str = str(k.fact)[:100] if k.fact else "N/A"

        if level == DisclosureLevel.REFERENCE:
            summary = f"Knowledge: {k.domain}"
            relevance = f"Domain: {k.domain}"
        elif level == DisclosureLevel.SUMMARY:
            summary = self._truncate(
                f"[{k.domain}] {fact_str}",
                self.max_summary_length,
            )
            relevance = f"Confidence: {k.confidence:.0%}"
        else:
            summary = f"Domain: {k.domain}\nFact: {fact_str}\nSource: {k.source}"
            relevance = f"Domain knowledge, {k.confidence:.0%} confidence"

        key_fields = {}
        if level >= DisclosureLevel.KEY_DETAILS:
            key_fields = {
                "domain": k.domain,
                "confidence": k.confidence,
                "source": k.source,
            }

        return MemorySummary(
            id=k.id,
            memory_type="domain_knowledge",
            summary=summary,
            relevance_hint=relevance,
            estimated_tokens=len(summary) // self.chars_per_token + 10,
            disclosure_level=level,
            key_fields=key_fields,
            _full_item=k if level == DisclosureLevel.FULL else None,
        )

    def extract_anti_pattern_summary(
        self,
        ap: AntiPattern,
        level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> MemorySummary:
        """Extract summary from an anti-pattern."""
        if level == DisclosureLevel.REFERENCE:
            summary = f"Warning: {ap.pattern[:30]}..."
            relevance = "Known pitfall"
        elif level == DisclosureLevel.SUMMARY:
            summary = self._truncate(
                f"Avoid: {ap.pattern}. {ap.why_bad}",
                self.max_summary_length,
            )
            relevance = f"Seen {ap.occurrence_count}x"
        else:
            summary = (
                f"Pattern to avoid: {ap.pattern}\n"
                f"Why bad: {ap.why_bad}\n"
                f"Instead: {ap.better_alternative}"
            )
            relevance = f"Occurred {ap.occurrence_count}x"

        key_fields = {}
        if level >= DisclosureLevel.KEY_DETAILS:
            key_fields = {
                "pattern": ap.pattern,
                "occurrence_count": ap.occurrence_count,
                "alternative": ap.better_alternative,
            }

        return MemorySummary(
            id=ap.id,
            memory_type="anti_pattern",
            summary=summary,
            relevance_hint=relevance,
            estimated_tokens=len(summary) // self.chars_per_token + 10,
            disclosure_level=level,
            key_fields=key_fields,
            _full_item=ap if level == DisclosureLevel.FULL else None,
        )

    def extract_preference_summary(
        self,
        p: UserPreference,
        level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ) -> MemorySummary:
        """Extract summary from a user preference."""
        if level == DisclosureLevel.REFERENCE:
            summary = f"Preference: {p.category}"
            relevance = "User constraint"
        elif level == DisclosureLevel.SUMMARY:
            summary = self._truncate(
                f"[{p.category}] {p.preference}",
                self.max_summary_length,
            )
            relevance = f"Priority: {p.priority}"
        else:
            summary = (
                f"Category: {p.category}\n"
                f"Preference: {p.preference}\n"
                f"Context: {p.context or 'General'}"
            )
            relevance = f"User preference, priority {p.priority}"

        key_fields = {}
        if level >= DisclosureLevel.KEY_DETAILS:
            key_fields = {
                "category": p.category,
                "priority": p.priority,
            }

        return MemorySummary(
            id=p.id,
            memory_type="preference",
            summary=summary,
            relevance_hint=relevance,
            estimated_tokens=len(summary) // self.chars_per_token + 10,
            disclosure_level=level,
            key_fields=key_fields,
            _full_item=p if level == DisclosureLevel.FULL else None,
        )

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."


class ProgressiveRetrieval:
    """
    Retrieval with progressive disclosure support.

    Returns summaries first, allowing agents to request full details
    only for items they need.

    Usage:
        progressive = ProgressiveRetrieval(retrieval_engine, storage)

        # Get summaries
        slice = progressive.retrieve_summaries(query, agent, project_id)

        # Get full details for specific items
        full_heuristic = progressive.get_full_item("heuristic-123", "heuristic")
    """

    def __init__(
        self,
        retrieval_engine: Any,  # RetrievalEngine
        storage: Any,  # StorageBackend
        default_level: DisclosureLevel = DisclosureLevel.SUMMARY,
    ):
        self.engine = retrieval_engine
        self.storage = storage
        self.default_level = default_level
        self.extractor = SummaryExtractor()

        # Cache for fetched full items
        self._item_cache: Dict[str, Any] = {}

    def retrieve_summaries(
        self,
        query: str,
        agent: str,
        project_id: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
        level: Optional[DisclosureLevel] = None,
        **kwargs,
    ) -> ProgressiveSlice:
        """
        Retrieve memory summaries (not full content).

        Returns compact summaries that fit more items in context.
        Use get_full_item() to fetch complete details when needed.
        """
        level = level or self.default_level

        # Get full results from engine
        raw_slice = self.engine.retrieve(
            query=query,
            agent=agent,
            project_id=project_id,
            user_id=user_id,
            top_k=top_k,
            **kwargs,
        )

        # Extract summaries
        heuristic_summaries = [
            self.extractor.extract_heuristic_summary(h, level)
            for h in raw_slice.heuristics
        ]
        outcome_summaries = [
            self.extractor.extract_outcome_summary(o, level)
            for o in raw_slice.outcomes
        ]
        knowledge_summaries = [
            self.extractor.extract_knowledge_summary(k, level)
            for k in raw_slice.domain_knowledge
        ]
        anti_pattern_summaries = [
            self.extractor.extract_anti_pattern_summary(ap, level)
            for ap in raw_slice.anti_patterns
        ]
        preference_summaries = [
            self.extractor.extract_preference_summary(p, level)
            for p in raw_slice.preferences
        ]

        # Calculate totals
        all_summaries = (
            heuristic_summaries
            + outcome_summaries
            + knowledge_summaries
            + anti_pattern_summaries
            + preference_summaries
        )
        total_tokens = sum(s.estimated_tokens for s in all_summaries)

        # Cache full items for lazy loading
        for h in raw_slice.heuristics:
            self._item_cache[h.id] = h
        for o in raw_slice.outcomes:
            self._item_cache[o.id] = o
        for k in raw_slice.domain_knowledge:
            self._item_cache[k.id] = k
        for ap in raw_slice.anti_patterns:
            self._item_cache[ap.id] = ap
        for p in raw_slice.preferences:
            self._item_cache[p.id] = p

        return ProgressiveSlice(
            heuristic_summaries=heuristic_summaries,
            outcome_summaries=outcome_summaries,
            knowledge_summaries=knowledge_summaries,
            anti_pattern_summaries=anti_pattern_summaries,
            preference_summaries=preference_summaries,
            query=query,
            agent=agent,
            total_available=raw_slice.total_items,
            summaries_included=len(all_summaries),
            estimated_summary_tokens=total_tokens,
        )

    def get_full_item(
        self,
        memory_id: str,
        memory_type: str,
    ) -> Optional[Any]:
        """
        Get full details for a specific memory item.

        This is the "fetch on demand" part of progressive disclosure.
        """
        # Check cache first
        if memory_id in self._item_cache:
            logger.debug(f"Cache hit for {memory_type}:{memory_id}")
            return self._item_cache[memory_id]

        # Fetch from storage
        logger.debug(f"Fetching {memory_type}:{memory_id} from storage")

        item = None
        if memory_type == "heuristic":
            item = self.storage.get_heuristic_by_id(memory_id)
        elif memory_type == "outcome":
            item = self.storage.get_outcome_by_id(memory_id)
        elif memory_type == "domain_knowledge":
            item = self.storage.get_domain_knowledge_by_id(memory_id)
        elif memory_type == "anti_pattern":
            item = self.storage.get_anti_pattern_by_id(memory_id)
        elif memory_type == "preference":
            item = self.storage.get_preference_by_id(memory_id)

        if item:
            self._item_cache[memory_id] = item

        return item

    def get_multiple_full_items(
        self,
        memory_ids: List[str],
        memory_type: str,
    ) -> List[Any]:
        """Get full details for multiple items."""
        return [
            item
            for item in (self.get_full_item(mid, memory_type) for mid in memory_ids)
            if item is not None
        ]

    def clear_cache(self) -> None:
        """Clear the item cache."""
        self._item_cache.clear()

    def format_summaries_for_context(
        self,
        progressive_slice: ProgressiveSlice,
        include_fetch_hint: bool = True,
    ) -> str:
        """
        Format summaries for inclusion in agent context.

        Returns a compact string representation suitable for prompts.
        """
        lines = []

        if progressive_slice.heuristic_summaries:
            lines.append("## Relevant Patterns")
            for s in progressive_slice.heuristic_summaries:
                lines.append(f"- [{s.id}] {s.summary} ({s.relevance_hint})")

        if progressive_slice.anti_pattern_summaries:
            lines.append("\n## Warnings")
            for s in progressive_slice.anti_pattern_summaries:
                lines.append(f"- [{s.id}] {s.summary} ({s.relevance_hint})")

        if progressive_slice.outcome_summaries:
            lines.append("\n## Recent Outcomes")
            for s in progressive_slice.outcome_summaries:
                lines.append(f"- [{s.id}] {s.summary}")

        if progressive_slice.preference_summaries:
            lines.append("\n## User Preferences")
            for s in progressive_slice.preference_summaries:
                lines.append(f"- [{s.id}] {s.summary}")

        if progressive_slice.knowledge_summaries:
            lines.append("\n## Domain Knowledge")
            for s in progressive_slice.knowledge_summaries:
                lines.append(f"- [{s.id}] {s.summary}")

        if include_fetch_hint:
            lines.append(
                f"\n_({progressive_slice.summaries_included} summaries shown, "
                f"~{progressive_slice.estimated_summary_tokens} tokens. "
                f"Request full details by ID if needed.)_"
            )

        return "\n".join(lines)
