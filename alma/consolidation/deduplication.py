"""
Heuristic Deduplication Engine - Extracted from consolidation for clarity.

IMPROVEMENTS:
- Separates deduplication logic from consolidation pipeline
- Improves clarity by creating focused, single-responsibility module
- Makes deduplication testable independently
- Reduces complexity of consolidation/pipeline.py
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from alma.types import Heuristic, Outcome

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""

    deduplicated: List[Heuristic | Outcome]
    duplicates_found: int
    merge_operations: int

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Deduplicated {len(self.deduplicated)} items "
            f"({self.duplicates_found} duplicates found, "
            f"{self.merge_operations} merges)"
        )


class DeduplicationEngine:
    """Deduplicates heuristics and outcomes by semantic similarity."""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplication engine.

        Args:
            similarity_threshold: Minimum similarity score to consider items identical
        """
        self.similarity_threshold = similarity_threshold
        logger.info(
            f"DeduplicationEngine initialized with "
            f"threshold={similarity_threshold}"
        )

    def deduplicate(
        self,
        items: List[Heuristic | Outcome],
    ) -> DeduplicationResult:
        """
        Deduplicate items by semantic similarity.

        Args:
            items: Items to deduplicate

        Returns:
            DeduplicationResult with deduplicated items and statistics
        """
        if not items:
            return DeduplicationResult(
                deduplicated=[],
                duplicates_found=0,
                merge_operations=0,
            )

        logger.debug(f"Deduplicating {len(items)} items")

        # Group similar items
        groups = self._group_similar_items(items)

        # Merge groups into representative items
        deduplicated = []
        merges = 0

        for group in groups:
            if len(group) > 1:
                # Multiple similar items - merge them
                merged = self._merge_group(group)
                deduplicated.append(merged)
                merges += len(group) - 1
            else:
                # Single item - keep as is
                deduplicated.append(group[0])

        result = DeduplicationResult(
            deduplicated=deduplicated,
            duplicates_found=len(items) - len(deduplicated),
            merge_operations=merges,
        )

        logger.info(result.summary())
        return result

    def _group_similar_items(
        self,
        items: List[Heuristic | Outcome],
    ) -> List[List[Heuristic | Outcome]]:
        """
        Group items by semantic similarity.

        Returns:
            List of groups, where each group contains similar items
        """
        if not items:
            return []

        groups: List[List[Heuristic | Outcome]] = []

        for item in items:
            # Find best matching group
            best_group = None
            best_similarity = 0.0

            for group in groups:
                group_rep = group[0]  # Use first item as representative
                similarity = self._calculate_similarity(item, group_rep)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_group = group

            # Add to best group if similarity threshold met
            if (
                best_group is not None
                and best_similarity >= self.similarity_threshold
            ):
                best_group.append(item)
            else:
                # Create new group
                groups.append([item])

        return groups

    def _calculate_similarity(
        self,
        item1: Heuristic | Outcome,
        item2: Heuristic | Outcome,
    ) -> float:
        """
        Calculate semantic similarity between items.

        Simple implementation based on text content.
        Could be enhanced with embedding-based similarity.

        Args:
            item1: First item
            item2: Second item

        Returns:
            Similarity score 0.0-1.0
        """
        # Extract text from item
        text1 = self._extract_text(item1)
        text2 = self._extract_text(item2)

        # Simple text similarity (token overlap)
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)

        return overlap / total if total > 0 else 0.0

    def _extract_text(self, item: Heuristic | Outcome) -> str:
        """Extract text content from item for comparison.

        Handles all ALMA memory types:
        - Heuristic: condition + strategy
        - Outcome: task_description + strategy_used
        - DomainKnowledge: domain + fact
        - AntiPattern: pattern + why_bad + better_alternative
        """
        if hasattr(item, "condition") and hasattr(item, "strategy"):
            return f"{item.condition} {item.strategy}"
        elif hasattr(item, "task_description") and hasattr(item, "strategy_used"):
            return f"{item.task_description} {item.strategy_used}"
        elif hasattr(item, "domain") and hasattr(item, "fact"):
            return f"{item.domain} {item.fact}"
        elif hasattr(item, "pattern") and hasattr(item, "why_bad"):
            return f"{item.pattern} {item.why_bad} {getattr(item, 'better_alternative', '')}"
        elif hasattr(item, "title"):
            return getattr(item, "title", "")
        else:
            return str(item)

    def _merge_group(
        self,
        group: List[Heuristic | Outcome],
    ) -> Heuristic | Outcome:
        """
        Merge a group of similar items into a representative.

        Simple strategy: keep first item, copy metadata from others.

        Args:
            group: Group of similar items

        Returns:
            Merged representative item
        """
        if not group:
            raise ValueError("Cannot merge empty group")

        # Use first item as base
        representative = group[0]

        # Could enhance with confidence merging, timestamp updates, etc.
        logger.debug(
            f"Merged {len(group)} items into representative "
            f"(kept first, discarded {len(group) - 1})"
        )

        return representative
