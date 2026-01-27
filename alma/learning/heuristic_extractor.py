"""
ALMA Heuristic Extraction.

Analyzes outcomes to identify patterns and create heuristics.
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alma.storage.base import StorageBackend
from alma.types import Heuristic, MemoryScope, Outcome

logger = logging.getLogger(__name__)


@dataclass
class PatternCandidate:
    """A potential pattern for heuristic creation."""
    task_type: str
    strategy: str
    occurrence_count: int
    success_count: int
    failure_count: int
    outcomes: List[Outcome] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.occurrence_count == 0:
            return 0.0
        return self.success_count / self.occurrence_count

    @property
    def confidence(self) -> float:
        """
        Calculate confidence based on success rate and sample size.

        Confidence is lower when sample size is small (uncertainty).
        """
        if self.occurrence_count == 0:
            return 0.0

        base_confidence = self.success_rate

        # Apply sample size penalty (Bayesian-inspired)
        # More samples = higher confidence, max confidence at 20+ samples
        sample_factor = min(self.occurrence_count / 20.0, 1.0)

        return base_confidence * (0.5 + 0.5 * sample_factor)


@dataclass
class ExtractionResult:
    """Result of heuristic extraction."""
    heuristics_created: int = 0
    heuristics_updated: int = 0
    patterns_analyzed: int = 0
    patterns_rejected: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heuristics_created": self.heuristics_created,
            "heuristics_updated": self.heuristics_updated,
            "patterns_analyzed": self.patterns_analyzed,
            "patterns_rejected": self.patterns_rejected,
            "rejected_reasons": self.rejected_reasons,
        }


class HeuristicExtractor:
    """
    Extracts heuristics from outcome patterns.

    Analyzes historical outcomes to identify successful strategies
    and creates heuristics when patterns are validated.
    """

    def __init__(
        self,
        storage: StorageBackend,
        scopes: Dict[str, MemoryScope],
        min_occurrences: int = 3,
        min_confidence: float = 0.5,
        strategy_similarity_threshold: float = 0.5,
    ):
        """
        Initialize extractor.

        Args:
            storage: Storage backend
            scopes: Agent scope definitions
            min_occurrences: Minimum outcomes before creating heuristic
            min_confidence: Minimum confidence to create heuristic
            strategy_similarity_threshold: How similar strategies must be to group
        """
        self.storage = storage
        self.scopes = scopes
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self.similarity_threshold = strategy_similarity_threshold

    def extract(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract heuristics from all outcomes.

        Args:
            project_id: Project to analyze
            agent: Specific agent or None for all

        Returns:
            ExtractionResult with summary
        """
        result = ExtractionResult()

        # Get all outcomes
        outcomes = self.storage.get_outcomes(
            project_id=project_id,
            agent=agent,
            top_k=10000,
            success_only=False,
        )

        if not outcomes:
            logger.info("No outcomes to analyze")
            return result

        # Group outcomes by agent and task type
        grouped = self._group_outcomes(outcomes)

        for (ag, _task_type), type_outcomes in grouped.items():
            # Find patterns within this group
            patterns = self._identify_patterns(type_outcomes)
            result.patterns_analyzed += len(patterns)

            for pattern in patterns:
                created, reason = self._maybe_create_heuristic(
                    agent=ag,
                    project_id=project_id,
                    pattern=pattern,
                )

                if created:
                    result.heuristics_created += 1
                else:
                    result.patterns_rejected += 1
                    result.rejected_reasons[reason] = (
                        result.rejected_reasons.get(reason, 0) + 1
                    )

        logger.info(
            f"Extraction complete: {result.heuristics_created} heuristics created, "
            f"{result.patterns_rejected} patterns rejected"
        )

        return result

    def _group_outcomes(
        self,
        outcomes: List[Outcome],
    ) -> Dict[Tuple[str, str], List[Outcome]]:
        """Group outcomes by agent and task type."""
        grouped: Dict[Tuple[str, str], List[Outcome]] = defaultdict(list)
        for outcome in outcomes:
            key = (outcome.agent, outcome.task_type)
            grouped[key].append(outcome)
        return grouped

    def _identify_patterns(
        self,
        outcomes: List[Outcome],
    ) -> List[PatternCandidate]:
        """
        Identify patterns in outcomes by grouping similar strategies.

        Uses fuzzy matching to group strategies that are similar.
        """
        # Group by similar strategies
        strategy_groups: Dict[str, List[Outcome]] = defaultdict(list)

        for outcome in outcomes:
            # Find existing group or create new one
            matched = False
            for canonical in list(strategy_groups.keys()):
                if self._strategies_similar(outcome.strategy_used, canonical):
                    strategy_groups[canonical].append(outcome)
                    matched = True
                    break

            if not matched:
                # Create new group
                strategy_groups[outcome.strategy_used].append(outcome)

        # Convert to PatternCandidates
        patterns = []
        for strategy, group_outcomes in strategy_groups.items():
            success_count = sum(1 for o in group_outcomes if o.success)
            patterns.append(PatternCandidate(
                task_type=group_outcomes[0].task_type,
                strategy=strategy,
                occurrence_count=len(group_outcomes),
                success_count=success_count,
                failure_count=len(group_outcomes) - success_count,
                outcomes=group_outcomes,
            ))

        return patterns

    def _maybe_create_heuristic(
        self,
        agent: str,
        project_id: str,
        pattern: PatternCandidate,
    ) -> Tuple[bool, str]:
        """
        Create a heuristic if the pattern meets criteria.

        Returns:
            Tuple of (created: bool, reason: str)
        """
        # Check minimum occurrences
        scope = self.scopes.get(agent)
        min_occ = self.min_occurrences
        if scope:
            min_occ = scope.min_occurrences_for_heuristic

        if pattern.occurrence_count < min_occ:
            return False, f"insufficient_occurrences_{pattern.occurrence_count}"

        # Check confidence
        if pattern.confidence < self.min_confidence:
            return False, f"low_confidence_{pattern.confidence:.2f}"

        # Check if heuristic already exists
        existing = self._find_existing_heuristic(
            agent=agent,
            project_id=project_id,
            task_type=pattern.task_type,
            strategy=pattern.strategy,
        )

        if existing:
            # Update existing heuristic
            self._update_heuristic(existing, pattern)
            return True, "updated"

        # Create new heuristic
        heuristic = Heuristic(
            id=f"heur_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            condition=f"task type: {pattern.task_type}",
            strategy=pattern.strategy,
            confidence=pattern.confidence,
            occurrence_count=pattern.occurrence_count,
            success_count=pattern.success_count,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        self.storage.save_heuristic(heuristic)
        logger.info(
            f"Created heuristic for {agent}: {pattern.strategy[:50]}... "
            f"(confidence: {pattern.confidence:.0%})"
        )

        return True, "created"

    def _find_existing_heuristic(
        self,
        agent: str,
        project_id: str,
        task_type: str,
        strategy: str,
    ) -> Optional[Heuristic]:
        """Find an existing heuristic that matches this pattern."""
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            top_k=100,
            min_confidence=0.0,
        )

        for h in heuristics:
            if (task_type in h.condition and
                    self._strategies_similar(h.strategy, strategy)):
                return h

        return None

    def _update_heuristic(
        self,
        heuristic: Heuristic,
        pattern: PatternCandidate,
    ):
        """Update an existing heuristic with new data."""
        # Merge counts
        heuristic.occurrence_count = max(
            heuristic.occurrence_count,
            pattern.occurrence_count
        )
        heuristic.success_count = max(
            heuristic.success_count,
            pattern.success_count
        )

        # Update confidence
        heuristic.confidence = pattern.confidence
        heuristic.last_validated = datetime.now(timezone.utc)

        self.storage.save_heuristic(heuristic)
        logger.debug(f"Updated heuristic {heuristic.id}")

    def _strategies_similar(self, s1: str, s2: str) -> bool:
        """
        Check if two strategies are similar enough to be grouped.

        Uses word overlap with normalization.
        """
        # Normalize strategies
        words1 = set(self._normalize_strategy(s1))
        words2 = set(self._normalize_strategy(s2))

        if not words1 or not words2:
            return s1.lower() == s2.lower()

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0

        return similarity >= self.similarity_threshold

    def _normalize_strategy(self, strategy: str) -> List[str]:
        """Normalize strategy text for comparison."""
        # Remove common stop words and normalize
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "then", "first", "next",
        }

        words = strategy.lower().replace(",", " ").replace(".", " ").split()
        return [w for w in words if w not in stop_words and len(w) > 2]


def extract_heuristics_from_outcome(
    outcome: Outcome,
    existing_heuristics: List[Heuristic],
    min_confidence: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to check if an outcome contributes to a heuristic.

    Returns update details if the outcome should update a heuristic.
    """
    for h in existing_heuristics:
        # Check if this outcome matches an existing heuristic
        if h.agent == outcome.agent and outcome.task_type in h.condition:
            return {
                "heuristic_id": h.id,
                "action": "validate" if outcome.success else "invalidate",
                "current_confidence": h.confidence,
            }

    return None
