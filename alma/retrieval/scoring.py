"""
ALMA Memory Scoring.

Combines semantic similarity, recency, and success rate for optimal retrieval.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TypeVar, Callable
from dataclasses import dataclass

from alma.types import Heuristic, Outcome, DomainKnowledge, AntiPattern


@dataclass
class ScoringWeights:
    """
    Configurable weights for memory scoring.

    All weights should sum to 1.0 for normalized scores.
    """
    similarity: float = 0.4      # Semantic relevance to query
    recency: float = 0.3         # How recently the memory was validated/used
    success_rate: float = 0.2    # Historical success rate
    confidence: float = 0.1      # Stored confidence score

    def __post_init__(self):
        """Validate weights sum to approximately 1.0."""
        total = self.similarity + self.recency + self.success_rate + self.confidence
        if not (0.99 <= total <= 1.01):
            # Normalize if not summing to 1
            self.similarity /= total
            self.recency /= total
            self.success_rate /= total
            self.confidence /= total


@dataclass
class ScoredItem:
    """A memory item with its computed score."""
    item: Any
    score: float
    similarity_score: float
    recency_score: float
    success_score: float
    confidence_score: float


class MemoryScorer:
    """
    Scores memories based on multiple factors for optimal retrieval.

    Factors:
    - Semantic similarity (from vector search)
    - Recency (newer memories preferred, with decay)
    - Success rate (for heuristics and outcomes)
    - Confidence (stored confidence values)
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        recency_half_life_days: float = 30.0,
    ):
        """
        Initialize scorer.

        Args:
            weights: Scoring weights for each factor
            recency_half_life_days: Days after which recency score is halved
        """
        self.weights = weights or ScoringWeights()
        self.recency_half_life = recency_half_life_days

    def score_heuristics(
        self,
        heuristics: List[Heuristic],
        similarities: Optional[List[float]] = None,
    ) -> List[ScoredItem]:
        """
        Score and rank heuristics.

        Args:
            heuristics: List of heuristics to score
            similarities: Optional pre-computed similarity scores (0-1)

        Returns:
            Sorted list of ScoredItems (highest first)
        """
        if not heuristics:
            return []

        similarities = similarities or [1.0] * len(heuristics)
        scored = []

        for h, sim in zip(heuristics, similarities):
            recency = self._compute_recency_score(h.last_validated)
            success = h.success_rate
            confidence = h.confidence

            total = (
                self.weights.similarity * sim +
                self.weights.recency * recency +
                self.weights.success_rate * success +
                self.weights.confidence * confidence
            )

            scored.append(ScoredItem(
                item=h,
                score=total,
                similarity_score=sim,
                recency_score=recency,
                success_score=success,
                confidence_score=confidence,
            ))

        return sorted(scored, key=lambda x: -x.score)

    def score_outcomes(
        self,
        outcomes: List[Outcome],
        similarities: Optional[List[float]] = None,
    ) -> List[ScoredItem]:
        """
        Score and rank outcomes.

        Successful outcomes score higher, but failures are still included
        for learning purposes.

        Args:
            outcomes: List of outcomes to score
            similarities: Optional pre-computed similarity scores (0-1)

        Returns:
            Sorted list of ScoredItems (highest first)
        """
        if not outcomes:
            return []

        similarities = similarities or [1.0] * len(outcomes)
        scored = []

        for o, sim in zip(outcomes, similarities):
            recency = self._compute_recency_score(o.timestamp)
            # Success gets full score, failure gets partial (still useful to learn from)
            success = 1.0 if o.success else 0.3
            # Outcomes don't have stored confidence, default to 1.0
            confidence = 1.0

            total = (
                self.weights.similarity * sim +
                self.weights.recency * recency +
                self.weights.success_rate * success +
                self.weights.confidence * confidence
            )

            scored.append(ScoredItem(
                item=o,
                score=total,
                similarity_score=sim,
                recency_score=recency,
                success_score=success,
                confidence_score=confidence,
            ))

        return sorted(scored, key=lambda x: -x.score)

    def score_domain_knowledge(
        self,
        knowledge: List[DomainKnowledge],
        similarities: Optional[List[float]] = None,
    ) -> List[ScoredItem]:
        """
        Score and rank domain knowledge.

        Args:
            knowledge: List of domain knowledge to score
            similarities: Optional pre-computed similarity scores (0-1)

        Returns:
            Sorted list of ScoredItems (highest first)
        """
        if not knowledge:
            return []

        similarities = similarities or [1.0] * len(knowledge)
        scored = []

        for dk, sim in zip(knowledge, similarities):
            recency = self._compute_recency_score(dk.last_verified)
            # Knowledge doesn't have success rate, use 1.0
            success = 1.0
            confidence = dk.confidence

            total = (
                self.weights.similarity * sim +
                self.weights.recency * recency +
                self.weights.success_rate * success +
                self.weights.confidence * confidence
            )

            scored.append(ScoredItem(
                item=dk,
                score=total,
                similarity_score=sim,
                recency_score=recency,
                success_score=success,
                confidence_score=confidence,
            ))

        return sorted(scored, key=lambda x: -x.score)

    def score_anti_patterns(
        self,
        anti_patterns: List[AntiPattern],
        similarities: Optional[List[float]] = None,
    ) -> List[ScoredItem]:
        """
        Score and rank anti-patterns.

        Anti-patterns that were seen recently are more relevant.

        Args:
            anti_patterns: List of anti-patterns to score
            similarities: Optional pre-computed similarity scores (0-1)

        Returns:
            Sorted list of ScoredItems (highest first)
        """
        if not anti_patterns:
            return []

        similarities = similarities or [1.0] * len(anti_patterns)
        scored = []

        for ap, sim in zip(anti_patterns, similarities):
            recency = self._compute_recency_score(ap.last_seen)
            # More occurrences = more important to avoid
            # Normalize occurrence count (cap at 10 for scoring)
            success = min(ap.occurrence_count / 10.0, 1.0)
            confidence = 1.0

            total = (
                self.weights.similarity * sim +
                self.weights.recency * recency +
                self.weights.success_rate * success +
                self.weights.confidence * confidence
            )

            scored.append(ScoredItem(
                item=ap,
                score=total,
                similarity_score=sim,
                recency_score=recency,
                success_score=success,
                confidence_score=confidence,
            ))

        return sorted(scored, key=lambda x: -x.score)

    def _compute_recency_score(self, timestamp: datetime) -> float:
        """
        Compute recency score using exponential decay.

        Score = 0.5 ^ (days_ago / half_life)

        Args:
            timestamp: When the memory was last validated/used

        Returns:
            Score between 0 and 1 (1 = now, decays over time)
        """
        now = datetime.now(timezone.utc)

        # Handle naive datetimes
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        delta = now - timestamp
        days_ago = delta.total_seconds() / (24 * 60 * 60)

        # Exponential decay: score halves every half_life days
        return math.pow(0.5, days_ago / self.recency_half_life)

    def apply_score_threshold(
        self,
        scored_items: List[ScoredItem],
        min_score: float = 0.2,
    ) -> List[ScoredItem]:
        """
        Filter out items below a minimum score threshold.

        Args:
            scored_items: List of scored items
            min_score: Minimum score to keep (0-1)

        Returns:
            Filtered list
        """
        return [item for item in scored_items if item.score >= min_score]


def compute_composite_score(
    similarity: float,
    recency_days: float,
    success_rate: float,
    confidence: float,
    weights: Optional[ScoringWeights] = None,
    recency_half_life: float = 30.0,
) -> float:
    """
    Convenience function to compute a single composite score.

    Args:
        similarity: Semantic similarity (0-1)
        recency_days: Days since last validation
        success_rate: Historical success rate (0-1)
        confidence: Stored confidence (0-1)
        weights: Optional scoring weights
        recency_half_life: Days after which recency score halves

    Returns:
        Composite score (0-1)
    """
    weights = weights or ScoringWeights()

    recency_score = math.pow(0.5, recency_days / recency_half_life)

    return (
        weights.similarity * similarity +
        weights.recency * recency_score +
        weights.success_rate * success_rate +
        weights.confidence * confidence
    )
