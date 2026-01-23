"""
ALMA Forgetting Mechanism.

Implements intelligent memory pruning to prevent bloat and maintain relevance.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from alma.types import Heuristic, Outcome, DomainKnowledge, AntiPattern
from alma.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class PruneReason(Enum):
    """Reason for pruning a memory item."""
    STALE = "stale"                    # Too old without validation
    LOW_CONFIDENCE = "low_confidence"  # Below confidence threshold
    LOW_SUCCESS_RATE = "low_success"   # Too many failures
    SUPERSEDED = "superseded"          # Replaced by better heuristic
    DUPLICATE = "duplicate"            # Duplicate of another item
    QUOTA_EXCEEDED = "quota"           # Agent memory quota exceeded


@dataclass
class PruneResult:
    """Result of a prune operation."""
    reason: PruneReason
    item_type: str
    item_id: str
    agent: str
    project_id: str
    details: str = ""


@dataclass
class PruneSummary:
    """Summary of a complete prune operation."""
    outcomes_pruned: int = 0
    heuristics_pruned: int = 0
    knowledge_pruned: int = 0
    anti_patterns_pruned: int = 0
    total_pruned: int = 0
    pruned_items: List[PruneResult] = field(default_factory=list)
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcomes_pruned": self.outcomes_pruned,
            "heuristics_pruned": self.heuristics_pruned,
            "knowledge_pruned": self.knowledge_pruned,
            "anti_patterns_pruned": self.anti_patterns_pruned,
            "total_pruned": self.total_pruned,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class PrunePolicy:
    """
    Configuration for memory pruning behavior.

    Defines thresholds and quotas for different memory types.
    """
    # Age-based pruning
    outcome_max_age_days: int = 90
    knowledge_max_age_days: int = 180
    anti_pattern_max_age_days: int = 365

    # Confidence thresholds
    heuristic_min_confidence: float = 0.3
    knowledge_min_confidence: float = 0.5

    # Success rate thresholds
    heuristic_min_success_rate: float = 0.4
    min_occurrences_before_prune: int = 5  # Don't prune until enough data

    # Quota limits (per agent)
    max_heuristics_per_agent: int = 100
    max_outcomes_per_agent: int = 500
    max_knowledge_per_agent: int = 200
    max_anti_patterns_per_agent: int = 50

    # Staleness (time since last validation)
    heuristic_stale_days: int = 60
    knowledge_stale_days: int = 90


class ForgettingEngine:
    """
    Manages memory pruning and forgetting.

    Implements multiple strategies:
    - Age-based decay (old memories are pruned)
    - Confidence-based pruning (low confidence items removed)
    - Success-rate based pruning (unsuccessful patterns removed)
    - Quota enforcement (prevents memory bloat)
    - Staleness detection (unvalidated memories removed)
    """

    def __init__(
        self,
        storage: StorageBackend,
        policy: Optional[PrunePolicy] = None,
    ):
        """
        Initialize forgetting engine.

        Args:
            storage: Storage backend to prune
            policy: Pruning policy configuration
        """
        self.storage = storage
        self.policy = policy or PrunePolicy()

    def prune(
        self,
        project_id: str,
        agent: Optional[str] = None,
        dry_run: bool = False,
    ) -> PruneSummary:
        """
        Run a complete prune operation.

        Args:
            project_id: Project to prune
            agent: Specific agent or None for all
            dry_run: If True, only report what would be pruned

        Returns:
            PruneSummary with details
        """
        import time
        start_time = time.time()

        summary = PruneSummary()

        # Prune each memory type
        summary.outcomes_pruned = self._prune_stale_outcomes(
            project_id, agent, dry_run, summary.pruned_items
        )

        summary.heuristics_pruned = self._prune_heuristics(
            project_id, agent, dry_run, summary.pruned_items
        )

        summary.knowledge_pruned = self._prune_domain_knowledge(
            project_id, agent, dry_run, summary.pruned_items
        )

        summary.anti_patterns_pruned = self._prune_anti_patterns(
            project_id, agent, dry_run, summary.pruned_items
        )

        # Enforce quotas
        quota_pruned = self._enforce_quotas(
            project_id, agent, dry_run, summary.pruned_items
        )
        summary.heuristics_pruned += quota_pruned.get("heuristics", 0)
        summary.outcomes_pruned += quota_pruned.get("outcomes", 0)

        summary.total_pruned = (
            summary.outcomes_pruned +
            summary.heuristics_pruned +
            summary.knowledge_pruned +
            summary.anti_patterns_pruned
        )

        summary.execution_time_ms = int((time.time() - start_time) * 1000)

        action = "Would prune" if dry_run else "Pruned"
        logger.info(
            f"{action} {summary.total_pruned} items for project={project_id}, "
            f"agent={agent or 'all'}"
        )

        return summary

    def _prune_stale_outcomes(
        self,
        project_id: str,
        agent: Optional[str],
        dry_run: bool,
        results: List[PruneResult],
    ) -> int:
        """Prune outcomes older than max age."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.policy.outcome_max_age_days
        )

        if dry_run:
            # Get count of outcomes to prune
            outcomes = self.storage.get_outcomes(
                project_id=project_id,
                agent=agent,
                top_k=10000,
                success_only=False,
            )
            count = sum(1 for o in outcomes if o.timestamp < cutoff)
            for o in outcomes:
                if o.timestamp < cutoff:
                    results.append(PruneResult(
                        reason=PruneReason.STALE,
                        item_type="outcome",
                        item_id=o.id,
                        agent=o.agent,
                        project_id=project_id,
                        details=f"Older than {self.policy.outcome_max_age_days} days",
                    ))
            return count
        else:
            return self.storage.delete_outcomes_older_than(
                project_id=project_id,
                older_than=cutoff,
                agent=agent,
            )

    def _prune_heuristics(
        self,
        project_id: str,
        agent: Optional[str],
        dry_run: bool,
        results: List[PruneResult],
    ) -> int:
        """Prune heuristics based on confidence and success rate."""
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            top_k=10000,
            min_confidence=0.0,
        )

        to_delete = []
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(days=self.policy.heuristic_stale_days)

        for h in heuristics:
            reason = None
            details = ""

            # Check confidence
            if h.confidence < self.policy.heuristic_min_confidence:
                reason = PruneReason.LOW_CONFIDENCE
                details = f"Confidence {h.confidence:.2f} < {self.policy.heuristic_min_confidence}"

            # Check success rate (only if enough occurrences)
            elif (h.occurrence_count >= self.policy.min_occurrences_before_prune and
                  h.success_rate < self.policy.heuristic_min_success_rate):
                reason = PruneReason.LOW_SUCCESS_RATE
                details = f"Success rate {h.success_rate:.2f} < {self.policy.heuristic_min_success_rate}"

            # Check staleness
            elif h.last_validated < stale_cutoff:
                reason = PruneReason.STALE
                details = f"Not validated since {h.last_validated.date()}"

            if reason:
                to_delete.append(h)
                results.append(PruneResult(
                    reason=reason,
                    item_type="heuristic",
                    item_id=h.id,
                    agent=h.agent,
                    project_id=project_id,
                    details=details,
                ))

        if not dry_run:
            for h in to_delete:
                self.storage.delete_heuristic(h.id)

        return len(to_delete)

    def _prune_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str],
        dry_run: bool,
        results: List[PruneResult],
    ) -> int:
        """Prune old or low-confidence domain knowledge."""
        knowledge = self.storage.get_domain_knowledge(
            project_id=project_id,
            agent=agent,
            top_k=10000,
        )

        to_delete = []
        now = datetime.now(timezone.utc)
        age_cutoff = now - timedelta(days=self.policy.knowledge_max_age_days)
        stale_cutoff = now - timedelta(days=self.policy.knowledge_stale_days)

        for dk in knowledge:
            reason = None
            details = ""

            # Check confidence
            if dk.confidence < self.policy.knowledge_min_confidence:
                reason = PruneReason.LOW_CONFIDENCE
                details = f"Confidence {dk.confidence:.2f} < {self.policy.knowledge_min_confidence}"

            # Check age
            elif dk.last_verified < age_cutoff:
                reason = PruneReason.STALE
                details = f"Older than {self.policy.knowledge_max_age_days} days"

            # Check staleness
            elif dk.last_verified < stale_cutoff:
                reason = PruneReason.STALE
                details = f"Not verified since {dk.last_verified.date()}"

            if reason:
                to_delete.append(dk)
                results.append(PruneResult(
                    reason=reason,
                    item_type="domain_knowledge",
                    item_id=dk.id,
                    agent=dk.agent,
                    project_id=project_id,
                    details=details,
                ))

        if not dry_run:
            for dk in to_delete:
                self.storage.delete_domain_knowledge(dk.id)

        return len(to_delete)

    def _prune_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str],
        dry_run: bool,
        results: List[PruneResult],
    ) -> int:
        """Prune old anti-patterns."""
        anti_patterns = self.storage.get_anti_patterns(
            project_id=project_id,
            agent=agent,
            top_k=10000,
        )

        to_delete = []
        now = datetime.now(timezone.utc)
        age_cutoff = now - timedelta(days=self.policy.anti_pattern_max_age_days)

        for ap in anti_patterns:
            if ap.last_seen < age_cutoff:
                to_delete.append(ap)
                results.append(PruneResult(
                    reason=PruneReason.STALE,
                    item_type="anti_pattern",
                    item_id=ap.id,
                    agent=ap.agent,
                    project_id=project_id,
                    details=f"Not seen since {ap.last_seen.date()}",
                ))

        if not dry_run:
            for ap in to_delete:
                self.storage.delete_anti_pattern(ap.id)

        return len(to_delete)

    def _enforce_quotas(
        self,
        project_id: str,
        agent: Optional[str],
        dry_run: bool,
        results: List[PruneResult],
    ) -> Dict[str, int]:
        """Enforce per-agent memory quotas."""
        pruned = {"heuristics": 0, "outcomes": 0}

        if agent:
            agents = [agent]
        else:
            # Get all agents with data
            stats = self.storage.get_stats(project_id=project_id)
            agents = stats.get("agents", [])

        for ag in agents:
            # Check heuristic quota
            heuristics = self.storage.get_heuristics(
                project_id=project_id,
                agent=ag,
                top_k=self.policy.max_heuristics_per_agent + 100,
                min_confidence=0.0,
            )

            if len(heuristics) > self.policy.max_heuristics_per_agent:
                # Sort by confidence (lowest first)
                sorted_h = sorted(heuristics, key=lambda x: x.confidence)
                to_remove = len(heuristics) - self.policy.max_heuristics_per_agent

                for h in sorted_h[:to_remove]:
                    results.append(PruneResult(
                        reason=PruneReason.QUOTA_EXCEEDED,
                        item_type="heuristic",
                        item_id=h.id,
                        agent=ag,
                        project_id=project_id,
                        details=f"Exceeded quota of {self.policy.max_heuristics_per_agent}",
                    ))
                    if not dry_run:
                        self.storage.delete_heuristic(h.id)
                    pruned["heuristics"] += 1

            # Check outcome quota
            outcomes = self.storage.get_outcomes(
                project_id=project_id,
                agent=ag,
                top_k=self.policy.max_outcomes_per_agent + 100,
                success_only=False,
            )

            if len(outcomes) > self.policy.max_outcomes_per_agent:
                # Sort by timestamp (oldest first)
                sorted_o = sorted(outcomes, key=lambda x: x.timestamp)
                to_remove = len(outcomes) - self.policy.max_outcomes_per_agent

                for o in sorted_o[:to_remove]:
                    results.append(PruneResult(
                        reason=PruneReason.QUOTA_EXCEEDED,
                        item_type="outcome",
                        item_id=o.id,
                        agent=ag,
                        project_id=project_id,
                        details=f"Exceeded quota of {self.policy.max_outcomes_per_agent}",
                    ))
                    if not dry_run:
                        self.storage.delete_outcome(o.id)
                    pruned["outcomes"] += 1

        return pruned

    def compute_decay_score(
        self,
        item_age_days: float,
        confidence: float,
        success_rate: float,
        occurrence_count: int,
    ) -> float:
        """
        Compute a decay score for an item (lower = more likely to forget).

        Factors:
        - Recency (newer = higher)
        - Confidence (higher = higher)
        - Success rate (higher = higher)
        - Validation frequency (more = higher)

        Returns:
            Score between 0 and 1
        """
        # Age decay (half-life of 30 days)
        age_score = 0.5 ** (item_age_days / 30.0)

        # Normalize occurrence count (cap at 20)
        occurrence_score = min(occurrence_count / 20.0, 1.0)

        # Weighted combination
        return (
            0.3 * age_score +
            0.3 * confidence +
            0.2 * success_rate +
            0.2 * occurrence_score
        )

    def identify_candidates(
        self,
        project_id: str,
        agent: Optional[str] = None,
        max_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Identify memory items that are candidates for pruning.

        Returns items with lowest decay scores.

        Args:
            project_id: Project to analyze
            agent: Specific agent or None for all
            max_candidates: Maximum candidates to return

        Returns:
            List of candidate items with scores
        """
        candidates = []
        now = datetime.now(timezone.utc)

        # Analyze heuristics
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            top_k=1000,
            min_confidence=0.0,
        )

        for h in heuristics:
            age_days = (now - h.created_at).total_seconds() / (24 * 60 * 60)
            score = self.compute_decay_score(
                item_age_days=age_days,
                confidence=h.confidence,
                success_rate=h.success_rate,
                occurrence_count=h.occurrence_count,
            )
            candidates.append({
                "type": "heuristic",
                "id": h.id,
                "agent": h.agent,
                "score": score,
                "age_days": int(age_days),
                "confidence": h.confidence,
                "summary": h.strategy[:50],
            })

        # Sort by score (lowest first = best candidates for pruning)
        candidates.sort(key=lambda x: x["score"])

        return candidates[:max_candidates]
