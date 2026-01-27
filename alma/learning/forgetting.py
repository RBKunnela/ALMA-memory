"""
ALMA Forgetting Mechanism.

Implements intelligent memory pruning to prevent bloat and maintain relevance.

Features:
- Confidence decay over time (exponential, linear, step functions)
- Staleness detection based on last_validated timestamps
- Automated cleanup job scheduling
- Memory growth monitoring and alerting
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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


# ==================== DECAY FUNCTIONS ====================


class DecayFunction(ABC):
    """Abstract base class for confidence decay functions."""

    @abstractmethod
    def compute_decay(self, days_since_validation: float) -> float:
        """
        Compute decay multiplier for a given time since validation.

        Args:
            days_since_validation: Days since last validation

        Returns:
            Multiplier between 0 and 1 to apply to confidence
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this decay function."""
        pass


class ExponentialDecay(DecayFunction):
    """
    Exponential decay with configurable half-life.

    Confidence = original * 0.5^(days/half_life)
    """

    def __init__(self, half_life_days: float = 30.0):
        """
        Initialize exponential decay.

        Args:
            half_life_days: Days until confidence halves
        """
        self.half_life_days = half_life_days

    def compute_decay(self, days_since_validation: float) -> float:
        """Compute exponential decay multiplier."""
        return 0.5 ** (days_since_validation / self.half_life_days)

    def get_name(self) -> str:
        return f"exponential(half_life={self.half_life_days}d)"


class LinearDecay(DecayFunction):
    """
    Linear decay to zero over a specified period.

    Confidence decreases linearly from 1 to min_value over decay_period.
    """

    def __init__(
        self,
        decay_period_days: float = 90.0,
        min_value: float = 0.1,
    ):
        """
        Initialize linear decay.

        Args:
            decay_period_days: Days until confidence reaches min_value
            min_value: Minimum confidence value (floor)
        """
        self.decay_period_days = decay_period_days
        self.min_value = min_value

    def compute_decay(self, days_since_validation: float) -> float:
        """Compute linear decay multiplier."""
        decay = 1.0 - (days_since_validation / self.decay_period_days) * (1.0 - self.min_value)
        return max(self.min_value, decay)

    def get_name(self) -> str:
        return f"linear(period={self.decay_period_days}d, min={self.min_value})"


class StepDecay(DecayFunction):
    """
    Step-wise decay with configurable thresholds.

    Confidence drops at specific day thresholds.
    """

    def __init__(
        self,
        steps: Optional[List[tuple]] = None,
    ):
        """
        Initialize step decay.

        Args:
            steps: List of (days, multiplier) tuples, sorted by days ascending
                   Default: [(30, 0.9), (60, 0.7), (90, 0.5), (180, 0.3)]
        """
        self.steps = steps or [
            (30, 0.9),
            (60, 0.7),
            (90, 0.5),
            (180, 0.3),
        ]
        # Ensure sorted
        self.steps = sorted(self.steps, key=lambda x: x[0])

    def compute_decay(self, days_since_validation: float) -> float:
        """Compute step decay multiplier."""
        multiplier = 1.0
        for threshold_days, step_multiplier in self.steps:
            if days_since_validation >= threshold_days:
                multiplier = step_multiplier
            else:
                break
        return multiplier

    def get_name(self) -> str:
        return f"step({len(self.steps)} steps)"


class NoDecay(DecayFunction):
    """No decay - confidence remains constant."""

    def compute_decay(self, days_since_validation: float) -> float:
        return 1.0

    def get_name(self) -> str:
        return "none"


# ==================== CONFIDENCE DECAYER ====================


@dataclass
class DecayResult:
    """Result of applying confidence decay."""
    items_processed: int = 0
    items_updated: int = 0
    items_pruned: int = 0
    avg_decay_applied: float = 0.0
    execution_time_ms: int = 0


class ConfidenceDecayer:
    """
    Applies confidence decay to memories based on time since validation.

    Unlike pruning (which removes items), decay reduces confidence over time,
    making items less likely to be retrieved while preserving them for potential
    revalidation.
    """

    def __init__(
        self,
        storage: StorageBackend,
        decay_function: Optional[DecayFunction] = None,
        prune_below_confidence: float = 0.1,
    ):
        """
        Initialize confidence decayer.

        Args:
            storage: Storage backend to update
            decay_function: Function to compute decay (default: ExponentialDecay)
            prune_below_confidence: Auto-prune items that decay below this threshold
        """
        self.storage = storage
        self.decay_function = decay_function or ExponentialDecay(half_life_days=30.0)
        self.prune_below_confidence = prune_below_confidence

    def apply_decay(
        self,
        project_id: str,
        agent: Optional[str] = None,
        dry_run: bool = False,
    ) -> DecayResult:
        """
        Apply confidence decay to all eligible memories.

        Args:
            project_id: Project to process
            agent: Specific agent or None for all
            dry_run: If True, calculate but don't update

        Returns:
            DecayResult with statistics
        """
        start_time = time.time()
        result = DecayResult()
        now = datetime.now(timezone.utc)
        total_decay = 0.0

        # Process heuristics
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            top_k=10000,
            min_confidence=0.0,
        )

        for h in heuristics:
            result.items_processed += 1
            days_since = (now - h.last_validated).total_seconds() / (24 * 60 * 60)
            decay_multiplier = self.decay_function.compute_decay(days_since)

            new_confidence = h.confidence * decay_multiplier
            total_decay += (1.0 - decay_multiplier)

            if new_confidence != h.confidence:
                if new_confidence < self.prune_below_confidence:
                    # Below threshold - prune
                    if not dry_run:
                        self.storage.delete_heuristic(h.id)
                    result.items_pruned += 1
                else:
                    # Update confidence
                    if not dry_run:
                        self.storage.update_heuristic_confidence(h.id, new_confidence)
                    result.items_updated += 1

        # Process domain knowledge
        knowledge = self.storage.get_domain_knowledge(
            project_id=project_id,
            agent=agent,
            top_k=10000,
        )

        for dk in knowledge:
            result.items_processed += 1
            days_since = (now - dk.last_verified).total_seconds() / (24 * 60 * 60)
            decay_multiplier = self.decay_function.compute_decay(days_since)

            new_confidence = dk.confidence * decay_multiplier
            total_decay += (1.0 - decay_multiplier)

            if new_confidence != dk.confidence:
                if new_confidence < self.prune_below_confidence:
                    if not dry_run:
                        self.storage.delete_domain_knowledge(dk.id)
                    result.items_pruned += 1
                else:
                    if not dry_run:
                        self.storage.update_knowledge_confidence(dk.id, new_confidence)
                    result.items_updated += 1

        result.execution_time_ms = int((time.time() - start_time) * 1000)
        if result.items_processed > 0:
            result.avg_decay_applied = total_decay / result.items_processed

        action = "Would apply" if dry_run else "Applied"
        logger.info(
            f"{action} decay to {result.items_processed} items: "
            f"{result.items_updated} updated, {result.items_pruned} pruned "
            f"(avg decay: {result.avg_decay_applied:.2%})"
        )

        return result


# ==================== MEMORY HEALTH MONITOR ====================


@dataclass
class MemoryHealthMetrics:
    """Metrics about memory health and growth."""
    total_items: int = 0
    heuristic_count: int = 0
    outcome_count: int = 0
    knowledge_count: int = 0
    anti_pattern_count: int = 0
    avg_heuristic_confidence: float = 0.0
    avg_heuristic_age_days: float = 0.0
    stale_heuristic_count: int = 0
    low_confidence_count: int = 0
    storage_bytes: int = 0
    agents_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "heuristic_count": self.heuristic_count,
            "outcome_count": self.outcome_count,
            "knowledge_count": self.knowledge_count,
            "anti_pattern_count": self.anti_pattern_count,
            "avg_heuristic_confidence": round(self.avg_heuristic_confidence, 3),
            "avg_heuristic_age_days": round(self.avg_heuristic_age_days, 1),
            "stale_heuristic_count": self.stale_heuristic_count,
            "low_confidence_count": self.low_confidence_count,
            "storage_bytes": self.storage_bytes,
            "agents_count": self.agents_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthAlert:
    """An alert about memory health issues."""
    level: str  # "warning", "critical"
    category: str
    message: str
    current_value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HealthThresholds:
    """Thresholds for health monitoring alerts."""
    # Warning thresholds
    max_total_items_warning: int = 5000
    max_stale_percentage_warning: float = 0.3
    min_avg_confidence_warning: float = 0.5
    max_agent_items_warning: int = 500

    # Critical thresholds
    max_total_items_critical: int = 10000
    max_stale_percentage_critical: float = 0.5
    min_avg_confidence_critical: float = 0.3
    max_storage_bytes_critical: int = 100 * 1024 * 1024  # 100MB


class MemoryHealthMonitor:
    """
    Monitors memory health and growth, generating alerts when thresholds exceeded.

    Tracks:
    - Total memory item counts
    - Average confidence levels
    - Staleness ratios
    - Storage size
    - Per-agent statistics
    """

    def __init__(
        self,
        storage: StorageBackend,
        thresholds: Optional[HealthThresholds] = None,
        stale_days: int = 60,
        low_confidence_threshold: float = 0.3,
    ):
        """
        Initialize health monitor.

        Args:
            storage: Storage backend to monitor
            thresholds: Alert thresholds
            stale_days: Days since validation to consider stale
            low_confidence_threshold: Confidence below which to count as low
        """
        self.storage = storage
        self.thresholds = thresholds or HealthThresholds()
        self.stale_days = stale_days
        self.low_confidence_threshold = low_confidence_threshold

        # History for trend analysis
        self._metrics_history: List[MemoryHealthMetrics] = []
        self._max_history = 100

        # Alert callbacks
        self._alert_handlers: List[Callable[[HealthAlert], None]] = []

    def add_alert_handler(self, handler: Callable[[HealthAlert], None]) -> None:
        """Add a callback to be called when alerts are generated."""
        self._alert_handlers.append(handler)

    def collect_metrics(self, project_id: str) -> MemoryHealthMetrics:
        """
        Collect current memory health metrics.

        Args:
            project_id: Project to analyze

        Returns:
            MemoryHealthMetrics snapshot
        """
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(days=self.stale_days)

        metrics = MemoryHealthMetrics()

        # Get all heuristics
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            top_k=10000,
            min_confidence=0.0,
        )
        metrics.heuristic_count = len(heuristics)

        if heuristics:
            total_confidence = 0.0
            total_age = 0.0
            for h in heuristics:
                total_confidence += h.confidence
                age_days = (now - h.created_at).total_seconds() / (24 * 60 * 60)
                total_age += age_days
                if h.last_validated < stale_cutoff:
                    metrics.stale_heuristic_count += 1
                if h.confidence < self.low_confidence_threshold:
                    metrics.low_confidence_count += 1

            metrics.avg_heuristic_confidence = total_confidence / len(heuristics)
            metrics.avg_heuristic_age_days = total_age / len(heuristics)

        # Get other counts
        outcomes = self.storage.get_outcomes(
            project_id=project_id,
            top_k=10000,
            success_only=False,
        )
        metrics.outcome_count = len(outcomes)

        knowledge = self.storage.get_domain_knowledge(
            project_id=project_id,
            top_k=10000,
        )
        metrics.knowledge_count = len(knowledge)

        anti_patterns = self.storage.get_anti_patterns(
            project_id=project_id,
            top_k=10000,
        )
        metrics.anti_pattern_count = len(anti_patterns)

        metrics.total_items = (
            metrics.heuristic_count +
            metrics.outcome_count +
            metrics.knowledge_count +
            metrics.anti_pattern_count
        )

        # Get agent count
        stats = self.storage.get_stats(project_id=project_id)
        metrics.agents_count = len(stats.get("agents", []))

        # Estimate storage size (rough approximation)
        # Average ~500 bytes per item
        metrics.storage_bytes = metrics.total_items * 500

        # Store in history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]

        return metrics

    def check_health(self, project_id: str) -> List[HealthAlert]:
        """
        Check memory health and generate alerts if thresholds exceeded.

        Args:
            project_id: Project to check

        Returns:
            List of health alerts (empty if healthy)
        """
        metrics = self.collect_metrics(project_id)
        alerts: List[HealthAlert] = []
        t = self.thresholds

        # Check total items
        if metrics.total_items >= t.max_total_items_critical:
            alerts.append(HealthAlert(
                level="critical",
                category="total_items",
                message="Memory item count critically high",
                current_value=metrics.total_items,
                threshold=t.max_total_items_critical,
            ))
        elif metrics.total_items >= t.max_total_items_warning:
            alerts.append(HealthAlert(
                level="warning",
                category="total_items",
                message="Memory item count approaching limit",
                current_value=metrics.total_items,
                threshold=t.max_total_items_warning,
            ))

        # Check staleness
        if metrics.heuristic_count > 0:
            stale_percentage = metrics.stale_heuristic_count / metrics.heuristic_count
            if stale_percentage >= t.max_stale_percentage_critical:
                alerts.append(HealthAlert(
                    level="critical",
                    category="staleness",
                    message="Too many stale heuristics",
                    current_value=f"{stale_percentage:.0%}",
                    threshold=f"{t.max_stale_percentage_critical:.0%}",
                ))
            elif stale_percentage >= t.max_stale_percentage_warning:
                alerts.append(HealthAlert(
                    level="warning",
                    category="staleness",
                    message="Many heuristics are stale",
                    current_value=f"{stale_percentage:.0%}",
                    threshold=f"{t.max_stale_percentage_warning:.0%}",
                ))

        # Check average confidence
        if metrics.heuristic_count > 0:
            if metrics.avg_heuristic_confidence < t.min_avg_confidence_critical:
                alerts.append(HealthAlert(
                    level="critical",
                    category="confidence",
                    message="Average heuristic confidence critically low",
                    current_value=f"{metrics.avg_heuristic_confidence:.2f}",
                    threshold=f"{t.min_avg_confidence_critical:.2f}",
                ))
            elif metrics.avg_heuristic_confidence < t.min_avg_confidence_warning:
                alerts.append(HealthAlert(
                    level="warning",
                    category="confidence",
                    message="Average heuristic confidence is low",
                    current_value=f"{metrics.avg_heuristic_confidence:.2f}",
                    threshold=f"{t.min_avg_confidence_warning:.2f}",
                ))

        # Check storage size
        if metrics.storage_bytes >= t.max_storage_bytes_critical:
            alerts.append(HealthAlert(
                level="critical",
                category="storage",
                message="Memory storage size critically high",
                current_value=f"{metrics.storage_bytes / (1024*1024):.1f}MB",
                threshold=f"{t.max_storage_bytes_critical / (1024*1024):.1f}MB",
            ))

        # Notify handlers
        for alert in alerts:
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

        return alerts

    def get_growth_trend(self, project_id: str) -> Dict[str, Any]:
        """
        Analyze memory growth trend from history.

        Args:
            project_id: Project to analyze

        Returns:
            Trend analysis
        """
        if len(self._metrics_history) < 2:
            return {
                "status": "insufficient_data",
                "samples": len(self._metrics_history),
            }

        first = self._metrics_history[0]
        last = self._metrics_history[-1]

        time_span = (last.timestamp - first.timestamp).total_seconds()
        if time_span <= 0:
            return {"status": "insufficient_time_span"}

        days_span = time_span / (24 * 60 * 60)
        item_growth = last.total_items - first.total_items
        growth_per_day = item_growth / days_span if days_span > 0 else 0

        return {
            "status": "ok",
            "samples": len(self._metrics_history),
            "time_span_days": round(days_span, 1),
            "total_growth": item_growth,
            "growth_per_day": round(growth_per_day, 2),
            "first_total": first.total_items,
            "last_total": last.total_items,
            "confidence_trend": round(
                last.avg_heuristic_confidence - first.avg_heuristic_confidence, 3
            ),
        }


# ==================== CLEANUP SCHEDULER ====================


@dataclass
class CleanupJob:
    """Configuration for a scheduled cleanup job."""
    name: str
    project_id: str
    interval_hours: float
    agent: Optional[str] = None
    policy: Optional[PrunePolicy] = None
    apply_decay: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True


@dataclass
class CleanupResult:
    """Result of a cleanup job execution."""
    job_name: str
    project_id: str
    started_at: datetime
    completed_at: datetime
    prune_summary: Optional[PruneSummary] = None
    decay_result: Optional[DecayResult] = None
    alerts: List[HealthAlert] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class CleanupScheduler:
    """
    Schedules and executes automated memory cleanup jobs.

    Features:
    - Configurable job intervals
    - Prune + decay in single operation
    - Health check integration
    - Job execution history
    - Thread-safe operation
    """

    def __init__(
        self,
        storage: StorageBackend,
        forgetting_engine: Optional[ForgettingEngine] = None,
        decayer: Optional[ConfidenceDecayer] = None,
        health_monitor: Optional[MemoryHealthMonitor] = None,
    ):
        """
        Initialize cleanup scheduler.

        Args:
            storage: Storage backend
            forgetting_engine: Engine for pruning (created if not provided)
            decayer: Engine for decay (created if not provided)
            health_monitor: Health monitor (created if not provided)
        """
        self.storage = storage
        self.forgetting_engine = forgetting_engine or ForgettingEngine(storage)
        self.decayer = decayer or ConfidenceDecayer(storage)
        self.health_monitor = health_monitor or MemoryHealthMonitor(storage)

        self._jobs: Dict[str, CleanupJob] = {}
        self._history: List[CleanupResult] = []
        self._max_history = 50
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_job(self, job: CleanupJob) -> None:
        """
        Register a cleanup job.

        Args:
            job: Job configuration
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            job.next_run = now + timedelta(hours=job.interval_hours)
            self._jobs[job.name] = job
            logger.info(f"Registered cleanup job '{job.name}' for project {job.project_id}")

    def unregister_job(self, name: str) -> bool:
        """
        Unregister a cleanup job.

        Args:
            name: Job name

        Returns:
            True if job was found and removed
        """
        with self._lock:
            if name in self._jobs:
                del self._jobs[name]
                logger.info(f"Unregistered cleanup job '{name}'")
                return True
            return False

    def run_job(self, name: str, dry_run: bool = False) -> CleanupResult:
        """
        Manually run a specific job.

        Args:
            name: Job name
            dry_run: If True, don't actually modify data

        Returns:
            CleanupResult with execution details
        """
        with self._lock:
            if name not in self._jobs:
                raise ValueError(f"Job '{name}' not found")
            job = self._jobs[name]

        return self._execute_job(job, dry_run)

    def run_all_due(self) -> List[CleanupResult]:
        """
        Run all jobs that are due.

        Returns:
            List of results for executed jobs
        """
        results = []
        now = datetime.now(timezone.utc)

        with self._lock:
            due_jobs = [
                job for job in self._jobs.values()
                if job.enabled and job.next_run and job.next_run <= now
            ]

        for job in due_jobs:
            try:
                result = self._execute_job(job)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running job '{job.name}': {e}")
                results.append(CleanupResult(
                    job_name=job.name,
                    project_id=job.project_id,
                    started_at=now,
                    completed_at=datetime.now(timezone.utc),
                    success=False,
                    error=str(e),
                ))

        return results

    def _execute_job(self, job: CleanupJob, dry_run: bool = False) -> CleanupResult:
        """Execute a cleanup job."""
        started_at = datetime.now(timezone.utc)
        result = CleanupResult(
            job_name=job.name,
            project_id=job.project_id,
            started_at=started_at,
            completed_at=started_at,
        )

        try:
            # Run prune
            engine = ForgettingEngine(
                self.storage,
                job.policy or self.forgetting_engine.policy,
            )
            result.prune_summary = engine.prune(
                project_id=job.project_id,
                agent=job.agent,
                dry_run=dry_run,
            )

            # Run decay if enabled
            if job.apply_decay:
                result.decay_result = self.decayer.apply_decay(
                    project_id=job.project_id,
                    agent=job.agent,
                    dry_run=dry_run,
                )

            # Check health
            result.alerts = self.health_monitor.check_health(job.project_id)

            # Update job timing
            with self._lock:
                now = datetime.now(timezone.utc)
                job.last_run = now
                job.next_run = now + timedelta(hours=job.interval_hours)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Cleanup job '{job.name}' failed: {e}")

        result.completed_at = datetime.now(timezone.utc)

        # Store in history
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        return result

    def start_background(self, check_interval_seconds: int = 60) -> None:
        """
        Start background job execution thread.

        Args:
            check_interval_seconds: How often to check for due jobs
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True

        def run():
            while self._running:
                try:
                    self.run_all_due()
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                time.sleep(check_interval_seconds)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        logger.info(f"Cleanup scheduler started (interval: {check_interval_seconds}s)")

    def stop_background(self) -> None:
        """Stop the background execution thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Cleanup scheduler stopped")

    def get_jobs(self) -> List[Dict[str, Any]]:
        """Get all registered jobs."""
        with self._lock:
            return [
                {
                    "name": job.name,
                    "project_id": job.project_id,
                    "interval_hours": job.interval_hours,
                    "agent": job.agent,
                    "enabled": job.enabled,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                }
                for job in self._jobs.values()
            ]

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job execution history."""
        with self._lock:
            recent = self._history[-limit:]
            return [
                {
                    "job_name": r.job_name,
                    "project_id": r.project_id,
                    "started_at": r.started_at.isoformat(),
                    "completed_at": r.completed_at.isoformat(),
                    "duration_ms": int(
                        (r.completed_at - r.started_at).total_seconds() * 1000
                    ),
                    "success": r.success,
                    "items_pruned": r.prune_summary.total_pruned if r.prune_summary else 0,
                    "items_decayed": r.decay_result.items_updated if r.decay_result else 0,
                    "alerts": len(r.alerts),
                    "error": r.error,
                }
                for r in reversed(recent)
            ]
