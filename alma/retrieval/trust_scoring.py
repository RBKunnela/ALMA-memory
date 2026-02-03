"""
ALMA Trust-Integrated Scoring.

Extends memory scoring to incorporate trust patterns from agent behavior.
Based on Veritas trust evolution concepts:
- Trust decays without activity
- Per-behavior trust tracking
- Cross-session trust persistence

Features:
- Trust-weighted scoring for memories from different sources
- Agent trust profile integration
- Trust pattern storage and retrieval
- Violation-aware scoring (anti-patterns from violations)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alma.retrieval.scoring import MemoryScorer, ScoredItem, ScoringWeights
from alma.types import (
    AntiPattern,
    Heuristic,
    MemorySlice,
    Outcome,
)

logger = logging.getLogger(__name__)


class TrustLevel:
    """Trust level constants matching Veritas framework."""

    UNTRUSTED = 0.0
    MINIMAL = 0.2
    LOW = 0.4
    MODERATE = 0.5
    GOOD = 0.7
    HIGH = 0.85
    FULL = 1.0

    @classmethod
    def label(cls, score: float) -> str:
        """Get human-readable label for trust score."""
        if score >= cls.FULL:
            return "FULL"
        elif score >= cls.HIGH:
            return "HIGH"
        elif score >= cls.GOOD:
            return "GOOD"
        elif score >= cls.MODERATE:
            return "MODERATE"
        elif score >= cls.LOW:
            return "LOW"
        elif score >= cls.MINIMAL:
            return "MINIMAL"
        return "UNTRUSTED"


@dataclass
class TrustWeights(ScoringWeights):
    """
    Extended scoring weights that include trust factor.

    Extends base ScoringWeights with trust component.
    All weights are normalized to sum to 1.0.
    """

    # Inherited from ScoringWeights
    similarity: float = 0.35
    recency: float = 0.25
    success_rate: float = 0.15
    confidence: float = 0.10

    # Trust factor
    trust: float = 0.15  # Agent/source trust score

    def __post_init__(self):
        """Validate and normalize weights."""
        total = (
            self.similarity
            + self.recency
            + self.success_rate
            + self.confidence
            + self.trust
        )
        if not (0.99 <= total <= 1.01):
            # Normalize
            self.similarity /= total
            self.recency /= total
            self.success_rate /= total
            self.confidence /= total
            self.trust /= total


@dataclass
class AgentTrustProfile:
    """
    Trust profile for an agent, stored in ALMA memory.

    Tracks trust metrics that influence memory scoring.
    """

    agent_id: str
    current_trust: float = TrustLevel.MODERATE
    sessions_completed: int = 0
    total_actions: int = 0
    total_violations: int = 0
    consecutive_clean_sessions: int = 0
    last_session: Optional[datetime] = None
    trust_half_life_days: int = 30

    # Per-behavior trust scores
    behavior_trust: Dict[str, float] = field(default_factory=lambda: {
        "verification_before_claim": 1.0,
        "loud_failure": 1.0,
        "honest_uncertainty": 1.0,
        "paper_trail": 1.0,
        "diligent_execution": 1.0,
    })

    def calculate_trust(self) -> float:
        """Calculate current trust with decay."""
        if self.sessions_completed == 0:
            return TrustLevel.MODERATE

        # Performance factor
        if self.total_actions > 0:
            performance = 1.0 - (self.total_violations / self.total_actions)
        else:
            performance = TrustLevel.MODERATE

        # Behavior average
        behavior_avg = (
            sum(self.behavior_trust.values()) / len(self.behavior_trust)
            if self.behavior_trust
            else 1.0
        )

        # Time decay
        decay = self._calculate_decay()

        # Streak bonus/penalty
        streak_bonus = min(0.1, self.consecutive_clean_sessions * 0.02)

        # Combine
        raw_trust = performance * 0.4 + behavior_avg * 0.4 + TrustLevel.MODERATE * 0.2
        self.current_trust = max(0.0, min(1.0, raw_trust * decay + streak_bonus))

        return self.current_trust

    def _calculate_decay(self) -> float:
        """Calculate trust decay based on time since last session."""
        if not self.last_session:
            return 1.0

        now = datetime.now(timezone.utc)
        if self.last_session.tzinfo is None:
            last = self.last_session.replace(tzinfo=timezone.utc)
        else:
            last = self.last_session

        days_since = (now - last).total_seconds() / (24 * 60 * 60)
        decay = math.exp(-0.693 * days_since / self.trust_half_life_days)
        return max(0.5, decay)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "agent_id": self.agent_id,
            "current_trust": self.current_trust,
            "sessions_completed": self.sessions_completed,
            "total_actions": self.total_actions,
            "total_violations": self.total_violations,
            "consecutive_clean_sessions": self.consecutive_clean_sessions,
            "last_session": self.last_session.isoformat() if self.last_session else None,
            "trust_half_life_days": self.trust_half_life_days,
            "behavior_trust": self.behavior_trust,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTrustProfile":
        """Create from dictionary."""
        last_session = data.get("last_session")
        if isinstance(last_session, str):
            last_session = datetime.fromisoformat(last_session)

        return cls(
            agent_id=data["agent_id"],
            current_trust=data.get("current_trust", TrustLevel.MODERATE),
            sessions_completed=data.get("sessions_completed", 0),
            total_actions=data.get("total_actions", 0),
            total_violations=data.get("total_violations", 0),
            consecutive_clean_sessions=data.get("consecutive_clean_sessions", 0),
            last_session=last_session,
            trust_half_life_days=data.get("trust_half_life_days", 30),
            behavior_trust=data.get("behavior_trust", {}),
        )


@dataclass
class TrustScoredItem(ScoredItem):
    """Extended scored item with trust information."""

    trust_score: float = 1.0
    source_agent: Optional[str] = None
    trust_level: str = "MODERATE"


class TrustAwareScorer(MemoryScorer):
    """
    Memory scorer that incorporates trust patterns.

    Extends base MemoryScorer to weight memories by:
    - Source agent's trust level
    - Memory's verification status
    - Trust-relevant metadata
    """

    def __init__(
        self,
        weights: Optional[TrustWeights] = None,
        recency_half_life_days: float = 30.0,
        trust_profiles: Optional[Dict[str, AgentTrustProfile]] = None,
        default_trust: float = TrustLevel.MODERATE,
    ):
        # Convert TrustWeights to base ScoringWeights for parent
        tw = weights or TrustWeights()
        base_weights = ScoringWeights(
            similarity=tw.similarity,
            recency=tw.recency,
            success_rate=tw.success_rate,
            confidence=tw.confidence,
        )
        super().__init__(weights=base_weights, recency_half_life_days=recency_half_life_days)

        self.trust_weights = tw
        self.trust_profiles = trust_profiles or {}
        self.default_trust = default_trust

    def set_trust_profile(self, agent_id: str, profile: AgentTrustProfile) -> None:
        """Set trust profile for an agent."""
        self.trust_profiles[agent_id] = profile

    def get_agent_trust(self, agent_id: str) -> float:
        """Get trust score for an agent."""
        if agent_id in self.trust_profiles:
            return self.trust_profiles[agent_id].calculate_trust()
        return self.default_trust

    def score_heuristics_with_trust(
        self,
        heuristics: List[Heuristic],
        similarities: Optional[List[float]] = None,
        requesting_agent: Optional[str] = None,
    ) -> List[TrustScoredItem]:
        """
        Score heuristics including trust factor.

        Args:
            heuristics: List of heuristics to score
            similarities: Optional pre-computed similarity scores
            requesting_agent: Agent requesting the memories (for trust context)

        Returns:
            Sorted list of TrustScoredItems
        """
        if not heuristics:
            return []

        similarities = similarities or [1.0] * len(heuristics)
        scored = []

        for h, sim in zip(heuristics, similarities, strict=False):
            recency = self._compute_recency_score(h.last_validated)
            success = h.success_rate
            confidence = h.confidence

            # Get trust for source agent
            source_trust = self.get_agent_trust(h.agent)

            # Check if memory has verification metadata
            verified = h.metadata.get("verified", False)
            if verified:
                source_trust = min(1.0, source_trust * 1.1)  # 10% boost for verified

            # Compute weighted score
            total = (
                self.trust_weights.similarity * sim
                + self.trust_weights.recency * recency
                + self.trust_weights.success_rate * success
                + self.trust_weights.confidence * confidence
                + self.trust_weights.trust * source_trust
            )

            scored.append(
                TrustScoredItem(
                    item=h,
                    score=total,
                    similarity_score=sim,
                    recency_score=recency,
                    success_score=success,
                    confidence_score=confidence,
                    trust_score=source_trust,
                    source_agent=h.agent,
                    trust_level=TrustLevel.label(source_trust),
                )
            )

        return sorted(scored, key=lambda x: -x.score)

    def score_outcomes_with_trust(
        self,
        outcomes: List[Outcome],
        similarities: Optional[List[float]] = None,
    ) -> List[TrustScoredItem]:
        """Score outcomes including trust factor."""
        if not outcomes:
            return []

        similarities = similarities or [1.0] * len(outcomes)
        scored = []

        for o, sim in zip(outcomes, similarities, strict=False):
            recency = self._compute_recency_score(o.timestamp)
            success = 1.0 if o.success else 0.3
            confidence = 1.0

            source_trust = self.get_agent_trust(o.agent)

            # Boost trust for outcomes with user feedback
            if o.user_feedback:
                source_trust = min(1.0, source_trust * 1.05)

            total = (
                self.trust_weights.similarity * sim
                + self.trust_weights.recency * recency
                + self.trust_weights.success_rate * success
                + self.trust_weights.confidence * confidence
                + self.trust_weights.trust * source_trust
            )

            scored.append(
                TrustScoredItem(
                    item=o,
                    score=total,
                    similarity_score=sim,
                    recency_score=recency,
                    success_score=success,
                    confidence_score=confidence,
                    trust_score=source_trust,
                    source_agent=o.agent,
                    trust_level=TrustLevel.label(source_trust),
                )
            )

        return sorted(scored, key=lambda x: -x.score)

    def score_anti_patterns_with_trust(
        self,
        anti_patterns: List[AntiPattern],
        similarities: Optional[List[float]] = None,
    ) -> List[TrustScoredItem]:
        """
        Score anti-patterns including trust factor.

        Anti-patterns from high-trust agents are MORE important (they've
        learned from experience), so trust boosts rather than reduces score.
        """
        if not anti_patterns:
            return []

        similarities = similarities or [1.0] * len(anti_patterns)
        scored = []

        for ap, sim in zip(anti_patterns, similarities, strict=False):
            recency = self._compute_recency_score(ap.last_seen)
            # More occurrences = more important
            occurrence_score = min(ap.occurrence_count / 10.0, 1.0)
            confidence = 1.0

            source_trust = self.get_agent_trust(ap.agent)

            # For anti-patterns, higher trust = more reliable warning
            # Check if this came from a trust violation
            from_violation = ap.metadata.get("from_trust_violation", False)
            if from_violation:
                source_trust = min(1.0, source_trust * 1.2)  # 20% boost

            total = (
                self.trust_weights.similarity * sim
                + self.trust_weights.recency * recency
                + self.trust_weights.success_rate * occurrence_score
                + self.trust_weights.confidence * confidence
                + self.trust_weights.trust * source_trust
            )

            scored.append(
                TrustScoredItem(
                    item=ap,
                    score=total,
                    similarity_score=sim,
                    recency_score=recency,
                    success_score=occurrence_score,
                    confidence_score=confidence,
                    trust_score=source_trust,
                    source_agent=ap.agent,
                    trust_level=TrustLevel.label(source_trust),
                )
            )

        return sorted(scored, key=lambda x: -x.score)

    def score_with_trust(
        self,
        memory_slice: MemorySlice,
        similarities: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, List[TrustScoredItem]]:
        """
        Score all memories in a slice with trust integration.

        Args:
            memory_slice: Memories to score
            similarities: Optional dict of similarity scores by type

        Returns:
            Dict with scored items by type
        """
        similarities = similarities or {}

        return {
            "heuristics": self.score_heuristics_with_trust(
                memory_slice.heuristics,
                similarities.get("heuristics"),
            ),
            "outcomes": self.score_outcomes_with_trust(
                memory_slice.outcomes,
                similarities.get("outcomes"),
            ),
            "anti_patterns": self.score_anti_patterns_with_trust(
                memory_slice.anti_patterns,
                similarities.get("anti_patterns"),
            ),
        }


class TrustPatternStore:
    """
    Stores and retrieves trust patterns in ALMA memory.

    Enables:
    - Storing trust violations as AntiPatterns
    - Storing successful verification patterns as Heuristics
    - Retrieving trust-relevant memories for a task
    """

    def __init__(self, alma_core: Any):  # ALMA instance
        self.alma = alma_core

    async def store_trust_violation(
        self,
        agent_id: str,
        project_id: str,
        violation_type: str,
        description: str,
        severity: str,
        remediation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a trust violation as an ALMA AntiPattern.

        This allows trust violations to be retrieved as warnings
        for similar future tasks.
        """
        anti_pattern_data = {
            "pattern": f"Trust violation: {violation_type}",
            "why_bad": description,
            "better_alternative": remediation,
            "metadata": {
                "from_trust_violation": True,
                "severity": severity,
                "original_context": context or {},
            },
        }

        # Use ALMA's learn method to store
        await self.alma.learn(
            task_type="trust_enforcement",
            strategy=f"Avoid: {violation_type}",
            outcome="failure",
            details=anti_pattern_data,
            agent=agent_id,
            project_id=project_id,
        )

        logger.info(
            f"Stored trust violation as anti-pattern: {violation_type} "
            f"for agent {agent_id}"
        )

    async def store_verification_pattern(
        self,
        agent_id: str,
        project_id: str,
        task_type: str,
        verification_strategy: str,
        evidence_types: List[str],
        verification_rate: float,
    ) -> None:
        """
        Store a successful verification pattern as an ALMA Heuristic.

        This allows effective verification strategies to be retrieved
        for similar future tasks.
        """
        await self.alma.learn(
            task_type=task_type,
            strategy=verification_strategy,
            outcome="success",
            details={
                "evidence_types_used": evidence_types,
                "verification_rate": verification_rate,
                "metadata": {
                    "is_verification_pattern": True,
                },
            },
            agent=agent_id,
            project_id=project_id,
        )

        logger.info(
            f"Stored verification pattern for {task_type}: "
            f"{verification_strategy} ({verification_rate:.0%} rate)"
        )

    async def store_trust_profile(
        self,
        profile: AgentTrustProfile,
        project_id: str,
    ) -> None:
        """Store agent trust profile as domain knowledge."""
        await self.alma.add_knowledge(
            domain="trust_profiles",
            fact=profile.to_dict(),
            source="veritas_trust_evolution",
            confidence=1.0,
            agent="trust_system",
            project_id=project_id,
        )

    async def retrieve_trust_profile(
        self,
        agent_id: str,
        project_id: str,
    ) -> Optional[AgentTrustProfile]:
        """Retrieve agent trust profile from domain knowledge."""
        memories = await self.alma.retrieve(
            task=f"trust_profile:{agent_id}",
            agent="trust_system",
            project_id=project_id,
        )

        for knowledge in memories.domain_knowledge:
            if knowledge.domain == "trust_profiles":
                fact = knowledge.fact
                if isinstance(fact, dict) and fact.get("agent_id") == agent_id:
                    return AgentTrustProfile.from_dict(fact)

        return None

    async def retrieve_trust_warnings(
        self,
        task_description: str,
        agent_id: str,
        project_id: str,
    ) -> List[str]:
        """
        Retrieve trust-related warnings for a task.

        Returns warnings based on past trust violations
        that are relevant to the current task.
        """
        memories = await self.alma.retrieve(
            task=f"trust_verification:{task_description}",
            agent=agent_id,
            project_id=project_id,
            include_anti_patterns=True,
        )

        warnings = []
        for ap in memories.anti_patterns:
            if ap.metadata.get("from_trust_violation"):
                if ap.occurrence_count >= 2:
                    warnings.append(
                        f"Repeated trust issue: {ap.pattern}. "
                        f"Suggestion: {ap.better_alternative}"
                    )

        return warnings
