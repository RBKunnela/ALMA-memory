"""
ALMA Learning Protocols.

Defines how agents learn from outcomes while respecting scope constraints.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemoryScope,
    Outcome,
    UserPreference,
)

if TYPE_CHECKING:
    from alma.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class LearningProtocol:
    """
    Manages how agents learn from task outcomes.

    Key principles:
    - Validate scope before any learning
    - Require minimum occurrences before creating heuristics
    - Support forgetting to prevent memory bloat
    """

    def __init__(
        self,
        storage: StorageBackend,
        scopes: Dict[str, MemoryScope],
        embedder: Optional["EmbeddingProvider"] = None,
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize learning protocol.

        Args:
            storage: Storage backend for persistence
            scopes: Dict of agent_name -> MemoryScope
            embedder: Optional embedding provider for semantic similarity
            similarity_threshold: Cosine similarity threshold for strategy matching (default 0.75)
        """
        self.storage = storage
        self.scopes = scopes
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def learn(
        self,
        agent: str,
        project_id: str,
        task: str,
        outcome: bool,
        strategy_used: str,
        task_type: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> Outcome:
        """
        Learn from a task outcome.

        Creates an Outcome record and potentially updates/creates heuristics.

        Args:
            agent: Agent that executed the task
            project_id: Project context
            task: Task description
            outcome: True if successful, False if failed
            strategy_used: The approach taken
            task_type: Category for grouping
            duration_ms: Execution time
            error_message: Error details if failed
            feedback: User feedback

        Returns:
            The created Outcome record
        """
        # Validate agent has a scope (warn but don't block)
        scope = self.scopes.get(agent)
        if scope is None:
            logger.warning(f"Agent '{agent}' has no defined scope")

        # Create outcome record
        outcome_record = Outcome(
            id=f"out_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            task_type=task_type or self._infer_task_type(task),
            task_description=task,
            success=outcome,
            strategy_used=strategy_used,
            duration_ms=duration_ms,
            error_message=error_message,
            user_feedback=feedback,
            timestamp=datetime.now(timezone.utc),
        )

        # Save outcome
        self.storage.save_outcome(outcome_record)
        logger.info(
            f"Recorded outcome for {agent}: {'success' if outcome else 'failure'}"
        )

        # Check if we should create/update a heuristic
        self._maybe_create_heuristic(
            agent=agent,
            project_id=project_id,
            task_type=outcome_record.task_type,
            strategy=strategy_used,
            success=outcome,
            scope=scope,
        )

        # If failure with clear pattern, consider anti-pattern
        if not outcome and error_message:
            self._maybe_create_anti_pattern(
                agent=agent,
                project_id=project_id,
                task=task,
                strategy=strategy_used,
                error=error_message,
            )

        return outcome_record

    def _maybe_create_heuristic(
        self,
        agent: str,
        project_id: str,
        task_type: str,
        strategy: str,
        success: bool,
        scope: Optional[MemoryScope],
    ):
        """
        Create or update a heuristic if we have enough occurrences.

        Only creates heuristic after min_occurrences similar outcomes.
        """
        min_occurrences = 3
        if scope:
            min_occurrences = scope.min_occurrences_for_heuristic

        # Get similar outcomes to check occurrence count
        similar_outcomes = self.storage.get_outcomes(
            project_id=project_id,
            agent=agent,
            task_type=task_type,
            top_k=min_occurrences + 1,
            success_only=False,
        )

        # Filter to same strategy
        same_strategy = [
            o
            for o in similar_outcomes
            if self._strategies_similar(o.strategy_used, strategy)
        ]

        if len(same_strategy) >= min_occurrences:
            success_count = sum(1 for o in same_strategy if o.success)
            confidence = success_count / len(same_strategy)

            # Only create heuristic if confidence is meaningful
            if confidence > 0.5:
                heuristic = Heuristic(
                    id=f"heur_{uuid.uuid4().hex[:12]}",
                    agent=agent,
                    project_id=project_id,
                    condition=f"task type: {task_type}",
                    strategy=strategy,
                    confidence=confidence,
                    occurrence_count=len(same_strategy),
                    success_count=success_count,
                    last_validated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )
                self.storage.save_heuristic(heuristic)
                logger.info(
                    f"Created heuristic for {agent}: {strategy[:50]}... "
                    f"(confidence: {confidence:.0%})"
                )

    def _maybe_create_anti_pattern(
        self,
        agent: str,
        project_id: str,
        task: str,
        strategy: str,
        error: str,
    ):
        """Create anti-pattern if we see repeated failures with same pattern."""
        # Check for similar failures
        similar_failures = self.storage.get_outcomes(
            project_id=project_id,
            agent=agent,
            success_only=False,
            top_k=10,
        )

        # Filter to failures with similar error
        similar = [
            o
            for o in similar_failures
            if not o.success
            and o.error_message
            and self._errors_similar(o.error_message, error)
        ]

        if len(similar) >= 2:  # At least 2 similar failures
            anti_pattern = AntiPattern(
                id=f"anti_{uuid.uuid4().hex[:12]}",
                agent=agent,
                project_id=project_id,
                pattern=strategy,
                why_bad=error,
                better_alternative="[To be determined from successful outcomes]",
                occurrence_count=len(similar),
                last_seen=datetime.now(timezone.utc),
            )
            self.storage.save_anti_pattern(anti_pattern)
            logger.info(f"Created anti-pattern for {agent}: {strategy[:50]}...")

    def add_preference(
        self,
        user_id: str,
        category: str,
        preference: str,
        source: str,
    ) -> UserPreference:
        """Add a user preference."""
        pref = UserPreference(
            id=f"pref_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            category=category,
            preference=preference,
            source=source,
            confidence=1.0 if source == "explicit_instruction" else 0.7,
            timestamp=datetime.now(timezone.utc),
        )
        self.storage.save_user_preference(pref)
        return pref

    def add_domain_knowledge(
        self,
        agent: str,
        project_id: str,
        domain: str,
        fact: str,
        source: str,
    ) -> DomainKnowledge:
        """Add domain knowledge."""
        knowledge = DomainKnowledge(
            id=f"dk_{uuid.uuid4().hex[:12]}",
            agent=agent,
            project_id=project_id,
            domain=domain,
            fact=fact,
            source=source,
            confidence=1.0 if source == "user_stated" else 0.8,
            last_verified=datetime.now(timezone.utc),
        )
        self.storage.save_domain_knowledge(knowledge)
        return knowledge

    def forget(
        self,
        project_id: str,
        agent: Optional[str] = None,
        older_than_days: int = 90,
        below_confidence: float = 0.3,
    ) -> int:
        """
        Prune stale and low-confidence memories.

        Returns:
            Total number of items pruned
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        # Delete old outcomes
        outcomes_deleted = self.storage.delete_outcomes_older_than(
            project_id=project_id,
            older_than=cutoff,
            agent=agent,
        )

        # Delete low-confidence heuristics
        heuristics_deleted = self.storage.delete_low_confidence_heuristics(
            project_id=project_id,
            below_confidence=below_confidence,
            agent=agent,
        )

        total = outcomes_deleted + heuristics_deleted
        logger.info(
            f"Forgot {total} items: {outcomes_deleted} outcomes, "
            f"{heuristics_deleted} heuristics"
        )
        return total

    def _infer_task_type(self, task: str) -> str:
        """Infer task type from description."""
        task_lower = task.lower()
        if "test" in task_lower or "validate" in task_lower:
            return "testing"
        elif "api" in task_lower or "endpoint" in task_lower:
            return "api_testing"
        elif "form" in task_lower or "input" in task_lower:
            return "form_testing"
        elif "database" in task_lower or "query" in task_lower:
            return "database_validation"
        return "general"

    def _strategies_similar(self, s1: str, s2: str) -> bool:
        """
        Check if two strategies are similar enough to count together.

        Uses embedding-based cosine similarity when an embedder is available,
        otherwise falls back to simple word overlap.
        """
        if self.embedder is not None:
            return self._strategies_similar_embedding(s1, s2)
        return self._strategies_similar_word_overlap(s1, s2)

    def _strategies_similar_embedding(self, s1: str, s2: str) -> bool:
        """Check strategy similarity using embedding cosine similarity."""
        try:
            emb1 = self.embedder.encode(s1)
            emb2 = self.embedder.encode(s2)
            similarity = self._cosine_similarity(emb1, emb2)
            return similarity >= self.similarity_threshold
        except Exception as e:
            logger.warning(
                f"Embedding similarity failed, falling back to word overlap: {e}"
            )
            return self._strategies_similar_word_overlap(s1, s2)

    def _strategies_similar_word_overlap(self, s1: str, s2: str) -> bool:
        """Check strategy similarity using simple word overlap."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        overlap = len(words1 & words2)
        return overlap >= min(3, len(words1) // 2)

    def _cosine_similarity(self, v1: list, v2: list) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _errors_similar(self, e1: str, e2: str) -> bool:
        """Check if two errors are similar."""
        # Simple substring check
        e1_lower = e1.lower()
        e2_lower = e2.lower()
        return e1_lower in e2_lower or e2_lower in e1_lower
