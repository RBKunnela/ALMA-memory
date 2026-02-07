"""
ALMA Retrieval Feedback Tracker.

Tracks which memories were retrieved, correlates with task outcomes,
and computes effectiveness scores to adjust future retrieval.

Storage: Uses DomainKnowledge with domain="__alma_retrieval_feedback"
to leverage existing infrastructure without ABC changes.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional

from alma.rag.feedback_types import (
    RetrievalEffectiveness,
    RetrievalFeedback,
    RetrievalRecord,
)
from alma.storage.base import StorageBackend
from alma.types import DomainKnowledge

logger = logging.getLogger(__name__)

FEEDBACK_DOMAIN = "__alma_retrieval_feedback"
RECORD_SOURCE = "retrieval_record"
FEEDBACK_SOURCE = "retrieval_feedback"


class RetrievalFeedbackTracker:
    """Tracks retrieval-outcome correlations and computes effectiveness.

    The feedback loop:
    1. record_retrieval() -- called after each retrieval, logs what was returned.
    2. record_feedback() -- called after learn(), links retrieval to outcome.
    3. get_memory_effectiveness() -- aggregates per-memory stats.
    4. compute_weight_adjustments() -- returns scoring adjustments.

    Data is stored as DomainKnowledge with a special domain to avoid
    modifying the StorageBackend ABC.
    """

    def __init__(self, storage: StorageBackend, project_id: str) -> None:
        self.storage = storage
        self.project_id = project_id
        # In-memory index of recent retrieval records for correlation
        self._recent_records: Dict[str, RetrievalRecord] = {}
        self._max_recent = 100

    def record_retrieval(
        self,
        query: str,
        agent: str,
        memory_ids: List[str],
        scores: Optional[Dict[str, float]] = None,
        chunk_ids: Optional[List[str]] = None,
        mode: str = "default",
    ) -> RetrievalRecord:
        """Record what was retrieved for a query.

        Args:
            query: The search query.
            agent: Requesting agent.
            memory_ids: IDs of retrieved memories.
            scores: Optional mapping of memory_id -> score.
            chunk_ids: Optional RAG chunk IDs.
            mode: Retrieval mode used.

        Returns:
            The created RetrievalRecord.
        """
        record = RetrievalRecord(
            id=f"rr_{uuid.uuid4().hex[:12]}",
            query=query,
            agent=agent,
            project_id=self.project_id,
            memory_ids=memory_ids,
            chunk_ids=chunk_ids or [],
            scores=scores or {},
            mode=mode,
        )

        # Store in memory for fast correlation
        self._recent_records[record.id] = record
        if len(self._recent_records) > self._max_recent:
            oldest_key = next(iter(self._recent_records))
            del self._recent_records[oldest_key]

        # Persist as DomainKnowledge
        dk = DomainKnowledge(
            id=record.id,
            agent=agent,
            project_id=self.project_id,
            domain=FEEDBACK_DOMAIN,
            fact=json.dumps(
                {
                    "type": "retrieval_record",
                    "query": query,
                    "memory_ids": memory_ids,
                    "chunk_ids": record.chunk_ids,
                    "scores": scores or {},
                    "mode": mode,
                }
            ),
            source=RECORD_SOURCE,
            confidence=1.0,
            last_verified=record.timestamp,
        )
        self.storage.save_domain_knowledge(dk)

        logger.debug(
            f"Recorded retrieval: {len(memory_ids)} memories for '{query[:50]}'"
        )
        return record

    def record_feedback(
        self,
        retrieval_record_id: str,
        outcome_id: str,
        success: bool,
        helpful_memory_ids: Optional[List[str]] = None,
        unhelpful_memory_ids: Optional[List[str]] = None,
    ) -> Optional[RetrievalFeedback]:
        """Record feedback linking retrieval to outcome.

        Args:
            retrieval_record_id: ID of the RetrievalRecord.
            outcome_id: ID of the Outcome from learn().
            success: Whether the task succeeded.
            helpful_memory_ids: Memories explicitly marked as helpful.
            unhelpful_memory_ids: Memories explicitly marked as unhelpful.

        Returns:
            The created RetrievalFeedback, or None if record not found.
        """
        # Look up the retrieval record
        record = self._recent_records.get(retrieval_record_id)
        if record is None:
            logger.warning(
                f"Retrieval record {retrieval_record_id} not found in recent records"
            )
            return None

        feedback = RetrievalFeedback(
            id=f"rf_{uuid.uuid4().hex[:12]}",
            retrieval_record_id=retrieval_record_id,
            outcome_id=outcome_id,
            success=success,
            helpful_memory_ids=helpful_memory_ids or [],
            unhelpful_memory_ids=unhelpful_memory_ids or [],
        )

        # Persist as DomainKnowledge
        dk = DomainKnowledge(
            id=feedback.id,
            agent=record.agent,
            project_id=self.project_id,
            domain=FEEDBACK_DOMAIN,
            fact=json.dumps(
                {
                    "type": "retrieval_feedback",
                    "retrieval_record_id": retrieval_record_id,
                    "outcome_id": outcome_id,
                    "success": success,
                    "memory_ids": record.memory_ids,
                    "helpful_memory_ids": feedback.helpful_memory_ids,
                    "unhelpful_memory_ids": feedback.unhelpful_memory_ids,
                }
            ),
            source=FEEDBACK_SOURCE,
            confidence=1.0,
            last_verified=feedback.timestamp,
        )
        self.storage.save_domain_knowledge(dk)

        logger.info(
            f"Recorded feedback for retrieval {retrieval_record_id}: "
            f"{'success' if success else 'failure'}"
        )
        return feedback

    def get_memory_effectiveness(
        self,
        agent: Optional[str] = None,
    ) -> Dict[str, RetrievalEffectiveness]:
        """Compute effectiveness statistics per memory.

        Scans all feedback records and aggregates success/failure
        counts for each retrieved memory.

        Args:
            agent: Optional agent filter.

        Returns:
            Dict of memory_id -> RetrievalEffectiveness.
        """
        # Retrieve all feedback records from storage
        feedback_records = self.storage.get_domain_knowledge(
            project_id=self.project_id,
            agent=agent or "",
            top_k=1000,
        )

        effectiveness: Dict[str, RetrievalEffectiveness] = {}

        for dk in feedback_records:
            if dk.domain != FEEDBACK_DOMAIN or dk.source != FEEDBACK_SOURCE:
                continue

            try:
                data = json.loads(dk.fact)
                if data.get("type") != "retrieval_feedback":
                    continue

                success = data["success"]
                memory_ids = data.get("memory_ids", [])

                for mid in memory_ids:
                    if mid not in effectiveness:
                        effectiveness[mid] = RetrievalEffectiveness(memory_id=mid)

                    eff = effectiveness[mid]
                    eff.times_retrieved += 1
                    if success:
                        eff.times_in_success += 1
                    else:
                        eff.times_in_failure += 1
                    eff.last_retrieved = dk.last_verified

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse feedback record {dk.id}: {e}")
                continue

        # Compute correlations
        for eff in effectiveness.values():
            eff.compute_correlation()
            eff.compute_weight_adjustment()

        return effectiveness

    def compute_weight_adjustments(
        self,
        agent: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute weight adjustments for retrieval scoring.

        Returns a mapping of memory_id -> weight_adjustment that
        can be applied to the scoring engine.

        Args:
            agent: Optional agent filter.

        Returns:
            Dict of memory_id -> weight adjustment multiplier.
        """
        effectiveness = self.get_memory_effectiveness(agent)
        return {
            mid: eff.weight_adjustment
            for mid, eff in effectiveness.items()
            if abs(eff.weight_adjustment - 1.0) > 0.01  # Only non-neutral
        }

    def get_last_retrieval_record(self, agent: str) -> Optional[RetrievalRecord]:
        """Get the most recent retrieval record for an agent.

        Useful for auto-linking feedback after learn() calls.

        Args:
            agent: Agent name.

        Returns:
            Most recent RetrievalRecord for the agent, or None.
        """
        for record in reversed(list(self._recent_records.values())):
            if record.agent == agent:
                return record
        return None
