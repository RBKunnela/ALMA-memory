"""
ALMA Retrieval Feedback Loop.

Tracks which retrieved memories agents actually use vs ignore,
and adjusts future retrieval scores via post-scoring re-ranking.

The feedback loop consists of two components:
- FeedbackTracker: Records feedback signals to storage
- FeedbackAwareScorer: Re-ranks scored items using accumulated feedback
"""

import logging
import uuid
from typing import Dict, List

from alma.retrieval.scoring import ScoredItem
from alma.storage.base import StorageBackend
from alma.types import (
    FeedbackSignal,
    FeedbackSummary,
    MemoryType,
    RetrievalFeedback,
)

logger = logging.getLogger(__name__)


class FeedbackTracker:
    """
    Records retrieval feedback signals to storage.

    Wraps a storage backend to provide convenience methods for
    recording which memories were used vs ignored during agent
    task execution.

    Args:
        storage: Storage backend for persisting feedback records.

    Example:
        >>> tracker = FeedbackTracker(storage)
        >>> tracker.record_usage(
        ...     retrieved_ids=["m1", "m2", "m3"],
        ...     used_ids=["m1"],
        ...     memory_type=MemoryType.HEURISTIC,
        ...     agent="qa-agent",
        ...     project_id="proj-1",
        ...     query="how to test forms",
        ... )
    """

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    def record(self, feedback: RetrievalFeedback) -> str:
        """
        Record a single feedback signal.

        Args:
            feedback: The feedback record to persist.

        Returns:
            The feedback record ID.
        """
        return self.storage.save_retrieval_feedback(feedback)

    def record_usage(
        self,
        retrieved_ids: List[str],
        used_ids: List[str],
        memory_type: MemoryType,
        agent: str,
        project_id: str,
        query: str = "",
    ) -> List[str]:
        """
        Record usage feedback for a batch of retrieved memories.

        Marks each retrieved memory as USED if its ID appears in
        ``used_ids``, otherwise marks it as IGNORED.

        Args:
            retrieved_ids: All memory IDs that were returned by retrieval.
            used_ids: Subset of retrieved_ids that the agent actually used.
            memory_type: The type of memories being tracked.
            agent: Agent that performed the retrieval.
            project_id: Project context.
            query: Original retrieval query (for analytics).

        Returns:
            List of created feedback record IDs.
        """
        used_set = set(used_ids)
        feedback_ids: List[str] = []

        for memory_id in retrieved_ids:
            signal = (
                FeedbackSignal.USED if memory_id in used_set else FeedbackSignal.IGNORED
            )
            feedback = RetrievalFeedback(
                id=str(uuid.uuid4()),
                memory_id=memory_id,
                memory_type=memory_type,
                query=query,
                agent=agent,
                project_id=project_id,
                signal=signal,
            )
            feedback_id = self.storage.save_retrieval_feedback(feedback)
            feedback_ids.append(feedback_id)

        logger.debug(
            "Recorded usage feedback: %d used, %d ignored for agent=%s",
            len(used_set & set(retrieved_ids)),
            len(set(retrieved_ids) - used_set),
            agent,
        )
        return feedback_ids

    def get_summaries(
        self,
        memory_ids: List[str],
        memory_type: MemoryType,
    ) -> Dict[str, FeedbackSummary]:
        """
        Get aggregated feedback summaries for a set of memories.

        Args:
            memory_ids: Memory IDs to get summaries for.
            memory_type: Type of the memories.

        Returns:
            Dict mapping memory_id to its FeedbackSummary.
            Only includes entries for memories that have feedback.
        """
        return self.storage.get_feedback_summary(memory_ids, memory_type)


class FeedbackAwareScorer:
    """
    Post-scoring re-ranker that blends feedback signals into scores.

    After the base scorer produces ``List[ScoredItem]``, this class
    adjusts each item's score based on accumulated feedback, then
    re-sorts the list.

    The blending formula is:
        final_score = (1 - weight) * base_score + weight * normalized_feedback

    Where ``normalized_feedback`` maps feedback_score from [-1, 1] to [0, 1].

    Args:
        feedback_tracker: Tracker to query for feedback summaries.
        feedback_weight: How much feedback influences the final score.
            0.0 = no influence, 1.0 = fully feedback-driven.
            Default 0.15 (15% influence).
    """

    def __init__(
        self,
        feedback_tracker: FeedbackTracker,
        feedback_weight: float = 0.15,
    ) -> None:
        if not 0.0 <= feedback_weight <= 1.0:
            raise ValueError(
                f"feedback_weight must be between 0.0 and 1.0, got {feedback_weight}"
            )
        self.feedback_tracker = feedback_tracker
        self.feedback_weight = feedback_weight

    def apply_feedback(
        self,
        items: List[ScoredItem],
        memory_type: MemoryType,
    ) -> List[ScoredItem]:
        """
        Adjust scores based on feedback and re-sort.

        Items without feedback retain their original scores unchanged.

        Args:
            items: Scored items from the base scorer.
            memory_type: Type of memories being scored.

        Returns:
            Re-ranked list of ScoredItem with adjusted scores.
        """
        if not items or self.feedback_weight == 0.0:
            return items

        # Extract memory IDs from scored items
        memory_ids = [getattr(item.item, "id", None) for item in items]
        valid_ids = [mid for mid in memory_ids if mid is not None]

        if not valid_ids:
            return items

        # Fetch feedback summaries
        summaries = self.feedback_tracker.get_summaries(valid_ids, memory_type)

        if not summaries:
            return items

        # Blend feedback into scores
        adjusted: List[ScoredItem] = []
        for item in items:
            item_id = getattr(item.item, "id", None)
            summary = summaries.get(item_id) if item_id else None

            if summary is not None:
                # Normalize feedback_score from [-1, 1] to [0, 1]
                normalized = (summary.feedback_score + 1.0) / 2.0
                new_score = (
                    1.0 - self.feedback_weight
                ) * item.score + self.feedback_weight * normalized
                adjusted.append(
                    ScoredItem(
                        item=item.item,
                        score=new_score,
                        similarity_score=item.similarity_score,
                        recency_score=item.recency_score,
                        success_score=item.success_score,
                        confidence_score=item.confidence_score,
                    )
                )
            else:
                adjusted.append(item)

        # Re-sort by adjusted score (descending)
        adjusted.sort(key=lambda x: x.score, reverse=True)
        return adjusted
