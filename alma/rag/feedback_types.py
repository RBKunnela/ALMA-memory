"""
ALMA Retrieval Feedback Types.

Data structures for tracking retrieval outcomes and computing effectiveness.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalRecord:
    """Records what was retrieved for a specific task.

    Created automatically when memories are retrieved, storing
    which memories were returned and their scores.

    Attributes:
        id: Unique record identifier.
        query: The query that triggered retrieval.
        agent: Agent that requested retrieval.
        project_id: Project context.
        memory_ids: IDs of memories that were retrieved.
        chunk_ids: IDs of RAG chunks (if RAG bridge was used).
        scores: Mapping of memory_id -> retrieval score.
        mode: Retrieval mode used (e.g., "broad", "precise").
        timestamp: When retrieval occurred.
        metadata: Additional retrieval context.
    """

    id: str
    query: str
    agent: str
    project_id: str
    memory_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    mode: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalFeedback:
    """Post-task feedback linking retrieval to outcome.

    Created when ALMA.learn() is called, connecting the retrieval
    record with the task outcome.

    Attributes:
        id: Unique feedback identifier.
        retrieval_record_id: Links to the RetrievalRecord.
        outcome_id: Links to the Outcome from learn().
        success: Whether the task succeeded.
        helpful_memory_ids: Memories the agent reported as helpful.
        unhelpful_memory_ids: Memories that were not useful.
        timestamp: When feedback was recorded.
    """

    id: str
    retrieval_record_id: str
    outcome_id: str
    success: bool
    helpful_memory_ids: List[str] = field(default_factory=list)
    unhelpful_memory_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RetrievalEffectiveness:
    """Aggregated effectiveness statistics for a single memory.

    Computed from multiple RetrievalFeedback records to track
    how well a memory contributes to task success.

    Attributes:
        memory_id: The memory being tracked.
        times_retrieved: Total number of times this memory was retrieved.
        times_in_success: Times retrieved when task succeeded.
        times_in_failure: Times retrieved when task failed.
        success_correlation: Ratio of success vs total retrievals.
        weight_adjustment: Recommended scoring weight adjustment.
        last_retrieved: When this memory was last retrieved.
    """

    memory_id: str
    times_retrieved: int = 0
    times_in_success: int = 0
    times_in_failure: int = 0
    success_correlation: float = 0.5
    weight_adjustment: float = 1.0
    last_retrieved: Optional[datetime] = None

    def compute_correlation(self) -> None:
        """Recompute success correlation from counts."""
        if self.times_retrieved > 0:
            self.success_correlation = self.times_in_success / self.times_retrieved
        else:
            self.success_correlation = 0.5

    def compute_weight_adjustment(self, neutral_threshold: float = 0.5) -> None:
        """Compute weight adjustment based on effectiveness.

        Memories that consistently appear in successful tasks get boosted.
        Memories that consistently appear in failures get demoted.

        Args:
            neutral_threshold: Correlation threshold for neutral (no adjustment).
        """
        if self.times_retrieved < 3:
            # Not enough data -- stay neutral
            self.weight_adjustment = 1.0
            return

        self.compute_correlation()

        if self.success_correlation > neutral_threshold + 0.1:
            # Boost: up to 1.5x for highly correlated memories
            self.weight_adjustment = (
                1.0 + (self.success_correlation - neutral_threshold) * 1.0
            )
        elif self.success_correlation < neutral_threshold - 0.1:
            # Demote: down to 0.5x for negatively correlated memories
            self.weight_adjustment = max(
                0.5, 1.0 - (neutral_threshold - self.success_correlation) * 1.0
            )
        else:
            self.weight_adjustment = 1.0
