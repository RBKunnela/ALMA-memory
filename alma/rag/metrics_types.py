"""
ALMA Retrieval Metrics Types.

Data structures for IR quality measurement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RelevanceJudgment:
    """Ground truth relevance for a query-document pair.

    Attributes:
        query_id: Identifier for the query.
        memory_id: Identifier for the memory/document.
        relevance: Relevance grade (0=irrelevant, 1=relevant, 2=highly relevant).
    """

    query_id: str
    memory_id: str
    relevance: int = 1  # 0, 1, or 2


@dataclass
class MetricsResult:
    """Result of a metrics evaluation.

    Contains standard IR metrics computed over a set of queries.

    Attributes:
        mrr: Mean Reciprocal Rank (higher is better, max 1.0).
        ndcg_at_k: NDCG at various K values.
        recall_at_k: Recall at various K values.
        precision_at_k: Precision at various K values.
        map_score: Mean Average Precision.
        num_queries: Number of queries evaluated.
        k_values: The K values used for @K metrics.
        timestamp: When the evaluation was run.
    """

    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    num_queries: int = 0
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "map_score": self.map_score,
            "num_queries": self.num_queries,
            "k_values": self.k_values,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricsHistory:
    """Tracks metrics over time for trend analysis.

    Attributes:
        results: List of historical MetricsResult.
        agent: Agent these metrics belong to.
        project_id: Project context.
    """

    results: List[MetricsResult] = field(default_factory=list)
    agent: str = ""
    project_id: str = ""

    def add(self, result: MetricsResult) -> None:
        """Add a metrics result to history."""
        self.results.append(result)

    @property
    def latest(self) -> Optional[MetricsResult]:
        """Get the most recent result."""
        return self.results[-1] if self.results else None

    def trend(self, metric: str = "mrr", last_n: int = 5) -> List[float]:
        """Get trend for a metric over last N evaluations.

        Args:
            metric: Metric name ("mrr", "map_score").
            last_n: Number of recent results to include.

        Returns:
            List of metric values (oldest to newest).
        """
        recent = self.results[-last_n:]
        return [getattr(r, metric, 0.0) for r in recent]
