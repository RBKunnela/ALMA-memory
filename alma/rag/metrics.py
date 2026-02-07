"""
ALMA Retrieval Metrics.

Standard IR metrics for measuring retrieval quality. Supports both
ground-truth evaluation (manual relevance labels) and outcome-based
evaluation (using task success as implicit relevance).

All metric functions are pure Python, zero external dependencies, deterministic.
"""

import logging
import math
from typing import Dict, List, Optional

from alma.rag.feedback import RetrievalFeedbackTracker
from alma.rag.metrics_types import MetricsResult, RelevanceJudgment

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """Computes standard IR metrics for ALMA retrieval quality.

    Two modes:
    1. Ground truth: User provides RelevanceJudgments, compute standard metrics.
    2. Outcome-based: Use task outcomes as implicit relevance (no manual labeling).
    """

    def __init__(self, k_values: Optional[List[int]] = None) -> None:
        """Initialize metrics calculator.

        Args:
            k_values: K values for @K metrics. Default: [1, 3, 5, 10].
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        judgments: List[RelevanceJudgment],
        rankings: Dict[str, List[str]],
    ) -> MetricsResult:
        """Evaluate retrieval quality using ground truth judgments.

        Args:
            judgments: List of relevance judgments (query_id, memory_id, relevance).
            rankings: Dict of query_id -> list of retrieved memory_ids (ordered by rank).

        Returns:
            MetricsResult with all computed metrics.
        """
        if not judgments or not rankings:
            return MetricsResult(k_values=self.k_values)

        # Group judgments by query
        qrels: Dict[str, Dict[str, int]] = {}
        for j in judgments:
            if j.query_id not in qrels:
                qrels[j.query_id] = {}
            qrels[j.query_id][j.memory_id] = j.relevance

        # Only evaluate queries that have both judgments and rankings
        query_ids = set(qrels.keys()) & set(rankings.keys())
        if not query_ids:
            return MetricsResult(k_values=self.k_values)

        # Compute per-query metrics
        mrr_values = []
        ap_values = []
        ndcg_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        recall_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        precision_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}

        for qid in query_ids:
            ranked = rankings[qid]
            rels = qrels[qid]

            # MRR
            mrr_values.append(self._reciprocal_rank(ranked, rels))

            # Average Precision
            ap_values.append(self._average_precision(ranked, rels))

            # @K metrics
            for k in self.k_values:
                ndcg_at_k[k].append(self._ndcg_at_k(ranked, rels, k))
                recall_at_k[k].append(self._recall_at_k(ranked, rels, k))
                precision_at_k[k].append(self._precision_at_k(ranked, rels, k))

        num_queries = len(query_ids)

        return MetricsResult(
            mrr=sum(mrr_values) / num_queries,
            ndcg_at_k={k: sum(v) / num_queries for k, v in ndcg_at_k.items()},
            recall_at_k={k: sum(v) / num_queries for k, v in recall_at_k.items()},
            precision_at_k={k: sum(v) / num_queries for k, v in precision_at_k.items()},
            map_score=sum(ap_values) / num_queries,
            num_queries=num_queries,
            k_values=self.k_values,
        )

    def evaluate_from_feedback(
        self,
        tracker: RetrievalFeedbackTracker,
        agent: Optional[str] = None,
    ) -> MetricsResult:
        """Evaluate retrieval quality using outcome-based feedback.

        Uses task success/failure as implicit relevance: memories
        retrieved in successful tasks are treated as relevant.

        Args:
            tracker: Feedback tracker with recorded data.
            agent: Optional agent filter.

        Returns:
            MetricsResult with outcome-based metrics.
        """
        effectiveness = tracker.get_memory_effectiveness(agent)
        if not effectiveness:
            return MetricsResult(k_values=self.k_values)

        # Convert effectiveness to pseudo-judgments
        # Memories with high success correlation are "relevant"
        judgments = []
        for mid, eff in effectiveness.items():
            if eff.times_retrieved >= 2:
                relevance = 0
                if eff.success_correlation > 0.7:
                    relevance = 2  # Highly relevant
                elif eff.success_correlation > 0.4:
                    relevance = 1  # Relevant
                judgments.append(
                    RelevanceJudgment(
                        query_id="__feedback__",
                        memory_id=mid,
                        relevance=relevance,
                    )
                )

        if not judgments:
            return MetricsResult(k_values=self.k_values)

        # Build a single ranking from effectiveness scores
        sorted_memories = sorted(
            effectiveness.items(),
            key=lambda x: -x[1].success_correlation,
        )
        ranking = {"__feedback__": [mid for mid, _ in sorted_memories]}

        return self.evaluate(judgments, ranking)

    # ---- Pure metric functions ----

    @staticmethod
    def _reciprocal_rank(ranked: List[str], rels: Dict[str, int]) -> float:
        """Compute Reciprocal Rank for a single query.

        RR = 1/rank of the first relevant document.
        """
        for i, doc_id in enumerate(ranked):
            if rels.get(doc_id, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _average_precision(ranked: List[str], rels: Dict[str, int]) -> float:
        """Compute Average Precision for a single query.

        AP = (1/R) * sum(P@k * rel(k)) for all k.
        """
        total_relevant = sum(1 for v in rels.values() if v > 0)
        if total_relevant == 0:
            return 0.0

        hits = 0
        sum_precision = 0.0
        for i, doc_id in enumerate(ranked):
            if rels.get(doc_id, 0) > 0:
                hits += 1
                sum_precision += hits / (i + 1)

        return sum_precision / total_relevant

    @staticmethod
    def _ndcg_at_k(ranked: List[str], rels: Dict[str, int], k: int) -> float:
        """Compute NDCG@K for a single query.

        NDCG = DCG / IDCG where DCG = sum(rel_i / log2(i+2)).
        """
        # DCG
        dcg = 0.0
        for i in range(min(k, len(ranked))):
            rel = rels.get(ranked[i], 0)
            dcg += rel / math.log2(i + 2)

        # Ideal DCG (perfect ranking)
        ideal_rels = sorted(rels.values(), reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_rels))):
            idcg += ideal_rels[i] / math.log2(i + 2)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _recall_at_k(ranked: List[str], rels: Dict[str, int], k: int) -> float:
        """Compute Recall@K for a single query.

        Recall@K = |relevant in top-K| / |total relevant|.
        """
        total_relevant = sum(1 for v in rels.values() if v > 0)
        if total_relevant == 0:
            return 0.0

        hits = sum(1 for doc_id in ranked[:k] if rels.get(doc_id, 0) > 0)
        return hits / total_relevant

    @staticmethod
    def _precision_at_k(ranked: List[str], rels: Dict[str, int], k: int) -> float:
        """Compute Precision@K for a single query.

        Precision@K = |relevant in top-K| / K.
        """
        if k == 0:
            return 0.0

        hits = sum(1 for doc_id in ranked[:k] if rels.get(doc_id, 0) > 0)
        return hits / k
