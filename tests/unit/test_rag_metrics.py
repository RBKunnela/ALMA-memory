"""Tests for ALMA Retrieval Metrics -- standard IR quality measurement."""

import math

from alma.rag.metrics import RetrievalMetrics
from alma.rag.metrics_types import MetricsHistory, MetricsResult, RelevanceJudgment


# ── Reciprocal Rank ──────────────────────────────────────────────────


class TestReciprocalRank:
    def test_first_doc_relevant(self):
        rr = RetrievalMetrics._reciprocal_rank(
            ["d1", "d2", "d3"],
            {"d1": 2, "d2": 0, "d3": 1},
        )
        assert rr == 1.0

    def test_second_doc_relevant(self):
        rr = RetrievalMetrics._reciprocal_rank(
            ["d1", "d2", "d3"],
            {"d2": 1, "d3": 1},
        )
        assert rr == 0.5  # 1/2

    def test_third_doc_relevant(self):
        rr = RetrievalMetrics._reciprocal_rank(
            ["d1", "d2", "d3"],
            {"d3": 1},
        )
        assert abs(rr - 1.0 / 3) < 1e-9

    def test_no_relevant_docs(self):
        rr = RetrievalMetrics._reciprocal_rank(
            ["d1", "d2"],
            {"d1": 0, "d2": 0},
        )
        assert rr == 0.0

    def test_empty_ranking(self):
        rr = RetrievalMetrics._reciprocal_rank([], {"d1": 1})
        assert rr == 0.0

    def test_empty_rels(self):
        rr = RetrievalMetrics._reciprocal_rank(["d1", "d2"], {})
        assert rr == 0.0


# ── Average Precision ────────────────────────────────────────────────


class TestAveragePrecision:
    def test_perfect_ranking(self):
        """All relevant docs at the top."""
        ap = RetrievalMetrics._average_precision(
            ["d1", "d2", "d3", "d4"],
            {"d1": 1, "d2": 1},
        )
        # P@1 = 1/1, P@2 = 2/2 -> AP = (1 + 1) / 2 = 1.0
        assert ap == 1.0

    def test_worst_ranking(self):
        """Relevant docs at the bottom."""
        ap = RetrievalMetrics._average_precision(
            ["d3", "d4", "d1", "d2"],
            {"d1": 1, "d2": 1},
        )
        # d1 at pos 3: P@3 = 1/3, d2 at pos 4: P@4 = 2/4 = 0.5
        # AP = (1/3 + 0.5) / 2 = 0.4166...
        expected = (1 / 3 + 0.5) / 2
        assert abs(ap - expected) < 1e-9

    def test_mixed_ranking(self):
        """One relevant, one irrelevant, one relevant."""
        ap = RetrievalMetrics._average_precision(
            ["d1", "d2", "d3"],
            {"d1": 1, "d3": 1},
        )
        # d1 at pos 1: P@1 = 1/1, d3 at pos 3: P@3 = 2/3
        # AP = (1 + 2/3) / 2 = 5/6
        expected = (1.0 + 2.0 / 3) / 2
        assert abs(ap - expected) < 1e-9

    def test_no_relevant_docs(self):
        ap = RetrievalMetrics._average_precision(
            ["d1", "d2"],
            {"d1": 0, "d2": 0},
        )
        assert ap == 0.0

    def test_empty_ranking(self):
        ap = RetrievalMetrics._average_precision([], {"d1": 1})
        assert ap == 0.0


# ── NDCG@K ───────────────────────────────────────────────────────────


class TestNDCGAtK:
    def test_perfect_ranking_k3(self):
        """Ideal ordering: highest relevance first."""
        ndcg = RetrievalMetrics._ndcg_at_k(
            ["d1", "d2", "d3"],
            {"d1": 2, "d2": 1, "d3": 0},
            k=3,
        )
        # DCG = 2/log2(2) + 1/log2(3) + 0/log2(4) = 2/1 + 1/1.585 = 2.631
        # IDCG = same (already perfect) -> NDCG = 1.0
        assert abs(ndcg - 1.0) < 1e-9

    def test_reversed_ranking_k2(self):
        """Worst ordering: lowest relevance first."""
        ndcg = RetrievalMetrics._ndcg_at_k(
            ["d3", "d1"],
            {"d1": 2, "d3": 0},
            k=2,
        )
        # DCG = 0/log2(2) + 2/log2(3) = 0 + 1.2619 = 1.2619
        # IDCG = 2/log2(2) + 0/log2(3) = 2.0
        dcg = 2 / math.log2(3)
        idcg = 2 / math.log2(2)
        expected = dcg / idcg
        assert abs(ndcg - expected) < 1e-9

    def test_k_larger_than_ranking(self):
        """K exceeds the number of ranked docs."""
        ndcg = RetrievalMetrics._ndcg_at_k(
            ["d1"],
            {"d1": 2},
            k=10,
        )
        # DCG = 2/log2(2) = 2.0, IDCG = 2.0 -> NDCG = 1.0
        assert abs(ndcg - 1.0) < 1e-9

    def test_no_relevant_docs(self):
        ndcg = RetrievalMetrics._ndcg_at_k(
            ["d1", "d2"],
            {"d1": 0, "d2": 0},
            k=2,
        )
        assert ndcg == 0.0

    def test_graded_relevance(self):
        """Documents with different relevance levels."""
        ndcg = RetrievalMetrics._ndcg_at_k(
            ["d2", "d1"],
            {"d1": 2, "d2": 1},
            k=2,
        )
        # DCG = 1/log2(2) + 2/log2(3) = 1.0 + 1.2619 = 2.2619
        # IDCG = 2/log2(2) + 1/log2(3) = 2.0 + 0.6309 = 2.6309
        dcg = 1 / math.log2(2) + 2 / math.log2(3)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert abs(ndcg - expected) < 1e-9


# ── Recall@K ─────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        recall = RetrievalMetrics._recall_at_k(
            ["d1", "d2", "d3"],
            {"d1": 1, "d2": 1},
            k=3,
        )
        assert recall == 1.0

    def test_partial_recall(self):
        recall = RetrievalMetrics._recall_at_k(
            ["d1", "d3", "d2"],
            {"d1": 1, "d2": 1},
            k=1,
        )
        assert recall == 0.5

    def test_zero_recall(self):
        recall = RetrievalMetrics._recall_at_k(
            ["d3", "d4"],
            {"d1": 1, "d2": 1},
            k=2,
        )
        assert recall == 0.0

    def test_no_relevant_docs(self):
        recall = RetrievalMetrics._recall_at_k(
            ["d1", "d2"],
            {"d1": 0},
            k=2,
        )
        assert recall == 0.0

    def test_k_zero(self):
        recall = RetrievalMetrics._recall_at_k(
            ["d1"],
            {"d1": 1},
            k=0,
        )
        assert recall == 0.0


# ── Precision@K ──────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_all_relevant(self):
        precision = RetrievalMetrics._precision_at_k(
            ["d1", "d2"],
            {"d1": 1, "d2": 2},
            k=2,
        )
        assert precision == 1.0

    def test_half_relevant(self):
        precision = RetrievalMetrics._precision_at_k(
            ["d1", "d2", "d3", "d4"],
            {"d1": 1, "d3": 1},
            k=4,
        )
        assert precision == 0.5

    def test_none_relevant(self):
        precision = RetrievalMetrics._precision_at_k(
            ["d3", "d4"],
            {"d1": 1, "d2": 1},
            k=2,
        )
        assert precision == 0.0

    def test_k_zero(self):
        precision = RetrievalMetrics._precision_at_k(
            ["d1"],
            {"d1": 1},
            k=0,
        )
        assert precision == 0.0

    def test_k_exceeds_ranking(self):
        """K is larger than the ranked list."""
        precision = RetrievalMetrics._precision_at_k(
            ["d1"],
            {"d1": 1},
            k=5,
        )
        # 1 hit in 5 slots = 0.2
        assert abs(precision - 0.2) < 1e-9


# ── Full evaluate() ──────────────────────────────────────────────────


class TestEvaluate:
    def test_empty_inputs(self):
        metrics = RetrievalMetrics(k_values=[1, 3])
        result = metrics.evaluate([], {})
        assert result.num_queries == 0
        assert result.mrr == 0.0

    def test_no_overlapping_queries(self):
        metrics = RetrievalMetrics(k_values=[1])
        result = metrics.evaluate(
            [RelevanceJudgment(query_id="q1", memory_id="d1", relevance=1)],
            {"q2": ["d1"]},  # Different query
        )
        assert result.num_queries == 0

    def test_single_query_perfect(self):
        metrics = RetrievalMetrics(k_values=[1, 3])
        judgments = [
            RelevanceJudgment(query_id="q1", memory_id="d1", relevance=2),
            RelevanceJudgment(query_id="q1", memory_id="d2", relevance=1),
        ]
        rankings = {"q1": ["d1", "d2", "d3"]}

        result = metrics.evaluate(judgments, rankings)

        assert result.num_queries == 1
        assert result.mrr == 1.0  # d1 is first
        assert result.map_score == 1.0  # Perfect ranking
        assert abs(result.ndcg_at_k[1] - 1.0) < 1e-9
        assert abs(result.ndcg_at_k[3] - 1.0) < 1e-9
        assert result.recall_at_k[1] == 0.5  # 1 of 2 relevant in top 1
        assert result.recall_at_k[3] == 1.0  # Both in top 3
        assert result.k_values == [1, 3]

    def test_multi_query_average(self):
        """MRR should average across queries."""
        metrics = RetrievalMetrics(k_values=[3])
        judgments = [
            RelevanceJudgment(query_id="q1", memory_id="d1", relevance=1),
            RelevanceJudgment(query_id="q2", memory_id="d3", relevance=1),
        ]
        rankings = {
            "q1": ["d1", "d2", "d3"],  # RR = 1.0
            "q2": ["d1", "d2", "d3"],  # RR = 1/3
        }

        result = metrics.evaluate(judgments, rankings)

        assert result.num_queries == 2
        expected_mrr = (1.0 + 1 / 3) / 2
        assert abs(result.mrr - expected_mrr) < 1e-9

    def test_custom_k_values(self):
        metrics = RetrievalMetrics(k_values=[2, 7])
        judgments = [
            RelevanceJudgment(query_id="q1", memory_id="d1", relevance=1),
        ]
        rankings = {"q1": ["d1"]}

        result = metrics.evaluate(judgments, rankings)

        assert 2 in result.ndcg_at_k
        assert 7 in result.ndcg_at_k
        assert 2 in result.recall_at_k
        assert 7 in result.recall_at_k

    def test_to_dict(self):
        result = MetricsResult(
            mrr=0.75,
            ndcg_at_k={1: 0.5, 3: 0.8},
            map_score=0.6,
            num_queries=10,
            k_values=[1, 3],
        )
        d = result.to_dict()
        assert d["mrr"] == 0.75
        assert d["ndcg_at_k"] == {1: 0.5, 3: 0.8}
        assert d["num_queries"] == 10
        assert "timestamp" in d


# ── MetricsHistory ───────────────────────────────────────────────────


class TestMetricsHistory:
    def test_empty_history(self):
        history = MetricsHistory()
        assert history.latest is None
        assert history.trend("mrr") == []

    def test_add_and_latest(self):
        history = MetricsHistory(agent="test-agent")
        r1 = MetricsResult(mrr=0.5, num_queries=5)
        r2 = MetricsResult(mrr=0.8, num_queries=10)
        history.add(r1)
        history.add(r2)

        assert history.latest is not None
        assert history.latest.mrr == 0.8

    def test_trend(self):
        history = MetricsHistory()
        for mrr_val in [0.3, 0.5, 0.6, 0.7, 0.9]:
            history.add(MetricsResult(mrr=mrr_val))

        trend = history.trend("mrr", last_n=3)
        assert trend == [0.6, 0.7, 0.9]

    def test_trend_all(self):
        history = MetricsHistory()
        history.add(MetricsResult(mrr=0.4, map_score=0.3))
        history.add(MetricsResult(mrr=0.6, map_score=0.5))

        trend = history.trend("map_score", last_n=10)
        assert trend == [0.3, 0.5]


# ── evaluate_from_feedback() ─────────────────────────────────────────


class TestEvaluateFromFeedback:
    def test_empty_feedback(self):
        """No effectiveness data -> empty result."""
        from unittest.mock import MagicMock

        from alma.rag.feedback import RetrievalFeedbackTracker
        from alma.storage.base import StorageBackend

        storage = MagicMock(spec=StorageBackend)
        storage.get_domain_knowledge.return_value = []
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        metrics = RetrievalMetrics(k_values=[1, 3])
        result = metrics.evaluate_from_feedback(tracker)

        assert result.num_queries == 0
        assert result.mrr == 0.0

    def test_feedback_with_data(self):
        """Memories with effectiveness data produce a metrics result."""
        import json
        from unittest.mock import MagicMock

        from alma.rag.feedback import (
            FEEDBACK_DOMAIN,
            FEEDBACK_SOURCE,
            RetrievalFeedbackTracker,
        )
        from alma.storage.base import StorageBackend
        from alma.types import DomainKnowledge

        storage = MagicMock(spec=StorageBackend)

        # Create feedback records: mem_a has 100% success (5/5),
        # mem_b has 40% success (2/5)
        records = []
        for i in range(5):
            records.append(
                DomainKnowledge(
                    id=f"rf_{i}",
                    agent="agent",
                    project_id="proj-1",
                    domain=FEEDBACK_DOMAIN,
                    fact=json.dumps(
                        {
                            "type": "retrieval_feedback",
                            "retrieval_record_id": f"rr_{i}",
                            "outcome_id": f"out_{i}",
                            "success": True,
                            "memory_ids": ["mem_a", "mem_b"],
                            "helpful_memory_ids": [],
                            "unhelpful_memory_ids": [],
                        }
                    ),
                    source=FEEDBACK_SOURCE,
                )
            )
        # Add 3 failures for mem_b only
        for i in range(5, 8):
            records.append(
                DomainKnowledge(
                    id=f"rf_{i}",
                    agent="agent",
                    project_id="proj-1",
                    domain=FEEDBACK_DOMAIN,
                    fact=json.dumps(
                        {
                            "type": "retrieval_feedback",
                            "retrieval_record_id": f"rr_{i}",
                            "outcome_id": f"out_{i}",
                            "success": False,
                            "memory_ids": ["mem_b"],
                            "helpful_memory_ids": [],
                            "unhelpful_memory_ids": [],
                        }
                    ),
                    source=FEEDBACK_SOURCE,
                )
            )
        storage.get_domain_knowledge.return_value = records
        tracker = RetrievalFeedbackTracker(storage=storage, project_id="proj-1")

        metrics = RetrievalMetrics(k_values=[1, 3])
        result = metrics.evaluate_from_feedback(tracker)

        # We have 2 memories, both retrieved >= 2 times
        # mem_a: 5/5 success -> correlation 1.0 -> relevance 2
        # mem_b: 5/8 success -> correlation 0.625 -> relevance 1
        assert result.num_queries == 1
        assert result.mrr > 0  # At least one relevant doc found
