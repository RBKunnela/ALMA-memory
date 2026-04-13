"""
LongMemEval Metrics

Standard retrieval metrics for evaluating memory system performance:
- Recall@K: fraction of questions where correct answer is in top K results
- NDCG@K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- Precision@K: precision at cutoff K

All metrics operate on a list of per-question results, where each result
tracks which retrieved items matched the ground truth.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class QuestionResult:
    """
    Result for a single benchmark question.

    Attributes:
        question_id: Unique identifier for the question
        question_type: Category of the question (e.g. "single-session-user")
        correct_ids: Set of ground-truth session/document IDs that contain the answer
        ranked_ids: Ordered list of retrieved document IDs (most relevant first)
        retrieval_time_ms: Time taken for this retrieval in milliseconds
    """

    question_id: str
    question_type: str
    correct_ids: Set[str]
    ranked_ids: List[str]
    retrieval_time_ms: float = 0.0


@dataclass
class BenchmarkMetrics:
    """
    Aggregated metrics across all benchmark questions.

    Attributes:
        recall_at_k: Maps K -> recall score (fraction of questions with hit in top K)
        ndcg_at_k: Maps K -> NDCG score
        mrr: Mean Reciprocal Rank
        precision_at_k: Maps K -> precision score
        per_type: Maps question_type -> BenchmarkMetrics for breakdown
        total_questions: Number of questions evaluated
        total_time_s: Total benchmark time in seconds
    """

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    per_type: Dict[str, "BenchmarkMetrics"] = field(default_factory=dict)
    total_questions: int = 0
    total_time_s: float = 0.0


def _dcg(relevances: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at rank K.

    Uses the standard formula: sum(rel_i / log2(i + 2)) for i in 0..k-1.

    Args:
        relevances: List of relevance scores (1.0 for relevant, 0.0 otherwise)
        k: Cutoff rank

    Returns:
        DCG score
    """
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def recall_at_k(results: List[QuestionResult], k: int) -> float:
    """
    Recall@K: fraction of questions where at least one correct ID is in the top K.

    This is the primary metric for LongMemEval -- it measures whether the
    memory system can surface the relevant session in the top K results.

    Args:
        results: List of per-question results
        k: Number of top results to consider

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    hits = 0
    for r in results:
        top_k_ids = set(r.ranked_ids[:k])
        if top_k_ids & r.correct_ids:
            hits += 1

    return hits / len(results)


def recall_all_at_k(results: List[QuestionResult], k: int) -> float:
    """
    Recall-All@K: fraction of questions where ALL correct IDs are in the top K.

    Stricter than recall_at_k -- requires every ground-truth session to be retrieved.

    Args:
        results: List of per-question results
        k: Number of top results to consider

    Returns:
        Recall-All score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    hits = 0
    for r in results:
        top_k_ids = set(r.ranked_ids[:k])
        if r.correct_ids.issubset(top_k_ids):
            hits += 1

    return hits / len(results)


def ndcg_at_k(results: List[QuestionResult], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K.

    Measures ranking quality -- higher score means correct results appear
    earlier in the ranked list.

    Args:
        results: List of per-question results
        k: Cutoff rank

    Returns:
        NDCG score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    scores = []
    for r in results:
        relevances = [
            1.0 if doc_id in r.correct_ids else 0.0 for doc_id in r.ranked_ids[:k]
        ]
        ideal = sorted(relevances, reverse=True)
        idcg = _dcg(ideal, k)
        if idcg == 0:
            scores.append(0.0)
        else:
            scores.append(_dcg(relevances, k) / idcg)

    return sum(scores) / len(scores)


def mean_reciprocal_rank(results: List[QuestionResult]) -> float:
    """
    Mean Reciprocal Rank: average of 1/rank of the first correct result.

    MRR rewards systems that place the correct answer as high as possible.

    Args:
        results: List of per-question results

    Returns:
        MRR score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    rr_sum = 0.0
    for r in results:
        for rank, doc_id in enumerate(r.ranked_ids, start=1):
            if doc_id in r.correct_ids:
                rr_sum += 1.0 / rank
                break

    return rr_sum / len(results)


def precision_at_k(results: List[QuestionResult], k: int) -> float:
    """
    Precision@K: average fraction of top K results that are correct.

    Args:
        results: List of per-question results
        k: Cutoff rank

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    prec_sum = 0.0
    for r in results:
        top_k_ids = r.ranked_ids[:k]
        if not top_k_ids:
            continue
        hits = sum(1 for doc_id in top_k_ids if doc_id in r.correct_ids)
        prec_sum += hits / len(top_k_ids)

    return prec_sum / len(results)


def compute_all_metrics(
    results: List[QuestionResult],
    ks: Optional[List[int]] = None,
    total_time_s: float = 0.0,
) -> BenchmarkMetrics:
    """
    Compute all metrics for a set of benchmark results.

    Calculates R@K, NDCG@K, Precision@K for each K, plus MRR.
    Also breaks down metrics by question_type.

    Args:
        results: List of per-question results
        ks: List of K values to evaluate (default: [1, 3, 5, 10, 30, 50])
        total_time_s: Total benchmark wall time in seconds

    Returns:
        BenchmarkMetrics with all computed metrics and per-type breakdown
    """
    if ks is None:
        ks = [1, 3, 5, 10, 30, 50]

    metrics = BenchmarkMetrics(
        total_questions=len(results),
        total_time_s=total_time_s,
    )

    for k in ks:
        metrics.recall_at_k[k] = recall_at_k(results, k)
        metrics.ndcg_at_k[k] = ndcg_at_k(results, k)
        metrics.precision_at_k[k] = precision_at_k(results, k)

    metrics.mrr = mean_reciprocal_rank(results)

    # Per-type breakdown
    type_groups: Dict[str, List[QuestionResult]] = {}
    for r in results:
        type_groups.setdefault(r.question_type, []).append(r)

    for qtype, type_results in sorted(type_groups.items()):
        type_metrics = BenchmarkMetrics(
            total_questions=len(type_results),
        )
        for k in ks:
            type_metrics.recall_at_k[k] = recall_at_k(type_results, k)
            type_metrics.ndcg_at_k[k] = ndcg_at_k(type_results, k)
            type_metrics.precision_at_k[k] = precision_at_k(type_results, k)
        type_metrics.mrr = mean_reciprocal_rank(type_results)
        metrics.per_type[qtype] = type_metrics

    return metrics


def format_results(metrics: BenchmarkMetrics, title: str = "ALMA") -> str:
    """
    Format benchmark metrics as a human-readable report.

    Args:
        metrics: Computed benchmark metrics
        title: Title for the report header

    Returns:
        Formatted string report
    """
    lines = []
    sep = "=" * 64
    thin_sep = "-" * 64

    lines.append(f"\n{sep}")
    lines.append(f"  {title} x LongMemEval Benchmark Results")
    lines.append(sep)
    lines.append(
        f"  Questions: {metrics.total_questions}    Time: {metrics.total_time_s:.1f}s"
    )
    if metrics.total_questions > 0 and metrics.total_time_s > 0:
        lines.append(
            f"  Per question: {metrics.total_time_s / metrics.total_questions:.2f}s"
        )
    lines.append(thin_sep)

    # Main metrics table
    lines.append("\n  SESSION-LEVEL METRICS:")
    lines.append(f"  {'K':>4}  {'Recall@K':>10}  {'NDCG@K':>10}  {'Prec@K':>10}")
    lines.append(f"  {'---':>4}  {'--------':>10}  {'------':>10}  {'------':>10}")

    for k in sorted(metrics.recall_at_k.keys()):
        r = metrics.recall_at_k.get(k, 0.0)
        n = metrics.ndcg_at_k.get(k, 0.0)
        p = metrics.precision_at_k.get(k, 0.0)
        lines.append(f"  {k:>4}  {r:>10.3f}  {n:>10.3f}  {p:>10.3f}")

    lines.append(f"\n  MRR: {metrics.mrr:.3f}")

    # Per-type breakdown
    if metrics.per_type:
        lines.append(f"\n{thin_sep}")
        lines.append("  PER-TYPE BREAKDOWN (Recall@5 / Recall@10):")
        lines.append(f"  {'Type':<35} {'R@5':>8} {'R@10':>8} {'Count':>6}")
        lines.append(f"  {'-' * 35:<35} {'---':>8} {'---':>8} {'---':>6}")
        for qtype, tm in sorted(metrics.per_type.items()):
            r5 = tm.recall_at_k.get(5, 0.0)
            r10 = tm.recall_at_k.get(10, 0.0)
            lines.append(
                f"  {qtype:<35} {r5:>8.3f} {r10:>8.3f} {tm.total_questions:>6}"
            )

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)
