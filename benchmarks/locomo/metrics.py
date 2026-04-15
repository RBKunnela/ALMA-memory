"""
LoCoMo Metrics

Reuses the retrieval primitives from :mod:`benchmarks.longmemeval.metrics`
(recall@k, MRR, NDCG) but operates at turn-id granularity and aggregates
results across the five LoCoMo QA categories.

Adversarial special-case: for category ``"adversarial"`` questions, the
*correct* outcome is an empty retrieval (refusal detection). A retrieved
list that contains no turns scores 1.0; a non-empty list scores 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from benchmarks.locomo.dataset import ALL_CATEGORIES
from benchmarks.longmemeval.metrics import (
    QuestionResult,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "LoCoMoMetrics",
    "LoCoMoQAResult",
    "adversarial_success",
]


@dataclass
class LoCoMoQAResult:
    """Per-QA result: retrieved turn IDs vs. evidence turn IDs."""

    qa_id: str
    category: str
    question: str
    evidence_turn_ids: List[str]
    retrieved_turn_ids: List[str]
    retrieval_time_ms: float = 0.0

    def to_question_result(self) -> QuestionResult:
        """Adapt to the :class:`QuestionResult` shape used by LongMemEval."""
        return QuestionResult(
            question_id=self.qa_id,
            question_type=self.category,
            correct_ids=set(self.evidence_turn_ids),
            ranked_ids=list(self.retrieved_turn_ids),
            retrieval_time_ms=self.retrieval_time_ms,
        )


def adversarial_success(retrieved_turn_ids: Sequence[str]) -> float:
    """
    Score an adversarial question.

    Adversarial questions have no supporting evidence in the conversation;
    the system should refuse / return nothing. An empty retrieval therefore
    counts as success.

    Args:
        retrieved_turn_ids: Ranked list of turn IDs the system returned.

    Returns:
        1.0 if the retrieval is empty, 0.0 otherwise.
    """
    return 1.0 if not list(retrieved_turn_ids) else 0.0


@dataclass
class LoCoMoMetrics:
    """Aggregate metrics computed from a list of :class:`LoCoMoQAResult`."""

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    adversarial_refusal_rate: float = 0.0
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_qa: int = 0
    total_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieval(
        self,
        retrieved_ids: Sequence[str],
        evidence_ids: Sequence[str],
        k_values: Iterable[int] = (1, 5, 10),
    ) -> Dict[str, float]:
        """
        Score a single retrieval against ground-truth evidence turn IDs.

        Convenience wrapper used by tests and runners that want per-QA numbers
        without constructing a full result list.

        Args:
            retrieved_ids: Ranked turn IDs the retrieval system returned.
            evidence_ids: Ground-truth evidence turn IDs for the question.
            k_values: Cutoffs for ``recall@k`` and ``ndcg@k``.

        Returns:
            Dict with keys ``recall@{k}``, ``ndcg@{k}``, and ``mrr``.
        """
        qr = QuestionResult(
            question_id="_single",
            question_type="_single",
            correct_ids=set(evidence_ids),
            ranked_ids=list(retrieved_ids),
        )
        out: Dict[str, float] = {}
        for k in k_values:
            out[f"recall@{k}"] = recall_at_k([qr], k)
            out[f"ndcg@{k}"] = ndcg_at_k([qr], k)
        out["mrr"] = mean_reciprocal_rank([qr])
        return out

    def aggregate_by_category(
        self,
        per_qa_results: Sequence[LoCoMoQAResult],
        k_values: Iterable[int] = (1, 5, 10),
    ) -> "LoCoMoMetrics":
        """
        Aggregate a list of per-QA results into overall + per-category metrics.

        Non-adversarial categories use standard recall/NDCG/MRR. The
        ``adversarial`` category bucket reports only ``refusal_rate`` since
        its "correct" answer is an empty retrieval.

        Args:
            per_qa_results: Per-QA retrieval results.
            k_values: Cutoffs for retrieval metrics.

        Returns:
            ``self``, populated with aggregated metrics for convenience.
        """
        k_list = list(k_values)
        self.total_qa = len(per_qa_results)

        # Partition by category
        buckets: Dict[str, List[LoCoMoQAResult]] = {c: [] for c in ALL_CATEGORIES}
        for qa in per_qa_results:
            buckets.setdefault(qa.category, []).append(qa)

        # Overall non-adversarial numbers
        non_adv_qr = [
            qa.to_question_result()
            for qa in per_qa_results
            if qa.category != "adversarial"
        ]
        if non_adv_qr:
            for k in k_list:
                self.recall_at_k[k] = recall_at_k(non_adv_qr, k)
                self.ndcg_at_k[k] = ndcg_at_k(non_adv_qr, k)
            self.mrr = mean_reciprocal_rank(non_adv_qr)

        # Adversarial refusal rate
        adv = buckets.get("adversarial", [])
        if adv:
            successes = sum(adversarial_success(qa.retrieved_turn_ids) for qa in adv)
            self.adversarial_refusal_rate = successes / len(adv)

        # Per-category breakdown
        per_cat: Dict[str, Dict[str, float]] = {}
        for cat, items in buckets.items():
            if not items:
                continue
            cat_block: Dict[str, float] = {"count": float(len(items))}
            if cat == "adversarial":
                cat_block["refusal_rate"] = sum(
                    adversarial_success(qa.retrieved_turn_ids) for qa in items
                ) / len(items)
            else:
                qrs = [qa.to_question_result() for qa in items]
                for k in k_list:
                    cat_block[f"recall@{k}"] = recall_at_k(qrs, k)
                    cat_block[f"ndcg@{k}"] = ndcg_at_k(qrs, k)
                cat_block["mrr"] = mean_reciprocal_rank(qrs)
            per_cat[cat] = cat_block

        self.per_category = per_cat
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Serialise metrics to a JSON-safe dict."""
        return {
            "total_qa": self.total_qa,
            "total_time_s": self.total_time_s,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
            "mrr": self.mrr,
            "adversarial_refusal_rate": self.adversarial_refusal_rate,
            "per_category": self.per_category,
        }
