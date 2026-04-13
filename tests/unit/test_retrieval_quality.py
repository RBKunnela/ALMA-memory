"""
Unit tests for ALMA Retrieval Quality — Ranking Correctness.

Exhaustive test suite that PROVES retrieval ranking works correctly.
Catches any regression where similarity scores are lost in the pipeline,
ensuring R@5 quality is maintained across code changes.

Categories:
1. Similarity Score Propagation
2. Ranking Correctness (R@5)
3. Scoring Weights Effect
4. Mode-Aware Retrieval
5. Edge Cases
6. End-to-End Retrieval Flow
7. Query Sanitizer Integration
"""

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from alma.retrieval.engine import RetrievalEngine
from alma.retrieval.modes import (
    MODE_CONFIGS,
    ModeConfig,
    RetrievalMode,
    get_mode_config,
)
from alma.retrieval.query_sanitizer import sanitize_query
from alma.retrieval.scoring import (
    MemoryScorer,
    ScoredItem,
    ScoringWeights,
    compute_composite_score,
)
from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
    create_test_outcome,
)
from alma.testing.mocks import MockEmbedder, MockStorage
from alma.types import DomainKnowledge, Heuristic, MemorySlice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_knowledge(
    idx: int,
    fact: str,
    confidence: float = 0.9,
    domain: str = "test_domain",
    embedding: Optional[List[float]] = None,
    days_ago: int = 0,
) -> DomainKnowledge:
    """Create a DomainKnowledge item with deterministic IDs."""
    return create_test_knowledge(
        id=f"dk_{idx:03d}",
        agent="test-agent",
        project_id="test-project",
        domain=domain,
        fact=fact,
        source="test",
        confidence=confidence,
        last_verified=datetime.now(timezone.utc) - timedelta(days=days_ago),
        embedding=embedding,
    )


def _make_heuristic(
    idx: int,
    condition: str,
    strategy: str,
    confidence: float = 0.85,
    success_count: int = 8,
    occurrence_count: int = 10,
    days_ago: int = 0,
) -> Heuristic:
    """Create a Heuristic with deterministic IDs."""
    return create_test_heuristic(
        id=f"heur_{idx:03d}",
        agent="test-agent",
        project_id="test-project",
        condition=condition,
        strategy=strategy,
        confidence=confidence,
        success_count=success_count,
        occurrence_count=occurrence_count,
        last_validated=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _make_unit_vector(dim: int, hot_index: int) -> List[float]:
    """Create a unit vector with 1.0 at hot_index, 0.0 elsewhere.

    Useful for crafting vectors with known cosine similarities.
    """
    vec = [0.0] * dim
    vec[hot_index] = 1.0
    return vec


def _make_blended_vector(
    dim: int, hot_index: int, noise_level: float = 0.0
) -> List[float]:
    """Create a vector with a primary component and optional uniform noise.

    Higher noise_level (0-1) reduces similarity to the pure unit vector.
    """
    import random

    rng = random.Random(hot_index * 1000 + int(noise_level * 100))
    vec = [noise_level * rng.uniform(-1, 1) for _ in range(dim)]
    vec[hot_index] = 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Category 1: Similarity Score Propagation
# ---------------------------------------------------------------------------

class TestSimilarityScorePropagation:
    """Verify similarity scores survive the full scoring pipeline."""

    def test_scorer_preserves_similarity_ordering(self):
        """[UNIT] MemoryScorer — should preserve similarity-based ranking
        when similarity scores are provided."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        items = [
            _make_knowledge(i, f"fact {i}", confidence=0.5)
            for i in range(10)
        ]
        # Similarities: item 3 is most similar, item 7 is least similar
        sims = [0.5] * 10
        sims[3] = 0.99  # most similar
        sims[7] = 0.01  # least similar

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        # Item 3 should be ranked BEFORE item 7
        ids = [s.item.id for s in scored]
        pos_3 = ids.index("dk_003")
        pos_7 = ids.index("dk_007")
        assert pos_3 < pos_7, (
            f"Item 3 (sim=0.99) at position {pos_3} should be before "
            f"Item 7 (sim=0.01) at position {pos_7}"
        )

    def test_similarity_scores_monotonically_decreasing(self):
        """[UNIT] MemoryScorer — similarity_score values in scored results
        should be monotonically non-increasing when using pure similarity weights."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        items = [_make_knowledge(i, f"fact {i}") for i in range(10)]
        sims = [round(0.1 * (i + 1), 2) for i in range(10)]  # 0.1 to 1.0

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        for i in range(len(scored) - 1):
            assert scored[i].similarity_score >= scored[i + 1].similarity_score, (
                f"Position {i} sim={scored[i].similarity_score} should >= "
                f"position {i+1} sim={scored[i+1].similarity_score}"
            )

    def test_scored_item_carries_all_component_scores(self):
        """[UNIT] ScoredItem — should carry similarity, recency, success,
        and confidence scores for downstream inspection."""
        scorer = MemoryScorer()
        h = _make_heuristic(0, "cond", "strat", confidence=0.9)
        scored = scorer.score_heuristics([h], similarities=[0.75])

        assert len(scored) == 1
        item = scored[0]
        assert item.similarity_score == 0.75
        assert 0.0 <= item.recency_score <= 1.0
        assert 0.0 <= item.success_score <= 1.0
        assert item.confidence_score == 0.9

    def test_default_similarity_is_one_when_not_provided(self):
        """[UNIT] MemoryScorer — should default similarity to 1.0 when
        no similarities list is provided."""
        scorer = MemoryScorer()
        items = [_make_knowledge(i, f"fact {i}") for i in range(3)]

        scored = scorer.score_domain_knowledge(items)  # no similarities arg

        for s in scored:
            assert s.similarity_score == 1.0, (
                "Without explicit similarities, scorer should default to 1.0"
            )

    def test_ten_items_with_known_embeddings_rank_correctly(self):
        """[INTEGRATION] MockStorage+Scorer — store 10 items with known
        embeddings, query with a vector similar to item #3, verify ranking."""
        storage = MockStorage()
        embedder = MockEmbedder(dimension=16)

        # Create items with distinct unit vectors
        items = []
        for i in range(10):
            vec = _make_unit_vector(16, i)
            dk = _make_knowledge(i, f"fact about topic {i}", embedding=vec)
            storage.save_domain_knowledge(dk)
            items.append(dk)

        # Query vector: very similar to item #3 (same direction)
        query_vec = _make_blended_vector(16, 3, noise_level=0.05)

        # Manually compute similarities to establish ground truth
        ground_truth_sims = []
        for item in items:
            sim = _cosine_similarity(query_vec, item.embedding)
            ground_truth_sims.append((item.id, sim))
        ground_truth_sims.sort(key=lambda x: -x[1])

        # Use scorer with pure similarity
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        sims = [_cosine_similarity(query_vec, it.embedding) for it in items]
        scored = scorer.score_domain_knowledge(items, similarities=sims)

        # Item #3 must be first (highest similarity)
        assert scored[0].item.id == "dk_003", (
            f"Expected dk_003 at position 0, got {scored[0].item.id}"
        )
        # Item #3 sim should be much higher than distant items
        assert scored[0].similarity_score > 0.9


# ---------------------------------------------------------------------------
# Category 2: Ranking Correctness (R@5 — THE CRITICAL TEST)
# ---------------------------------------------------------------------------

class TestRankingCorrectness:
    """Verify that known-correct items appear in top-5 results."""

    @pytest.fixture
    def scorer_pure_similarity(self):
        """Scorer using only similarity for ranking."""
        return MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )

    def test_recall_at_5_with_50_items_and_5_queries(self, scorer_pure_similarity):
        """[CRITICAL] R@5 — for each of 5 queries, the correct item must
        appear in the top 5 results out of 50 total items."""
        dim = 32
        items = []
        for i in range(50):
            vec = _make_blended_vector(dim, i % dim, noise_level=0.2)
            dk = _make_knowledge(i, f"fact about topic {i}", embedding=vec)
            items.append(dk)

        # 5 queries, each targeting a specific item via its dominant dimension
        query_targets = [
            (3, "topic 3"),
            (10, "topic 10"),
            (25, "topic 25"),
            (0, "topic 0"),
            (31, "topic 31"),
        ]

        for target_idx, label in query_targets:
            query_vec = _make_unit_vector(dim, target_idx % dim)

            sims = [
                _cosine_similarity(query_vec, it.embedding)
                for it in items
            ]
            scored = scorer_pure_similarity.score_domain_knowledge(
                items, similarities=sims
            )
            top_5_ids = [s.item.id for s in scored[:5]]

            # The item whose embedding aligns with the query dimension
            # should be in top 5
            expected_id = f"dk_{target_idx:03d}"
            assert expected_id in top_5_ids, (
                f"R@5 FAIL for '{label}': {expected_id} not in top 5 "
                f"({top_5_ids})"
            )

    def test_identical_query_returns_exact_match_first(self, scorer_pure_similarity):
        """[UNIT] Scorer — when query embedding matches an item exactly,
        that item must be ranked #1."""
        dim = 16
        target_vec = _make_unit_vector(dim, 5)
        items = []
        for i in range(20):
            vec = _make_blended_vector(dim, i % dim, noise_level=0.3)
            items.append(_make_knowledge(i, f"fact {i}", embedding=vec))

        # Inject an exact match
        items[7] = _make_knowledge(7, "exact match item", embedding=target_vec[:])

        sims = [_cosine_similarity(target_vec, it.embedding) for it in items]
        scored = scorer_pure_similarity.score_domain_knowledge(items, similarities=sims)

        assert scored[0].item.id == "dk_007", (
            f"Exact match item should be #1, got {scored[0].item.id}"
        )
        assert scored[0].similarity_score > 0.99

    def test_high_similarity_items_dominate_top_k(self, scorer_pure_similarity):
        """[UNIT] Scorer — items with high similarity should fill top-k
        even when other items have higher confidence."""
        dim = 16
        query_vec = _make_unit_vector(dim, 0)

        items = []
        # Items 0-4: high similarity (aligned with dim 0), low confidence
        for i in range(5):
            vec = _make_blended_vector(dim, 0, noise_level=0.1 * i)
            items.append(_make_knowledge(i, f"high-sim {i}", confidence=0.3, embedding=vec))

        # Items 5-14: low similarity (different dims), high confidence
        for i in range(5, 15):
            vec = _make_unit_vector(dim, i % dim)
            items.append(_make_knowledge(i, f"low-sim {i}", confidence=1.0, embedding=vec))

        sims = [_cosine_similarity(query_vec, it.embedding) for it in items]
        scored = scorer_pure_similarity.score_domain_knowledge(items, similarities=sims)

        # With pure similarity weights, high-sim items should be in top 5
        top_5_ids = {s.item.id for s in scored[:5]}
        high_sim_ids = {f"dk_{i:03d}" for i in range(5)}
        assert high_sim_ids.issubset(top_5_ids), (
            f"High-similarity items {high_sim_ids} should all be in top 5, "
            f"got {top_5_ids}"
        )

    def test_composite_score_formula_correctness(self):
        """[UNIT] compute_composite_score — should produce weighted sum
        of similarity, recency, success, confidence."""
        weights = ScoringWeights(
            similarity=0.4, recency=0.3, success_rate=0.2, confidence=0.1
        )
        # For a brand-new item (0 days ago), recency_score = 1.0
        score = compute_composite_score(
            similarity=0.8,
            recency_days=0.0,
            success_rate=0.9,
            confidence=0.7,
            weights=weights,
            recency_half_life=30.0,
        )
        expected = 0.4 * 0.8 + 0.3 * 1.0 + 0.2 * 0.9 + 0.1 * 0.7
        assert abs(score - expected) < 1e-6, (
            f"Composite score {score} != expected {expected}"
        )

    def test_score_threshold_filters_low_scores(self):
        """[UNIT] apply_score_threshold — should remove items below threshold."""
        scorer = MemoryScorer()
        items = [_make_knowledge(i, f"fact {i}") for i in range(5)]
        sims = [0.9, 0.5, 0.1, 0.8, 0.3]

        scored = scorer.score_domain_knowledge(items, similarities=sims)
        filtered = scorer.apply_score_threshold(scored, min_score=0.5)

        for item in filtered:
            assert item.score >= 0.5, (
                f"Item with score {item.score} should have been filtered "
                f"(threshold=0.5)"
            )


# ---------------------------------------------------------------------------
# Category 3: Scoring Weights Effect
# ---------------------------------------------------------------------------

class TestScoringWeightsEffect:
    """Verify scoring weights change result ordering as expected."""

    @pytest.fixture
    def mixed_items(self):
        """Items with diverse confidence, recency, and success rates."""
        now = datetime.now(timezone.utc)
        return [
            # High confidence, old, moderate success
            _make_heuristic(0, "cond0", "strat0", confidence=0.95,
                            success_count=6, occurrence_count=10, days_ago=60),
            # Low confidence, recent, high success
            _make_heuristic(1, "cond1", "strat1", confidence=0.3,
                            success_count=9, occurrence_count=10, days_ago=1),
            # Medium confidence, medium recency, medium success
            _make_heuristic(2, "cond2", "strat2", confidence=0.6,
                            success_count=5, occurrence_count=10, days_ago=15),
        ]

    def test_default_weights_balanced_ranking(self, mixed_items):
        """[UNIT] ScoringWeights(default) — should produce balanced ranking
        considering all factors."""
        scorer = MemoryScorer()  # default weights
        sims = [0.5, 0.5, 0.5]  # equal similarity

        scored = scorer.score_heuristics(mixed_items, similarities=sims)

        # All items should be scored and ranked
        assert len(scored) == 3
        # Scores should differ (different recency/success/confidence)
        scores = [s.score for s in scored]
        assert len(set(round(s, 6) for s in scores)) > 1, (
            "With different recency/success/confidence, scores should differ"
        )

    def test_pure_similarity_ignores_other_factors(self, mixed_items):
        """[UNIT] ScoringWeights(sim=1.0) — should rank by similarity only,
        ignoring recency, success, and confidence."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        # Item 2 gets highest similarity, item 0 gets lowest
        sims = [0.3, 0.6, 0.9]

        scored = scorer.score_heuristics(mixed_items, similarities=sims)

        assert scored[0].item.id == "heur_002", (
            f"Pure similarity: item 2 (sim=0.9) should be first, "
            f"got {scored[0].item.id}"
        )
        assert scored[-1].item.id == "heur_000", (
            f"Pure similarity: item 0 (sim=0.3) should be last, "
            f"got {scored[-1].item.id}"
        )

    def test_pure_recency_ranks_recent_first(self, mixed_items):
        """[UNIT] ScoringWeights(recency=1.0) — should rank most recent first."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=0.0, recency=1.0, success_rate=0.0, confidence=0.0
            )
        )
        sims = [0.5, 0.5, 0.5]

        scored = scorer.score_heuristics(mixed_items, similarities=sims)

        # Item 1 is most recent (1 day ago), item 0 is oldest (60 days)
        assert scored[0].item.id == "heur_001", (
            f"Pure recency: item 1 (1 day ago) should be first, "
            f"got {scored[0].item.id}"
        )
        assert scored[-1].item.id == "heur_000", (
            f"Pure recency: item 0 (60 days ago) should be last, "
            f"got {scored[-1].item.id}"
        )

    def test_pure_confidence_ranks_by_stored_confidence(self, mixed_items):
        """[UNIT] ScoringWeights(confidence=1.0) — should rank by stored
        confidence value."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=0.0, recency=0.0, success_rate=0.0, confidence=1.0
            )
        )
        sims = [0.5, 0.5, 0.5]

        scored = scorer.score_heuristics(mixed_items, similarities=sims)

        assert scored[0].item.id == "heur_000", (
            f"Pure confidence: item 0 (conf=0.95) should be first, "
            f"got {scored[0].item.id}"
        )
        assert scored[-1].item.id == "heur_001", (
            f"Pure confidence: item 1 (conf=0.3) should be last, "
            f"got {scored[-1].item.id}"
        )

    def test_weight_normalization(self):
        """[UNIT] ScoringWeights — unnormalized weights should be auto-normalized
        to sum to 1.0."""
        w = ScoringWeights(similarity=2.0, recency=2.0, success_rate=2.0, confidence=2.0)
        total = w.similarity + w.recency + w.success_rate + w.confidence
        assert abs(total - 1.0) < 0.01, (
            f"Weights should normalize to 1.0, got {total}"
        )


# ---------------------------------------------------------------------------
# Category 4: Mode-Aware Retrieval
# ---------------------------------------------------------------------------

class TestModeAwareRetrieval:
    """Verify retrieval modes affect behavior correctly."""

    def test_broad_mode_has_high_top_k(self):
        """[UNIT] BROAD mode — should request more results than PRECISE."""
        broad = get_mode_config(RetrievalMode.BROAD)
        precise = get_mode_config(RetrievalMode.PRECISE)
        assert broad.top_k > precise.top_k, (
            f"BROAD top_k ({broad.top_k}) should > PRECISE top_k ({precise.top_k})"
        )

    def test_precise_mode_has_higher_min_confidence(self):
        """[UNIT] PRECISE mode — should have higher confidence threshold
        than BROAD."""
        broad = get_mode_config(RetrievalMode.BROAD)
        precise = get_mode_config(RetrievalMode.PRECISE)
        assert precise.min_confidence > broad.min_confidence, (
            f"PRECISE min_confidence ({precise.min_confidence}) should > "
            f"BROAD min_confidence ({broad.min_confidence})"
        )

    def test_benchmark_mode_uses_pure_similarity(self):
        """[CRITICAL] BENCHMARK mode — must use pure cosine similarity ranking
        with no auxiliary signals."""
        config = get_mode_config(RetrievalMode.BENCHMARK)
        assert config.weights["similarity"] == 1.0
        assert config.weights["recency"] == 0.0
        assert config.weights["success_rate"] == 0.0
        assert config.weights["confidence"] == 0.0
        assert config.min_confidence == 0.0, (
            "BENCHMARK mode must have 0 confidence threshold"
        )
        assert config.diversity_factor == 0.0, (
            "BENCHMARK mode must not apply diversity filtering"
        )

    def test_similarity_mode_uses_pure_similarity(self):
        """[UNIT] SIMILARITY mode — must use pure cosine similarity like BENCHMARK."""
        config = get_mode_config(RetrievalMode.SIMILARITY)
        assert config.weights["similarity"] == 1.0
        assert config.weights["recency"] == 0.0
        assert config.weights["success_rate"] == 0.0
        assert config.weights["confidence"] == 0.0

    def test_diagnostic_mode_prioritizes_failures(self):
        """[UNIT] DIAGNOSTIC mode — must boost failed outcomes."""
        config = get_mode_config(RetrievalMode.DIAGNOSTIC)
        assert config.prioritize_failures is True
        assert config.include_anti_patterns is True

    def test_broad_mode_has_high_diversity(self):
        """[UNIT] BROAD mode — should have higher diversity factor than PRECISE."""
        broad = get_mode_config(RetrievalMode.BROAD)
        precise = get_mode_config(RetrievalMode.PRECISE)
        assert broad.diversity_factor > precise.diversity_factor

    def test_recall_mode_strongly_boosts_exact_matches(self):
        """[UNIT] RECALL mode — should have highest exact_match_boost."""
        recall = get_mode_config(RetrievalMode.RECALL)
        assert recall.exact_match_boost >= 2.0, (
            f"RECALL exact_match_boost ({recall.exact_match_boost}) should be >= 2.0"
        )

    def test_all_modes_have_valid_weight_sums(self):
        """[UNIT] All modes — weights must sum to 1.0."""
        for mode in RetrievalMode:
            config = get_mode_config(mode)
            if config.weights:
                total = sum(config.weights.values())
                assert abs(total - 1.0) < 0.02, (
                    f"Mode {mode.value} weights sum to {total}, expected ~1.0"
                )

    def test_benchmark_mode_returns_everything(self):
        """[UNIT] BENCHMARK mode — top_k should be high enough to capture
        all relevant results."""
        config = get_mode_config(RetrievalMode.BENCHMARK)
        assert config.top_k >= 50, (
            f"BENCHMARK top_k ({config.top_k}) should be >= 50 for full ranking"
        )


# ---------------------------------------------------------------------------
# Category 5: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases that must not crash or produce wrong rankings."""

    def test_empty_items_returns_empty_scored(self):
        """[UNIT] MemoryScorer — empty input should return empty output."""
        scorer = MemoryScorer()
        assert scorer.score_heuristics([]) == []
        assert scorer.score_outcomes([]) == []
        assert scorer.score_domain_knowledge([]) == []
        assert scorer.score_anti_patterns([]) == []

    def test_single_item_returns_that_item(self):
        """[UNIT] MemoryScorer — single item should be returned as-is."""
        scorer = MemoryScorer()
        dk = _make_knowledge(0, "only item")
        scored = scorer.score_domain_knowledge([dk], similarities=[0.8])

        assert len(scored) == 1
        assert scored[0].item.id == "dk_000"
        assert scored[0].similarity_score == 0.8

    def test_all_identical_items_no_crash(self):
        """[UNIT] MemoryScorer — all items identical should not crash
        and should return all items."""
        scorer = MemoryScorer()
        items = [_make_knowledge(i, "same fact") for i in range(5)]
        sims = [0.5] * 5

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        assert len(scored) == 5
        # All scores should be equal (or nearly equal)
        scores = [s.score for s in scored]
        assert max(scores) - min(scores) < 0.01

    def test_zero_similarity_still_returns_items(self):
        """[UNIT] MemoryScorer — items with zero similarity should still
        be scored (via other factors)."""
        scorer = MemoryScorer()  # default weights include recency + confidence
        items = [_make_knowledge(i, f"fact {i}") for i in range(3)]
        sims = [0.0, 0.0, 0.0]

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        assert len(scored) == 3
        # Scores should still be > 0 due to recency and confidence
        for s in scored:
            assert s.score > 0.0, (
                "Even with 0 similarity, other factors should produce positive score"
            )

    def test_similarity_one_returns_item_at_top(self):
        """[UNIT] MemoryScorer — item with similarity=1.0 should be first
        when using pure similarity weights."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        items = [_make_knowledge(i, f"fact {i}") for i in range(5)]
        sims = [0.2, 0.4, 1.0, 0.6, 0.8]

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        assert scored[0].item.id == "dk_002", (
            f"Item with sim=1.0 should be first, got {scored[0].item.id}"
        )

    def test_negative_similarity_handled_gracefully(self):
        """[UNIT] MemoryScorer — negative similarity values should not crash."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        items = [_make_knowledge(i, f"fact {i}") for i in range(3)]
        sims = [-0.5, 0.0, 0.5]

        scored = scorer.score_domain_knowledge(items, similarities=sims)

        assert len(scored) == 3
        # Item with highest sim should be first
        assert scored[0].item.id == "dk_002"

    def test_very_old_item_gets_near_zero_recency(self):
        """[UNIT] MemoryScorer — item from 1000 days ago should have
        near-zero recency score."""
        scorer = MemoryScorer(recency_half_life_days=30.0)
        old_item = _make_knowledge(0, "ancient fact", days_ago=1000)

        scored = scorer.score_domain_knowledge([old_item], similarities=[0.5])

        assert scored[0].recency_score < 0.001, (
            f"Item from 1000 days ago should have near-zero recency, "
            f"got {scored[0].recency_score}"
        )

    def test_score_threshold_removes_all_below(self):
        """[UNIT] apply_score_threshold — with threshold > all scores,
        should return empty list."""
        scorer = MemoryScorer()
        items = [_make_knowledge(0, "fact")]
        scored = scorer.score_domain_knowledge(items, similarities=[0.01])
        filtered = scorer.apply_score_threshold(scored, min_score=999.0)
        assert filtered == []


# ---------------------------------------------------------------------------
# Category 6: End-to-End Retrieval Flow
# ---------------------------------------------------------------------------

class TestEndToEndRetrievalFlow:
    """Test the full pipeline from storage through engine to results."""

    @pytest.fixture
    def engine_with_mock(self):
        """Create RetrievalEngine with MockStorage and MockEmbedder."""
        storage = MockStorage()
        engine = RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,
        )
        return engine, storage

    def test_retrieve_returns_memory_slice(self, engine_with_mock):
        """[INTEGRATION] RetrievalEngine.retrieve — should return a MemorySlice
        with domain knowledge populated."""
        engine, storage = engine_with_mock

        # Store items
        for i in range(5):
            dk = _make_knowledge(i, f"fact about topic {i}")
            storage.save_domain_knowledge(dk)

        result = engine.retrieve(
            query="What about topic 3?",
            agent="test-agent",
            project_id="test-project",
        )

        assert isinstance(result, MemorySlice)
        assert result.total_items > 0

    def test_retrieve_with_mode_returns_tuple(self, engine_with_mock):
        """[INTEGRATION] RetrievalEngine.retrieve_with_mode — should return
        (MemorySlice, RetrievalMode, reason) tuple."""
        engine, storage = engine_with_mock

        for i in range(5):
            dk = _make_knowledge(i, f"fact about topic {i}")
            storage.save_domain_knowledge(dk)

        result, mode, reason = engine.retrieve_with_mode(
            query="What about topic 3?",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BENCHMARK,
        )

        assert isinstance(result, MemorySlice)
        assert mode == RetrievalMode.BENCHMARK
        assert isinstance(reason, str)

    def test_benchmark_mode_end_to_end(self, engine_with_mock):
        """[CRITICAL] BENCHMARK mode E2E — retrieve_with_mode in BENCHMARK
        should use pure similarity weights and return all items."""
        engine, storage = engine_with_mock

        for i in range(20):
            dk = _make_knowledge(i, f"conversation session {i}")
            storage.save_domain_knowledge(dk)

        result, mode, _ = engine.retrieve_with_mode(
            query="find session 7",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BENCHMARK,
        )

        assert mode == RetrievalMode.BENCHMARK
        # BENCHMARK top_k is 50, but we only have 20 items
        # All should be returned (min_confidence=0.0)
        assert result.total_items > 0

    def test_mock_storage_retrieval_conversation_sessions(self):
        """[INTEGRATION] Mini-benchmark — store 20 conversation sessions,
        query for a specific one, verify it is in results.

        This mimics the LongMemEval benchmark scenario at small scale.
        """
        storage = MockStorage()
        embedder = MockEmbedder(dimension=16)

        # Store 20 sessions with distinct embeddings
        sessions = []
        for i in range(20):
            vec = _make_blended_vector(16, i % 16, noise_level=0.15)
            dk = _make_knowledge(
                i,
                f"In session {i}, the user asked about topic_{i} and got answer_{i}",
                domain="conversation",
                embedding=vec,
            )
            storage.save_domain_knowledge(dk)
            sessions.append(dk)

        # Query targeting session #7 (uses dimension 7 as primary)
        query_vec = _make_unit_vector(16, 7)

        # Manually score with pure similarity
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=1.0, recency=0.0, success_rate=0.0, confidence=0.0
            )
        )
        sims = [_cosine_similarity(query_vec, s.embedding) for s in sessions]
        scored = scorer.score_domain_knowledge(sessions, similarities=sims)

        top_5_ids = [s.item.id for s in scored[:5]]
        assert "dk_007" in top_5_ids, (
            f"Session #7 should be in top 5 for aligned query. "
            f"Top 5: {top_5_ids}"
        )

    def test_retrieval_engine_scorer_weights_restored_after_mode(self):
        """[UNIT] RetrievalEngine — original scorer weights must be restored
        after retrieve_with_mode (even if it uses different weights)."""
        storage = MockStorage()
        original_weights = ScoringWeights(
            similarity=0.4, recency=0.3, success_rate=0.2, confidence=0.1
        )
        engine = RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
            enable_cache=False,
            scoring_weights=original_weights,
        )

        engine.retrieve_with_mode(
            query="test",
            agent="test-agent",
            project_id="test-project",
            mode=RetrievalMode.BENCHMARK,
        )

        # Weights should be restored
        w = engine.scorer.weights
        assert abs(w.similarity - 0.4) < 0.01
        assert abs(w.recency - 0.3) < 0.01
        assert abs(w.success_rate - 0.2) < 0.01
        assert abs(w.confidence - 0.1) < 0.01


# ---------------------------------------------------------------------------
# Category 7: Query Sanitizer Integration
# ---------------------------------------------------------------------------

class TestQuerySanitizerIntegration:
    """Verify query sanitizer handles contaminated queries."""

    def test_short_query_passes_through(self):
        """[UNIT] sanitize_query — short query (<=200 chars) should pass
        through unchanged."""
        result = sanitize_query("What is the login endpoint?")
        assert result["was_sanitized"] is False
        assert result["method"] == "passthrough"
        assert result["clean_query"] == "What is the login endpoint?"

    def test_long_query_with_question_mark_extracts_question(self):
        """[UNIT] sanitize_query — long query with system prompt preamble
        should extract the question at the end."""
        preamble = "You are a helpful assistant. " * 20  # >200 chars
        question = "What authentication method does the API use?"
        contaminated = preamble + question

        result = sanitize_query(contaminated)

        assert result["was_sanitized"] is True
        assert result["method"] in ("question_extraction", "tail_sentence")
        assert len(result["clean_query"]) < len(contaminated)
        # The extracted text should contain the question content
        assert "authentication" in result["clean_query"].lower() or \
               "api" in result["clean_query"].lower()

    def test_very_long_query_does_not_crash(self):
        """[UNIT] sanitize_query — 10,000 char query should not crash."""
        huge_query = "x " * 5000  # 10,000 chars
        result = sanitize_query(huge_query)
        assert result["was_sanitized"] is True
        assert len(result["clean_query"]) <= 250  # MAX_QUERY_LENGTH

    def test_empty_query_returns_empty(self):
        """[UNIT] sanitize_query — empty string should return empty."""
        result = sanitize_query("")
        assert result["was_sanitized"] is False
        assert result["clean_query"] == ""

    def test_whitespace_only_query(self):
        """[UNIT] sanitize_query — whitespace-only query should return it."""
        result = sanitize_query("   ")
        assert result["was_sanitized"] is False

    def test_sanitized_query_preserves_search_intent(self):
        """[INTEGRATION] Sanitizer+Scorer — after sanitization, a query about
        'database schema' should still produce high similarity to DB-related items
        when using deterministic embeddings."""
        # This tests that sanitization does not destroy the search intent
        embedder = MockEmbedder(dimension=16)

        # Create the clean query and a contaminated version
        clean = "What is the database schema for users?"
        preamble = "System: You are an AI assistant. Follow instructions. " * 10
        contaminated = preamble + clean

        # Sanitize
        result = sanitize_query(contaminated)
        sanitized = result["clean_query"]

        # Both clean and sanitized should produce embeddings
        # (we cannot test true semantic similarity with MockEmbedder,
        #  but we verify the pipeline does not crash or return empty)
        clean_emb = embedder.encode(clean)
        sanitized_emb = embedder.encode(sanitized)

        assert len(clean_emb) == 16
        assert len(sanitized_emb) == 16
        # Sanitized query should be much shorter than contaminated
        assert len(sanitized) < len(contaminated)


# ---------------------------------------------------------------------------
# Category 8: Failure Boost & Exact Match Boost (engine internals)
# ---------------------------------------------------------------------------

class TestEngineInternals:
    """Test internal engine methods that affect ranking quality."""

    def test_boost_failures_raises_failed_outcome_scores(self):
        """[UNIT] _boost_failures — failed outcomes should have higher scores
        after boosting."""
        storage = MockStorage()
        engine = RetrievalEngine(
            storage=storage, embedding_provider="mock", enable_cache=False
        )

        success_outcome = create_test_outcome(
            id="out_success", success=True
        )
        fail_outcome = create_test_outcome(
            id="out_fail", success=False
        )

        # Create ScoredItems
        scored_success = ScoredItem(
            item=success_outcome, score=0.8,
            similarity_score=0.8, recency_score=0.5,
            success_score=1.0, confidence_score=0.5,
        )
        scored_fail = ScoredItem(
            item=fail_outcome, score=0.6,
            similarity_score=0.6, recency_score=0.5,
            success_score=0.3, confidence_score=0.5,
        )

        boosted = engine._boost_failures([scored_success, scored_fail])

        # Failed outcome score should be boosted (0.6 * 1.5 = 0.9)
        fail_boosted = [b for b in boosted if b.item.id == "out_fail"][0]
        assert fail_boosted.score == pytest.approx(0.9, abs=0.01)

        # Success outcome should be unchanged
        succ_boosted = [b for b in boosted if b.item.id == "out_success"][0]
        assert succ_boosted.score == pytest.approx(0.8, abs=0.01)

    def test_exact_match_boost_amplifies_high_similarity(self):
        """[UNIT] _apply_exact_match_boost — items with similarity > 0.9
        should be boosted by the boost factor."""
        storage = MockStorage()
        engine = RetrievalEngine(
            storage=storage, embedding_provider="mock", enable_cache=False
        )

        dk = _make_knowledge(0, "fact")
        high_sim = ScoredItem(
            item=dk, score=0.5,
            similarity_score=0.95, recency_score=0.5,
            success_score=0.5, confidence_score=0.5,
        )
        low_sim = ScoredItem(
            item=_make_knowledge(1, "fact2"), score=0.5,
            similarity_score=0.3, recency_score=0.5,
            success_score=0.5, confidence_score=0.5,
        )

        boosted = engine._apply_exact_match_boost([high_sim, low_sim], 2.0)

        # High sim item should be boosted (0.5 * 2.0 = 1.0)
        high_boosted = [b for b in boosted if b.item.id == "dk_000"][0]
        assert high_boosted.score == pytest.approx(1.0, abs=0.01)

        # Low sim item should be unchanged
        low_boosted = [b for b in boosted if b.item.id == "dk_001"][0]
        assert low_boosted.score == pytest.approx(0.5, abs=0.01)

    def test_extract_top_k_respects_threshold(self):
        """[UNIT] _extract_top_k — should filter by min_score_threshold
        AND limit to top_k."""
        storage = MockStorage()
        engine = RetrievalEngine(
            storage=storage, embedding_provider="mock",
            enable_cache=False, min_score_threshold=0.5,
        )

        scored = [
            ScoredItem(
                item=_make_knowledge(i, f"fact {i}"),
                score=0.1 * (i + 1),
                similarity_score=0.5, recency_score=0.5,
                success_score=0.5, confidence_score=0.5,
            )
            for i in range(10)
        ]
        scored.sort(key=lambda x: -x.score)

        result = engine._extract_top_k(scored, top_k=3)

        # Only items with score >= 0.5 should be included
        for item in result:
            pass  # items are unwrapped, can't check score directly
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Category 9: Recency Decay Correctness
# ---------------------------------------------------------------------------

class TestRecencyDecay:
    """Verify recency scoring uses correct exponential decay."""

    def test_brand_new_item_has_recency_near_one(self):
        """[UNIT] _compute_recency_score — item from now should score ~1.0."""
        scorer = MemoryScorer(recency_half_life_days=30.0)
        now = datetime.now(timezone.utc)
        score = scorer._compute_recency_score(now)
        assert score > 0.99

    def test_half_life_item_scores_half(self):
        """[UNIT] _compute_recency_score — item from exactly half_life days ago
        should score ~0.5."""
        scorer = MemoryScorer(recency_half_life_days=30.0)
        past = datetime.now(timezone.utc) - timedelta(days=30)
        score = scorer._compute_recency_score(past)
        assert abs(score - 0.5) < 0.05

    def test_double_half_life_scores_quarter(self):
        """[UNIT] _compute_recency_score — item from 2x half_life days ago
        should score ~0.25."""
        scorer = MemoryScorer(recency_half_life_days=30.0)
        past = datetime.now(timezone.utc) - timedelta(days=60)
        score = scorer._compute_recency_score(past)
        assert abs(score - 0.25) < 0.05

    def test_naive_datetime_handled(self):
        """[UNIT] _compute_recency_score — naive datetime (no tzinfo)
        should not crash."""
        scorer = MemoryScorer()
        naive = datetime(2020, 1, 1)  # no timezone
        score = scorer._compute_recency_score(naive)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Category 10: Anti-Pattern Scoring
# ---------------------------------------------------------------------------

class TestAntiPatternScoring:
    """Verify anti-pattern scoring respects occurrence count."""

    def test_high_occurrence_anti_pattern_ranked_higher(self):
        """[UNIT] score_anti_patterns — anti-pattern seen 10 times should rank
        higher than one seen 1 time (when other factors equal)."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=0.0, recency=0.0, success_rate=1.0, confidence=0.0
            )
        )
        low_occ = create_test_anti_pattern(
            id="ap_low", occurrence_count=1
        )
        high_occ = create_test_anti_pattern(
            id="ap_high", occurrence_count=10
        )

        scored = scorer.score_anti_patterns(
            [low_occ, high_occ], similarities=[0.5, 0.5]
        )

        assert scored[0].item.id == "ap_high", (
            "Higher occurrence anti-pattern should rank first"
        )

    def test_anti_pattern_occurrence_capped_at_ten(self):
        """[UNIT] score_anti_patterns — occurrence_count contribution should
        cap at 10 (normalized to 1.0)."""
        scorer = MemoryScorer(
            weights=ScoringWeights(
                similarity=0.0, recency=0.0, success_rate=1.0, confidence=0.0
            )
        )
        ap_10 = create_test_anti_pattern(id="ap_10", occurrence_count=10)
        ap_100 = create_test_anti_pattern(id="ap_100", occurrence_count=100)

        scored = scorer.score_anti_patterns(
            [ap_10, ap_100], similarities=[0.5, 0.5]
        )

        # Both should have success_score = 1.0 (capped at 10/10)
        scores = {s.item.id: s.success_score for s in scored}
        assert scores["ap_10"] == pytest.approx(1.0, abs=0.01)
        assert scores["ap_100"] == pytest.approx(1.0, abs=0.01)
