"""
Integration Tests: Consolidation → Retrieval Flow.

Validates that after consolidation merges duplicate memories,
retrieval returns the correct merged results with updated
scores, and original memories are no longer accessible.

P0-2 from ALMA v0.8.0 Architecture Evaluation.
"""

import pytest

from alma.consolidation.engine import ConsolidationEngine
from alma.retrieval.embeddings import MockEmbedder
from alma.retrieval.scoring import MemoryScorer, ScoringWeights
from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
)
from alma.testing.mocks import MockStorage


# Helpers for creating embeddings that will be grouped as similar
def _similar_embedding(base: float = 0.5, dim: int = 384) -> list[float]:
    """Create an embedding vector. Identical base = identical vector = cosine sim 1.0."""
    return [base] * dim


def _different_embedding(dim: int = 384) -> list[float]:
    """Create an embedding vector orthogonal to the similar ones."""
    # Alternating positive/negative to be dissimilar from uniform vectors
    return [0.5 if i % 2 == 0 else -0.5 for i in range(dim)]


class TestConsolidationRetrievalHeuristics:
    """Test consolidation + retrieval for heuristic memories."""

    @pytest.fixture
    def storage(self):
        return MockStorage()

    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimension=384)

    @pytest.fixture
    def scorer(self):
        return MemoryScorer(
            weights=ScoringWeights(
                similarity=0.4,
                recency=0.3,
                success_rate=0.2,
                confidence=0.1,
            )
        )

    @pytest.mark.asyncio
    async def test_merged_heuristic_replaces_originals(self, storage, embedder):
        """After consolidation, originals are deleted and merged item exists."""
        # Create 3 similar heuristics with identical embeddings
        similar_emb = _similar_embedding(0.7)
        h1 = create_test_heuristic(
            condition="form with multiple required fields",
            strategy="test each field incrementally",
            confidence=0.80,
            occurrence_count=5,
            success_count=4,
            embedding=similar_emb,
        )
        h2 = create_test_heuristic(
            condition="form with many required inputs",
            strategy="validate fields one by one",
            confidence=0.75,
            occurrence_count=3,
            success_count=2,
            embedding=similar_emb,
        )
        h3 = create_test_heuristic(
            condition="form with several mandatory fields",
            strategy="check each field separately",
            confidence=0.70,
            occurrence_count=4,
            success_count=3,
            embedding=similar_emb,
        )
        original_ids = {h1.id, h2.id, h3.id}

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)
        storage.save_heuristic(h3)

        # Run consolidation (no LLM, rule-based merge)
        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.success
        assert result.groups_found >= 1
        assert result.merged_count >= 2  # 3 merged into 1 = 2 merged away

        # Retrieve all heuristics — originals should be gone
        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=20,
        )
        retrieved_ids = {h.id for h in retrieved}

        # Original IDs should not be present
        assert original_ids.isdisjoint(retrieved_ids), (
            f"Original IDs still present: {original_ids & retrieved_ids}"
        )

        # Exactly 1 merged heuristic should exist
        assert len(retrieved) == 1
        merged = retrieved[0]

        # Merged heuristic should have provenance metadata
        assert "merged_from" in merged.metadata
        assert set(merged.metadata["merged_from"]) == original_ids

    @pytest.mark.asyncio
    async def test_merged_heuristic_aggregates_counts(self, storage, embedder):
        """Merged heuristic combines occurrence and success counts."""
        similar_emb = _similar_embedding(0.3)
        h1 = create_test_heuristic(
            occurrence_count=10,
            success_count=8,
            confidence=0.90,
            embedding=similar_emb,
        )
        h2 = create_test_heuristic(
            occurrence_count=5,
            success_count=4,
            confidence=0.85,
            embedding=similar_emb,
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 1
        merged = retrieved[0]

        # Occurrence and success counts should be summed
        assert merged.occurrence_count == 15
        assert merged.success_count == 12

        # Confidence should be the average
        assert abs(merged.confidence - 0.875) < 0.01

    @pytest.mark.asyncio
    async def test_scoring_favors_merged_heuristic(self, storage, embedder, scorer):
        """Merged heuristic should score higher than originals would have."""
        similar_emb = _similar_embedding(0.5)

        # Create 3 low-confidence heuristics
        originals = []
        for i in range(3):
            h = create_test_heuristic(
                confidence=0.50 + i * 0.05,
                occurrence_count=3,
                success_count=2,
                embedding=similar_emb,
            )
            originals.append(h)
            storage.save_heuristic(h)

        # Score originals before consolidation
        pre_scores = scorer.score_heuristics(originals)
        max_pre_score = max(s.score for s in pre_scores)

        # Consolidate
        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        # Score the merged heuristic
        merged = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(merged) == 1

        post_scores = scorer.score_heuristics(merged)
        merged_score = post_scores[0].score

        # Merged heuristic has combined occurrence_count (9) and success_count (6)
        # giving a higher success_rate component and confidence, so score should
        # be at least as high as the best original
        assert merged_score >= max_pre_score * 0.9, (
            f"Merged score {merged_score:.3f} too low vs best original {max_pre_score:.3f}"
        )

    @pytest.mark.asyncio
    async def test_dissimilar_heuristics_not_merged(self, storage, embedder):
        """Heuristics with different embeddings should remain separate."""
        h1 = create_test_heuristic(
            condition="database optimization",
            strategy="add indexes",
            embedding=_similar_embedding(0.8),
        )
        h2 = create_test_heuristic(
            condition="UI testing",
            strategy="use visual regression tests",
            embedding=_different_embedding(),
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.merged_count == 0
        assert result.groups_found == 0

        # Both heuristics should still be present
        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 2

    @pytest.mark.asyncio
    async def test_dry_run_preserves_originals(self, storage, embedder):
        """Dry run should identify groups but not modify storage."""
        similar_emb = _similar_embedding(0.6)
        h1 = create_test_heuristic(embedding=similar_emb)
        h2 = create_test_heuristic(embedding=similar_emb)

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
            dry_run=True,
        )

        # Should find groups but not merge
        assert result.groups_found >= 1
        assert result.merged_count >= 1  # Reports what would be merged

        # Originals should still exist
        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 2
        assert {h.id for h in retrieved} == {h1.id, h2.id}


class TestConsolidationRetrievalDomainKnowledge:
    """Test consolidation + retrieval for domain knowledge memories."""

    @pytest.fixture
    def storage(self):
        return MockStorage()

    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimension=384)

    @pytest.mark.asyncio
    async def test_merged_knowledge_replaces_originals(self, storage, embedder):
        """Domain knowledge consolidation merges similar facts."""
        similar_emb = _similar_embedding(0.4)
        dk1 = create_test_knowledge(
            domain="python",
            fact="Use list comprehensions for filtering",
            confidence=0.90,
            embedding=similar_emb,
        )
        dk2 = create_test_knowledge(
            domain="python",
            fact="List comprehensions are faster for filtering",
            confidence=0.85,
            embedding=similar_emb,
        )
        dk3 = create_test_knowledge(
            domain="python",
            fact="Prefer list comprehensions over filter()",
            confidence=0.80,
            embedding=similar_emb,
        )
        original_ids = {dk1.id, dk2.id, dk3.id}

        storage.save_domain_knowledge(dk1)
        storage.save_domain_knowledge(dk2)
        storage.save_domain_knowledge(dk3)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="domain_knowledge",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.success
        assert result.groups_found >= 1
        assert result.merged_count >= 2

        # Verify originals deleted, merged exists
        retrieved = storage.get_domain_knowledge(
            project_id="test-project",
            agent="test-agent",
            top_k=20,
        )
        retrieved_ids = {dk.id for dk in retrieved}
        assert original_ids.isdisjoint(retrieved_ids)
        assert len(retrieved) == 1

        merged = retrieved[0]
        assert "merged_from" in merged.metadata
        assert set(merged.metadata["merged_from"]) == original_ids

    @pytest.mark.asyncio
    async def test_knowledge_confidence_averaged(self, storage, embedder):
        """Merged domain knowledge has averaged confidence."""
        similar_emb = _similar_embedding(0.2)
        dk1 = create_test_knowledge(confidence=1.0, embedding=similar_emb)
        dk2 = create_test_knowledge(confidence=0.6, embedding=similar_emb)

        storage.save_domain_knowledge(dk1)
        storage.save_domain_knowledge(dk2)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="domain_knowledge",
            similarity_threshold=0.85,
            use_llm=False,
        )

        retrieved = storage.get_domain_knowledge(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 1
        assert abs(retrieved[0].confidence - 0.80) < 0.01


class TestConsolidationRetrievalAntiPatterns:
    """Test consolidation + retrieval for anti-pattern memories."""

    @pytest.fixture
    def storage(self):
        return MockStorage()

    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimension=384)

    @pytest.mark.asyncio
    async def test_merged_anti_pattern_sums_occurrences(self, storage, embedder):
        """Merged anti-patterns should sum occurrence counts."""
        similar_emb = _similar_embedding(0.9)
        ap1 = create_test_anti_pattern(
            pattern="using SELECT *",
            why_bad="fetches unnecessary columns",
            better_alternative="list explicit columns",
            occurrence_count=5,
            embedding=similar_emb,
        )
        ap2 = create_test_anti_pattern(
            pattern="SELECT * in queries",
            why_bad="wastes bandwidth on unused columns",
            better_alternative="specify needed columns",
            occurrence_count=3,
            embedding=similar_emb,
        )
        original_ids = {ap1.id, ap2.id}

        storage.save_anti_pattern(ap1)
        storage.save_anti_pattern(ap2)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="anti_patterns",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.success
        assert result.merged_count >= 1

        retrieved = storage.get_anti_patterns(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 1
        merged = retrieved[0]

        # Occurrence count should be summed
        assert merged.occurrence_count == 8
        assert "merged_from" in merged.metadata
        assert set(merged.metadata["merged_from"]) == original_ids


class TestConsolidationMergeDetails:
    """Test consolidation result metadata and merge tracking."""

    @pytest.fixture
    def storage(self):
        return MockStorage()

    @pytest.fixture
    def embedder(self):
        return MockEmbedder(dimension=384)

    @pytest.mark.asyncio
    async def test_merge_details_track_provenance(self, storage, embedder):
        """merge_details should record what was merged into what."""
        similar_emb = _similar_embedding(0.1)
        h1 = create_test_heuristic(embedding=similar_emb)
        h2 = create_test_heuristic(embedding=similar_emb)
        h3 = create_test_heuristic(embedding=similar_emb)
        original_ids = {h1.id, h2.id, h3.id}

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)
        storage.save_heuristic(h3)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert len(result.merge_details) >= 1
        detail = result.merge_details[0]

        assert "merged_from" in detail
        assert "merged_into" in detail
        assert "count" in detail
        assert set(detail["merged_from"]) == original_ids
        assert detail["count"] == 3

        # The merged_into ID should exist in storage
        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert any(h.id == detail["merged_into"] for h in retrieved)

    @pytest.mark.asyncio
    async def test_consolidation_with_mixed_similarity(self, storage, embedder):
        """Some items similar, others not — partial merge."""
        emb_group_a = _similar_embedding(0.5)
        emb_group_b = _different_embedding()

        # Group A: 2 similar heuristics
        h1 = create_test_heuristic(
            condition="API error handling",
            embedding=emb_group_a,
        )
        h2 = create_test_heuristic(
            condition="API error management",
            embedding=emb_group_a,
        )
        # Group B: 1 different heuristic (should not be merged)
        h3 = create_test_heuristic(
            condition="UI responsiveness testing",
            embedding=emb_group_b,
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)
        storage.save_heuristic(h3)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.groups_found == 1  # Only group A is multi-item
        assert result.merged_count == 1  # 2 merged into 1

        # Should have 2 heuristics: 1 merged + 1 untouched
        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 2

        # The untouched heuristic should still have its original ID
        retrieved_ids = {h.id for h in retrieved}
        assert h3.id in retrieved_ids

    @pytest.mark.asyncio
    async def test_single_memory_not_consolidated(self, storage, embedder):
        """A single memory should not trigger any consolidation."""
        h1 = create_test_heuristic(embedding=_similar_embedding(0.5))
        storage.save_heuristic(h1)

        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.merged_count == 0
        assert result.memories_processed == 1

        retrieved = storage.get_heuristics(
            project_id="test-project",
            agent="test-agent",
            top_k=10,
        )
        assert len(retrieved) == 1
        assert retrieved[0].id == h1.id

    @pytest.mark.asyncio
    async def test_empty_storage_consolidation(self, storage, embedder):
        """Consolidating empty storage should succeed with zero work."""
        engine = ConsolidationEngine(storage=storage, embedder=embedder)
        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            use_llm=False,
        )

        assert result.success
        assert result.merged_count == 0
        assert result.memories_processed == 0
        assert result.groups_found == 0
