"""
Unit tests for ALMA Memory Consolidation Engine.

Tests cover:
- Finding similar groups
- Merging heuristics (without LLM)
- Dry run mode
- Similarity threshold
- Consolidation result tracking
- Provenance preservation
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alma.consolidation.engine import ConsolidationEngine, ConsolidationResult
from alma.consolidation.prompts import (
    MERGE_ANTI_PATTERNS_PROMPT,
    MERGE_DOMAIN_KNOWLEDGE_PROMPT,
    MERGE_HEURISTICS_PROMPT,
)
from alma.retrieval.embeddings import MockEmbedder
from alma.storage.base import StorageBackend
from alma.types import DomainKnowledge, Heuristic


class TestConsolidationResult:
    """Tests for ConsolidationResult dataclass."""

    def test_result_initialization(self):
        """Test ConsolidationResult can be created."""
        result = ConsolidationResult(
            merged_count=5,
            groups_found=2,
            memories_processed=10,
        )

        assert result.merged_count == 5
        assert result.groups_found == 2
        assert result.memories_processed == 10
        assert result.errors == []
        assert result.merge_details == []

    def test_result_success_with_no_errors(self):
        """Test success property when no errors."""
        result = ConsolidationResult(
            merged_count=3,
            groups_found=1,
            memories_processed=5,
        )

        assert result.success is True

    def test_result_success_with_errors_but_merges(self):
        """Test success when there are errors but also merges."""
        result = ConsolidationResult(
            merged_count=2,
            groups_found=1,
            memories_processed=5,
            errors=["Some non-critical error"],
        )

        # Still considered successful if some merges happened
        assert result.success is True

    def test_result_failure_with_errors_no_merges(self):
        """Test failure when there are errors and no merges."""
        result = ConsolidationResult(
            merged_count=0,
            groups_found=0,
            memories_processed=5,
            errors=["Critical error"],
        )

        assert result.success is False

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ConsolidationResult(
            merged_count=3,
            groups_found=1,
            memories_processed=5,
            errors=["warning"],
            merge_details=[{"merged_from": ["id1", "id2"], "merged_into": "id3"}],
        )

        d = result.to_dict()

        assert d["merged_count"] == 3
        assert d["groups_found"] == 1
        assert d["memories_processed"] == 5
        assert d["errors"] == ["warning"]
        assert len(d["merge_details"]) == 1
        assert d["success"] is True


class TestComputeSimilarity:
    """Tests for cosine similarity computation."""

    @pytest.fixture
    def engine(self):
        """Create a consolidation engine with mock storage."""
        storage = MagicMock(spec=StorageBackend)
        return ConsolidationEngine(storage=storage, embedder=MockEmbedder())

    def test_identical_vectors(self, engine):
        """Test similarity of identical vectors is 1.0."""
        emb = [1.0, 0.0, 0.0]
        similarity = engine._compute_similarity(emb, emb)
        assert abs(similarity - 1.0) < 0.0001

    def test_orthogonal_vectors(self, engine):
        """Test similarity of orthogonal vectors is 0.0."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        similarity = engine._compute_similarity(emb1, emb2)
        assert abs(similarity) < 0.0001

    def test_opposite_vectors(self, engine):
        """Test similarity of opposite vectors is -1.0."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [-1.0, 0.0, 0.0]
        similarity = engine._compute_similarity(emb1, emb2)
        assert abs(similarity + 1.0) < 0.0001

    def test_partial_similarity(self, engine):
        """Test similarity of partially similar vectors."""
        emb1 = [1.0, 1.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity = engine._compute_similarity(emb1, emb2)
        # cos(45 degrees) = sqrt(2)/2 ~ 0.707
        assert 0.70 < similarity < 0.72

    def test_different_length_vectors_returns_zero(self, engine):
        """Test that different length vectors return 0.0."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0]
        similarity = engine._compute_similarity(emb1, emb2)
        assert similarity == 0.0

    def test_zero_vector_returns_zero(self, engine):
        """Test that zero vectors return 0.0 similarity."""
        emb1 = [0.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        similarity = engine._compute_similarity(emb1, emb2)
        assert similarity == 0.0


class TestFindSimilarGroups:
    """Tests for grouping memories by similarity."""

    @pytest.fixture
    def engine(self):
        """Create a consolidation engine with mock storage."""
        storage = MagicMock(spec=StorageBackend)
        return ConsolidationEngine(storage=storage, embedder=MockEmbedder())

    def test_empty_memories_returns_empty_groups(self, engine):
        """Test that empty input returns empty groups."""
        groups = engine._find_similar_groups([], threshold=0.85)
        assert groups == []

    def test_single_memory_returns_single_group(self, engine):
        """Test that single memory returns one group."""
        memory = MagicMock()
        memory.embedding = [1.0, 0.0, 0.0]

        groups = engine._find_similar_groups([memory], threshold=0.85)

        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_identical_embeddings_grouped_together(self, engine):
        """Test that memories with identical embeddings are grouped."""
        mem1 = MagicMock()
        mem1.embedding = [1.0, 0.0, 0.0]
        mem2 = MagicMock()
        mem2.embedding = [1.0, 0.0, 0.0]

        groups = engine._find_similar_groups([mem1, mem2], threshold=0.85)

        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_dissimilar_embeddings_separate_groups(self, engine):
        """Test that dissimilar memories stay in separate groups."""
        mem1 = MagicMock()
        mem1.embedding = [1.0, 0.0, 0.0]
        mem2 = MagicMock()
        mem2.embedding = [0.0, 1.0, 0.0]

        groups = engine._find_similar_groups([mem1, mem2], threshold=0.85)

        assert len(groups) == 2
        assert all(len(g) == 1 for g in groups)

    def test_threshold_boundary(self, engine):
        """Test similarity at threshold boundary."""
        # Create embeddings with known similarity
        mem1 = MagicMock()
        mem1.embedding = [1.0, 0.0, 0.0]
        mem2 = MagicMock()
        # This has cosine similarity of ~0.866 with mem1
        mem2.embedding = [0.866, 0.5, 0.0]

        # Should be grouped at 0.85 threshold
        groups = engine._find_similar_groups([mem1, mem2], threshold=0.85)
        assert len(groups) == 1

        # Should be separate at 0.90 threshold
        groups = engine._find_similar_groups([mem1, mem2], threshold=0.90)
        assert len(groups) == 2

    def test_transitive_grouping(self, engine):
        """Test that grouping is transitive (A~B, B~C -> A,B,C grouped)."""
        # A and B are similar, B and C are similar, but A and C might not be
        mem_a = MagicMock()
        mem_a.embedding = [1.0, 0.0, 0.0]
        mem_b = MagicMock()
        mem_b.embedding = [0.9, 0.436, 0.0]  # ~0.90 similarity to A
        mem_c = MagicMock()
        mem_c.embedding = [0.75, 0.66, 0.0]  # ~0.90 similarity to B

        groups = engine._find_similar_groups([mem_a, mem_b, mem_c], threshold=0.85)

        # All should be in one group due to transitivity
        assert len(groups) == 1
        assert len(groups[0]) == 3


class TestMergeHeuristics:
    """Tests for merging heuristics without LLM."""

    @pytest.fixture
    def engine(self):
        """Create a consolidation engine with mock storage."""
        storage = MagicMock(spec=StorageBackend)
        return ConsolidationEngine(storage=storage, embedder=MockEmbedder())

    @pytest.fixture
    def sample_heuristics(self):
        """Create sample heuristics for merging."""
        now = datetime.now(timezone.utc)
        return [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="form testing with validation",
                strategy="validate inputs first",
                confidence=0.85,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
                embedding=[1.0, 0.0, 0.0],
            ),
            Heuristic(
                id="h2",
                agent="test-agent",
                project_id="test-project",
                condition="form testing with input validation",
                strategy="check inputs before submit",
                confidence=0.90,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[0.98, 0.2, 0.0],
            ),
            Heuristic(
                id="h3",
                agent="test-agent",
                project_id="test-project",
                condition="validating form inputs",
                strategy="validate first then test",
                confidence=0.75,
                occurrence_count=3,
                success_count=2,
                last_validated=now,
                created_at=now,
                embedding=[0.95, 0.3, 0.0],
            ),
        ]

    @pytest.mark.asyncio
    async def test_merge_heuristics_without_llm(self, engine, sample_heuristics):
        """Test merging heuristics without LLM uses highest confidence."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        # Should use highest confidence heuristic as base
        assert (
            merged.condition == sample_heuristics[1].condition
        )  # h2 has 0.90 confidence
        assert merged.strategy == sample_heuristics[1].strategy

    @pytest.mark.asyncio
    async def test_merged_occurrence_count_is_sum(self, engine, sample_heuristics):
        """Test that occurrence count is sum of all merged heuristics."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        expected_total = sum(h.occurrence_count for h in sample_heuristics)
        assert merged.occurrence_count == expected_total  # 10 + 5 + 3 = 18

    @pytest.mark.asyncio
    async def test_merged_success_count_is_sum(self, engine, sample_heuristics):
        """Test that success count is sum of all merged heuristics."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        expected_total = sum(h.success_count for h in sample_heuristics)
        assert merged.success_count == expected_total  # 9 + 4 + 2 = 15

    @pytest.mark.asyncio
    async def test_merged_confidence_is_average(self, engine, sample_heuristics):
        """Test that confidence is average of all merged heuristics."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        expected_avg = sum(h.confidence for h in sample_heuristics) / len(
            sample_heuristics
        )
        assert abs(merged.confidence - expected_avg) < 0.001

    @pytest.mark.asyncio
    async def test_merged_has_provenance_metadata(self, engine, sample_heuristics):
        """Test that merged heuristic has merged_from in metadata."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        assert "merged_from" in merged.metadata
        assert set(merged.metadata["merged_from"]) == {"h1", "h2", "h3"}
        assert "merge_timestamp" in merged.metadata
        assert merged.metadata["original_count"] == 3

    @pytest.mark.asyncio
    async def test_merged_has_new_id(self, engine, sample_heuristics):
        """Test that merged heuristic has a new unique ID."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        assert merged.id not in ["h1", "h2", "h3"]
        assert merged.id.startswith("heuristic_")

    @pytest.mark.asyncio
    async def test_merged_has_embedding(self, engine, sample_heuristics):
        """Test that merged heuristic has computed embedding."""
        merged = await engine._merge_heuristics(
            group=sample_heuristics,
            use_llm=False,
            project_id="test-project",
            agent="test-agent",
        )

        assert merged.embedding is not None
        assert len(merged.embedding) > 0


class TestDryRunMode:
    """Tests for dry run mode."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock(spec=StorageBackend)
        now = datetime.now(timezone.utc)

        # Return sample heuristics
        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="form testing",
                strategy="validate inputs",
                confidence=0.85,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
                embedding=[1.0, 0.0, 0.0] + [0.0] * 381,  # 384 dim for MockEmbedder
            ),
            Heuristic(
                id="h2",
                agent="test-agent",
                project_id="test-project",
                condition="form testing again",
                strategy="validate inputs again",
                confidence=0.90,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[1.0, 0.0, 0.0] + [0.0] * 381,  # Same embedding = similar
            ),
        ]

        return storage

    @pytest.mark.asyncio
    async def test_dry_run_does_not_save(self, mock_storage):
        """Test that dry run mode does not save merged memory."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            dry_run=True,
        )

        # Should not save
        mock_storage.save_heuristic.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_delete(self, mock_storage):
        """Test that dry run mode does not delete originals."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            dry_run=True,
        )

        # Should not delete
        mock_storage.delete_heuristic.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_still_reports_results(self, mock_storage):
        """Test that dry run still reports what would be merged."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            dry_run=True,
        )

        # Should still report findings
        assert result.memories_processed == 2
        assert result.groups_found == 1
        assert result.merged_count == 1  # 2 memories merged into 1 = 1 merge
        assert len(result.merge_details) == 1


class TestSimilarityThreshold:
    """Tests for similarity threshold behavior."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage with memories of varying similarity."""
        storage = MagicMock(spec=StorageBackend)
        now = datetime.now(timezone.utc)

        # Create 4 heuristics with different similarities
        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="A",
                strategy="strategy A",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[1.0, 0.0, 0.0] + [0.0] * 381,
            ),
            Heuristic(
                id="h2",
                agent="test-agent",
                project_id="test-project",
                condition="B",
                strategy="strategy B",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[0.95, 0.3, 0.0] + [0.0] * 381,  # ~0.95 similar to h1
            ),
            Heuristic(
                id="h3",
                agent="test-agent",
                project_id="test-project",
                condition="C",
                strategy="strategy C",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[0.7, 0.7, 0.0] + [0.0] * 381,  # ~0.7 similar to h1
            ),
            Heuristic(
                id="h4",
                agent="test-agent",
                project_id="test-project",
                condition="D",
                strategy="strategy D",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[0.0, 1.0, 0.0] + [0.0] * 381,  # 0 similar to h1
            ),
        ]

        return storage

    @pytest.mark.asyncio
    async def test_high_threshold_fewer_merges(self, mock_storage):
        """Test that high threshold results in fewer merges."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.98,  # Very high threshold
            dry_run=True,
        )

        # At 0.98, no memories should be grouped (only h1 and h2 are ~0.95 similar)
        assert result.groups_found == 0
        assert result.merged_count == 0

    @pytest.mark.asyncio
    async def test_low_threshold_more_merges(self, mock_storage):
        """Test that low threshold results in more merges."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.65,  # Low threshold
            dry_run=True,
        )

        # At 0.65, h1, h2, h3 should be grouped (h3 is ~0.7 similar to h1)
        # h4 stays separate
        assert result.groups_found >= 1
        assert result.merged_count >= 1


class TestConsolidateIntegration:
    """Integration tests for the consolidate method."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock(spec=StorageBackend)
        now = datetime.now(timezone.utc)

        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="form testing",
                strategy="validate inputs",
                confidence=0.85,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
                embedding=[1.0] + [0.0] * 383,
            ),
            Heuristic(
                id="h2",
                agent="test-agent",
                project_id="test-project",
                condition="form testing similar",
                strategy="validate inputs too",
                confidence=0.90,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[1.0] + [0.0] * 383,  # Identical = will merge
            ),
        ]

        return storage

    @pytest.mark.asyncio
    async def test_consolidate_saves_merged_when_not_dry_run(self, mock_storage):
        """Test that consolidate saves merged memory when not dry run."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            dry_run=False,
        )

        # Should save merged heuristic
        assert mock_storage.save_heuristic.call_count == 1

    @pytest.mark.asyncio
    async def test_consolidate_deletes_originals_when_not_dry_run(self, mock_storage):
        """Test that consolidate deletes original memories when not dry run."""
        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            similarity_threshold=0.85,
            dry_run=False,
        )

        # Should delete both original heuristics
        assert mock_storage.delete_heuristic.call_count == 2
        mock_storage.delete_heuristic.assert_any_call("h1")
        mock_storage.delete_heuristic.assert_any_call("h2")

    @pytest.mark.asyncio
    async def test_consolidate_with_not_enough_memories(self, mock_storage):
        """Test consolidate with less than 2 memories."""
        mock_storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="only one",
                strategy="strategy",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )
        ]

        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            dry_run=True,
        )

        assert result.memories_processed == 1
        assert result.merged_count == 0
        assert result.groups_found == 0

    @pytest.mark.asyncio
    async def test_consolidate_handles_different_memory_types(self, mock_storage):
        """Test that consolidate works with different memory types."""
        now = datetime.now(timezone.utc)

        # Setup domain knowledge
        mock_storage.get_domain_knowledge.return_value = [
            DomainKnowledge(
                id="dk1",
                agent="test-agent",
                project_id="test-project",
                domain="auth",
                fact="JWT tokens expire in 24h",
                source="code",
                confidence=0.9,
                last_verified=now,
                embedding=[1.0] + [0.0] * 383,
            ),
            DomainKnowledge(
                id="dk2",
                agent="test-agent",
                project_id="test-project",
                domain="auth",
                fact="JWT tokens have 24 hour expiry",
                source="docs",
                confidence=0.85,
                last_verified=now,
                embedding=[1.0] + [0.0] * 383,
            ),
        ]

        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="domain_knowledge",
            dry_run=True,
        )

        assert result.memories_processed == 2
        assert result.groups_found == 1

    @pytest.mark.asyncio
    async def test_consolidate_handles_errors_gracefully(self, mock_storage):
        """Test that errors in one group don't stop processing others."""
        mock_storage.get_heuristics.side_effect = Exception("Database error")

        engine = ConsolidationEngine(storage=mock_storage, embedder=MockEmbedder())

        result = await engine.consolidate(
            agent="test-agent",
            project_id="test-project",
            memory_type="heuristics",
            dry_run=True,
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "Database error" in result.errors[0]


class TestMCPConsolidateTool:
    """Tests for the alma_consolidate MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.storage = MagicMock(spec=StorageBackend)
        alma.retrieval = MagicMock()

        now = datetime.now(timezone.utc)
        alma.storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="test-agent",
                project_id="test-project",
                condition="form testing",
                strategy="validate inputs",
                confidence=0.85,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
                embedding=[1.0] + [0.0] * 383,
            ),
            Heuristic(
                id="h2",
                agent="test-agent",
                project_id="test-project",
                condition="form testing similar",
                strategy="validate inputs too",
                confidence=0.90,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=[1.0] + [0.0] * 383,
            ),
        ]

        return alma

    @pytest.mark.asyncio
    async def test_consolidate_tool_validates_agent(self, mock_alma):
        """Test that the tool validates agent parameter."""
        from alma.mcp.tools import alma_consolidate

        result = await alma_consolidate(
            alma=mock_alma,
            agent="",
            memory_type="heuristics",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_consolidate_tool_validates_memory_type(self, mock_alma):
        """Test that the tool validates memory_type parameter."""
        from alma.mcp.tools import alma_consolidate

        result = await alma_consolidate(
            alma=mock_alma,
            agent="test-agent",
            memory_type="invalid_type",
        )

        assert result["success"] is False
        assert "memory_type must be one of" in result["error"]

    @pytest.mark.asyncio
    async def test_consolidate_tool_validates_threshold(self, mock_alma):
        """Test that the tool validates similarity_threshold parameter."""
        from alma.mcp.tools import alma_consolidate

        result = await alma_consolidate(
            alma=mock_alma,
            agent="test-agent",
            similarity_threshold=1.5,
        )

        assert result["success"] is False
        assert "similarity_threshold must be between" in result["error"]

    @pytest.mark.asyncio
    async def test_consolidate_tool_returns_results(self, mock_alma):
        """Test that the tool returns proper results."""
        from alma.mcp.tools import alma_consolidate

        mock_result = ConsolidationResult(
            merged_count=1,
            groups_found=1,
            memories_processed=2,
            errors=[],
            merge_details=[
                {"merged_from": ["h1", "h2"], "merged_into": "h3", "count": 2}
            ],
        )

        # Create a mock engine class that returns our mock instance
        mock_engine_instance = MagicMock()
        mock_engine_instance.consolidate = AsyncMock(return_value=mock_result)

        # Patch in alma.mcp.tools where the import happens dynamically
        with patch.dict(
            "sys.modules",
            {
                "alma.consolidation": MagicMock(
                    ConsolidationEngine=MagicMock(return_value=mock_engine_instance)
                )
            },
        ):
            result = await alma_consolidate(
                alma=mock_alma,
                agent="test-agent",
                memory_type="heuristics",
                dry_run=True,
            )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert "merged_count" in result
        assert "groups_found" in result
        assert "memories_processed" in result
        assert "message" in result

    @pytest.mark.asyncio
    async def test_consolidate_tool_invalidates_cache_when_not_dry_run(self, mock_alma):
        """Test that cache is invalidated after consolidation."""
        from alma.mcp.tools import alma_consolidate

        mock_result = ConsolidationResult(
            merged_count=1,
            groups_found=1,
            memories_processed=2,
            errors=[],
        )

        mock_engine_instance = MagicMock()
        mock_engine_instance.consolidate = AsyncMock(return_value=mock_result)

        with patch.dict(
            "sys.modules",
            {
                "alma.consolidation": MagicMock(
                    ConsolidationEngine=MagicMock(return_value=mock_engine_instance)
                )
            },
        ):
            result = await alma_consolidate(
                alma=mock_alma,
                agent="test-agent",
                memory_type="heuristics",
                dry_run=False,
            )

        # Cache should be invalidated after actual consolidation
        assert result["success"] is True
        if result["merged_count"] > 0:
            mock_alma.retrieval.invalidate_cache.assert_called()


class TestPrompts:
    """Tests for consolidation prompts."""

    def test_merge_heuristics_prompt_has_placeholders(self):
        """Test that heuristics prompt has required placeholders."""
        assert "{heuristics}" in MERGE_HEURISTICS_PROMPT

    def test_merge_domain_knowledge_prompt_has_placeholders(self):
        """Test that domain knowledge prompt has required placeholders."""
        assert "{knowledge_items}" in MERGE_DOMAIN_KNOWLEDGE_PROMPT

    def test_merge_anti_patterns_prompt_has_placeholders(self):
        """Test that anti-patterns prompt has required placeholders."""
        assert "{anti_patterns}" in MERGE_ANTI_PATTERNS_PROMPT

    def test_merge_heuristics_prompt_requests_json(self):
        """Test that prompts request JSON output."""
        assert "JSON" in MERGE_HEURISTICS_PROMPT
        assert "condition" in MERGE_HEURISTICS_PROMPT
        assert "strategy" in MERGE_HEURISTICS_PROMPT
        assert "confidence" in MERGE_HEURISTICS_PROMPT
