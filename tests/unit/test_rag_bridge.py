"""Tests for the RAGBridge -- full enhancement flow."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from alma.rag.bridge import RAGBridge
from alma.rag.types import RAGChunk, RAGContext
from alma.types import DomainKnowledge, Heuristic, MemorySlice


def _make_mock_alma():
    """Create a mock ALMA instance."""
    alma = MagicMock()
    alma.project_id = "proj-1"
    # Default: return empty memory slice
    alma.retrieve.return_value = MemorySlice(
        query="test query",
        agent="test-agent",
        retrieval_time_ms=5,
    )
    return alma


class TestRAGBridge:
    """Tests for the RAGBridge class."""

    def test_enhance_empty_chunks(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        result = bridge.enhance(
            chunks=[],
            query="test query",
            agent="test-agent",
        )

        assert isinstance(result, RAGContext)
        assert result.total_chunks == 0
        assert result.enhanced_chunks == []
        # Should NOT call alma.retrieve for empty input
        alma.retrieve.assert_not_called()

    def test_enhance_basic_flow(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        chunks = [
            RAGChunk(id="c1", text="JWT authentication setup guide", score=0.85),
            RAGChunk(id="c2", text="database migration with alembic", score=0.70),
        ]

        result = bridge.enhance(
            chunks=chunks,
            query="how to set up authentication",
            agent="backend-agent",
        )

        assert isinstance(result, RAGContext)
        assert result.total_chunks == 2
        assert len(result.enhanced_chunks) == 2
        assert result.query == "how to set up authentication"
        assert result.agent == "backend-agent"

        # ALMA retrieve should have been called
        alma.retrieve.assert_called_once_with(
            task="how to set up authentication",
            agent="backend-agent",
            user_id=None,
            top_k=5,
        )

    def test_enhance_with_memories(self):
        alma = _make_mock_alma()
        # Return real heuristics
        alma.retrieve.return_value = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="backend-agent",
                    project_id="proj-1",
                    condition="authentication setup",
                    strategy="use JWT with refresh token rotation",
                    confidence=0.9,
                    occurrence_count=5,
                    success_count=4,
                    last_validated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            domain_knowledge=[
                DomainKnowledge(
                    id="dk1",
                    agent="backend-agent",
                    project_id="proj-1",
                    domain="auth",
                    fact="Auth service runs on port 8443",
                    source="user_stated",
                    last_verified=datetime.now(timezone.utc),
                ),
            ],
            query="auth setup",
            agent="backend-agent",
            retrieval_time_ms=10,
        )

        bridge = RAGBridge(alma=alma)
        chunks = [
            RAGChunk(id="c1", text="use JWT with refresh token rotation for auth setup", score=0.9),
        ]

        result = bridge.enhance(
            chunks=chunks,
            query="auth setup",
            agent="backend-agent",
        )

        assert len(result.enhanced_chunks) == 1
        assert result.memory_augmentation != ""
        assert "Learned Strategies" in result.memory_augmentation
        assert "Domain Context" in result.memory_augmentation

    def test_enhance_with_user_id(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        chunks = [RAGChunk(id="c1", text="test", score=0.5)]
        bridge.enhance(
            chunks=chunks,
            query="test",
            agent="test-agent",
            user_id="user-123",
        )

        alma.retrieve.assert_called_once_with(
            task="test",
            agent="test-agent",
            user_id="user-123",
            top_k=5,
        )

    def test_enhanced_chunks_sorted_by_score(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        chunks = [
            RAGChunk(id="c1", text="low score chunk", score=0.3),
            RAGChunk(id="c2", text="high score chunk", score=0.9),
            RAGChunk(id="c3", text="mid score chunk", score=0.6),
        ]

        result = bridge.enhance(chunks=chunks, query="test", agent="test-agent")

        scores = [e.enhanced_score for e in result.enhanced_chunks]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_contains_retrieval_info(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        chunks = [RAGChunk(id="c1", text="test", score=0.5)]
        result = bridge.enhance(chunks=chunks, query="test", agent="test-agent")

        assert "memories_retrieved" in result.metadata
        assert "retrieval_time_ms" in result.metadata

    def test_custom_top_k(self):
        alma = _make_mock_alma()
        bridge = RAGBridge(alma=alma)

        chunks = [RAGChunk(id="c1", text="test", score=0.5)]
        bridge.enhance(chunks=chunks, query="test", agent="test-agent", top_k=10)

        alma.retrieve.assert_called_once_with(
            task="test",
            agent="test-agent",
            user_id=None,
            top_k=10,
        )
