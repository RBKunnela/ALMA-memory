"""Tests for RAG type dataclasses."""

from alma.rag.types import EnhancedChunk, MemorySignals, RAGChunk, RAGContext


class TestRAGChunk:
    def test_basic_construction(self):
        chunk = RAGChunk(id="c1", text="hello world", score=0.85)
        assert chunk.id == "c1"
        assert chunk.text == "hello world"
        assert chunk.score == 0.85
        assert chunk.source == ""
        assert chunk.metadata == {}
        assert chunk.embedding is None

    def test_full_construction(self):
        chunk = RAGChunk(
            id="c2",
            text="JWT auth guide",
            score=0.9,
            source="docs/auth.md",
            metadata={"page": 3},
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.source == "docs/auth.md"
        assert chunk.metadata["page"] == 3
        assert len(chunk.embedding) == 3


class TestMemorySignals:
    def test_defaults(self):
        signals = MemorySignals()
        assert signals.related_heuristics == []
        assert signals.related_outcomes == []
        assert signals.trust_score == 0.5
        assert signals.historical_success_rate == 0.5
        assert signals.confidence == 0.0
        assert signals.anti_pattern_warnings == []
        assert signals.boost_factor == 1.0

    def test_custom_values(self):
        signals = MemorySignals(
            related_heuristics=["h1", "h2"],
            trust_score=0.9,
            boost_factor=1.5,
        )
        assert len(signals.related_heuristics) == 2
        assert signals.boost_factor == 1.5


class TestEnhancedChunk:
    def test_construction(self):
        chunk = RAGChunk(id="c1", text="test", score=0.8)
        signals = MemorySignals(boost_factor=1.2)
        enhanced = EnhancedChunk(
            chunk=chunk,
            signals=signals,
            enhanced_score=0.96,
            rank=1,
        )
        assert enhanced.chunk.id == "c1"
        assert enhanced.enhanced_score == 0.96
        assert enhanced.rank == 1


class TestRAGContext:
    def test_defaults(self):
        ctx = RAGContext()
        assert ctx.enhanced_chunks == []
        assert ctx.memory_augmentation == ""
        assert ctx.total_chunks == 0

    def test_full_context(self):
        chunk = RAGChunk(id="c1", text="test", score=0.8)
        signals = MemorySignals()
        enhanced = EnhancedChunk(chunk=chunk, signals=signals, enhanced_score=0.8)

        ctx = RAGContext(
            enhanced_chunks=[enhanced],
            memory_augmentation="## Strategies\n- Use retry logic",
            query="how to handle errors",
            agent="backend-agent",
            total_chunks=1,
            token_budget_used=15,
        )
        assert len(ctx.enhanced_chunks) == 1
        assert ctx.agent == "backend-agent"
        assert ctx.token_budget_used == 15
