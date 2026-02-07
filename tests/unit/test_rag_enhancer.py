"""Tests for the MemoryEnhancer -- memory signal computation and boost/demotion."""

from datetime import datetime, timezone

from alma.rag.enhancer import MemoryEnhancer
from alma.rag.types import RAGChunk
from alma.types import AntiPattern, DomainKnowledge, Heuristic, MemorySlice, Outcome


def _make_heuristic(id_: str, condition: str, strategy: str, confidence: float = 0.8) -> Heuristic:
    return Heuristic(
        id=id_,
        agent="test",
        project_id="proj-1",
        condition=condition,
        strategy=strategy,
        confidence=confidence,
        occurrence_count=5,
        success_count=4,
        last_validated=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


def _make_outcome(id_: str, task: str, strategy: str, success: bool) -> Outcome:
    return Outcome(
        id=id_,
        agent="test",
        project_id="proj-1",
        task_type="testing",
        task_description=task,
        success=success,
        strategy_used=strategy,
        timestamp=datetime.now(timezone.utc),
    )


def _make_anti_pattern(id_: str, pattern: str, why: str) -> AntiPattern:
    return AntiPattern(
        id=id_,
        agent="test",
        project_id="proj-1",
        pattern=pattern,
        why_bad=why,
        better_alternative="Use a better approach",
        occurrence_count=3,
        last_seen=datetime.now(timezone.utc),
    )


class TestMemoryEnhancerChunks:
    """Tests for chunk enhancement logic."""

    def test_enhance_empty_chunks(self):
        enhancer = MemoryEnhancer()
        result = enhancer.enhance_chunks([], MemorySlice())
        assert result == []

    def test_chunks_sorted_by_enhanced_score(self):
        enhancer = MemoryEnhancer()
        chunks = [
            RAGChunk(id="c1", text="low relevance text", score=0.3),
            RAGChunk(id="c2", text="high relevance text about auth", score=0.9),
        ]
        result = enhancer.enhance_chunks(chunks, MemorySlice())
        assert result[0].chunk.id == "c2"  # Higher original score
        assert result[1].chunk.id == "c1"

    def test_ranks_assigned(self):
        enhancer = MemoryEnhancer()
        chunks = [
            RAGChunk(id="c1", text="aaa", score=0.5),
            RAGChunk(id="c2", text="bbb", score=0.8),
        ]
        result = enhancer.enhance_chunks(chunks, MemorySlice())
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_boost_on_matching_heuristic(self):
        enhancer = MemoryEnhancer()
        chunks = [
            RAGChunk(id="c1", text="use retry with exponential backoff for API calls", score=0.5),
            RAGChunk(id="c2", text="unrelated database migration topic", score=0.5),
        ]
        memory = MemorySlice(
            heuristics=[
                _make_heuristic("h1", "API error handling", "use retry with exponential backoff"),
            ],
            outcomes=[
                _make_outcome("o1", "handle API retry with exponential backoff", "retry", True),
                _make_outcome("o2", "handle API retry with exponential backoff", "retry", True),
            ],
        )
        result = enhancer.enhance_chunks(chunks, memory)

        # Chunk c1 should be boosted because it matches heuristic h1
        c1 = next(e for e in result if e.chunk.id == "c1")
        c2 = next(e for e in result if e.chunk.id == "c2")
        assert c1.enhanced_score >= c2.enhanced_score
        assert c1.signals.boost_factor > 1.0

    def test_penalty_on_anti_pattern(self):
        enhancer = MemoryEnhancer()
        chunks = [
            RAGChunk(id="c1", text="use fixed sleep waits for async operations", score=0.8),
        ]
        memory = MemorySlice(
            anti_patterns=[
                _make_anti_pattern("ap1", "fixed sleep waits", "causes flaky tests"),
            ],
        )
        result = enhancer.enhance_chunks(chunks, memory)
        c1 = result[0]
        assert c1.signals.boost_factor < 1.0
        assert len(c1.signals.anti_pattern_warnings) >= 1


class TestMemoryEnhancerAugmentation:
    """Tests for augmentation text generation."""

    def test_empty_memory(self):
        enhancer = MemoryEnhancer()
        text = enhancer.generate_augmentation(MemorySlice())
        assert text == ""

    def test_heuristics_in_augmentation(self):
        enhancer = MemoryEnhancer()
        memory = MemorySlice(
            heuristics=[
                _make_heuristic("h1", "API testing", "use retry logic"),
            ],
        )
        text = enhancer.generate_augmentation(memory)
        assert "Learned Strategies" in text
        assert "retry logic" in text

    def test_anti_patterns_in_augmentation(self):
        enhancer = MemoryEnhancer()
        memory = MemorySlice(
            anti_patterns=[
                _make_anti_pattern("ap1", "hardcoded URLs", "breaks in staging"),
            ],
        )
        text = enhancer.generate_augmentation(memory)
        assert "Anti-Patterns" in text
        assert "hardcoded URLs" in text

    def test_token_budget_truncation(self):
        enhancer = MemoryEnhancer()
        memory = MemorySlice(
            heuristics=[_make_heuristic(f"h{i}", f"condition {i}", f"strategy {i}") for i in range(20)],
        )
        text = enhancer.generate_augmentation(memory, max_tokens=10)
        # ~10 tokens * 4 chars = 40 chars max + truncation marker
        assert len(text) <= 60
        assert "[truncated]" in text

    def test_domain_knowledge_in_augmentation(self):
        enhancer = MemoryEnhancer()
        dk = DomainKnowledge(
            id="dk1",
            agent="test",
            project_id="proj-1",
            domain="auth",
            fact="Login uses JWT with 24h expiry",
            source="user_stated",
            last_verified=datetime.now(timezone.utc),
        )
        memory = MemorySlice(domain_knowledge=[dk])
        text = enhancer.generate_augmentation(memory)
        assert "Domain Context" in text
        assert "JWT" in text
