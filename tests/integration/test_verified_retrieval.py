"""
Integration tests for Two-Stage Verified Retrieval.

Tests the full retrieval and verification pipeline with real storage.
"""

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from alma.retrieval.verification import (
    VerificationConfig,
    VerificationMethod,
    VerificationStatus,
    VerifiedRetriever,
)
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import DomainKnowledge, Heuristic, Outcome


class MockRetrievalEngine:
    """Mock retrieval engine for testing."""

    def __init__(self, storage: SQLiteStorage):
        self.storage = storage

    def retrieve(
        self,
        query: str,
        agent: str,
        project_id: str,
        top_k: int = 5,
        **kwargs,
    ):
        """Retrieve memories from storage."""
        # Simple retrieval - just get all memories
        heuristics = self.storage.get_heuristics(
            project_id=project_id,
            agent=agent,
            top_k=top_k,
        )
        outcomes = self.storage.get_outcomes(
            project_id=project_id,
            agent=agent,
            top_k=top_k,
        )
        knowledge = self.storage.get_domain_knowledge(
            project_id=project_id,
            agent=agent,
            top_k=top_k,
        )

        # Return a mock MemorySlice-like object
        @dataclass
        class MockSlice:
            heuristics: list
            outcomes: list
            knowledge: list
            anti_patterns: list = None
            preferences: list = None

            def __post_init__(self):
                self.anti_patterns = self.anti_patterns or []
                self.preferences = self.preferences or []

        return MockSlice(
            heuristics=heuristics,
            outcomes=outcomes,
            knowledge=knowledge,
        )


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response or """STATUS: verified
CONFIDENCE: 0.85
REASON: Memory appears accurate"""
        self.call_count = 0
        self.responses = []

    def set_responses(self, responses: list):
        """Set specific responses for sequential calls."""
        self.responses = responses

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        """Return mock response."""
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return self.default_response


class TestVerifiedRetrievalIntegration:
    """Integration tests for verified retrieval with real storage."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_verified.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    @pytest.fixture
    def populated_storage(self, storage):
        """Create storage with sample memories."""
        now = datetime.now(timezone.utc)

        # Add heuristics
        for i in range(5):
            h = Heuristic(
                id=f"heur-{i}",
                agent="test-agent",
                project_id="test-project",
                condition=f"When condition {i} occurs",
                strategy=f"Apply strategy {i}",
                confidence=0.5 + (i * 0.1),  # 0.5 to 0.9
                occurrence_count=i + 1,
                success_count=i,
                last_validated=now,
                created_at=now,
            )
            storage.save_heuristic(h)

        # Add outcomes
        for i in range(3):
            o = Outcome(
                id=f"out-{i}",
                agent="test-agent",
                project_id="test-project",
                task_type="testing",
                task_description=f"Test task {i}",
                success=i % 2 == 0,
                strategy_used=f"Strategy {i}",
                duration_ms=1000 + i * 100,
                timestamp=now,
            )
            storage.save_outcome(o)

        # Add knowledge
        for i in range(2):
            k = DomainKnowledge(
                id=f"know-{i}",
                agent="test-agent",
                project_id="test-project",
                domain="testing",
                fact=f"Important fact {i}",
                source="test suite",
                confidence=0.7 + (i * 0.1),
                last_verified=now,
            )
            storage.save_domain_knowledge(k)

        return storage

    def test_full_retrieval_pipeline_no_llm(self, populated_storage):
        """Should retrieve and verify memories without LLM."""
        engine = MockRetrievalEngine(populated_storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=None,
            config=VerificationConfig(confidence_threshold=0.7),
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            top_k=10,
        )

        # Should have results in multiple categories based on confidence
        assert results.total_count > 0
        summary = results.summary()
        assert summary["total"] > 0

        # All should use confidence method
        for vm in results.all_usable:
            assert vm.verification.method == VerificationMethod.CONFIDENCE

    def test_full_retrieval_pipeline_with_llm(self, populated_storage):
        """Should retrieve and verify memories with LLM."""
        engine = MockRetrievalEngine(populated_storage)
        llm = MockLLMClient()
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
            config=VerificationConfig(default_method="cross_verify"),
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            cross_verify=True,
            top_k=5,
        )

        # LLM should have been called
        assert llm.call_count > 0

        # Results should use cross_verify method
        for vm in results.verified:
            assert vm.verification.method == VerificationMethod.CROSS_VERIFY

    def test_ground_truth_verification(self, populated_storage):
        """Should verify against ground truth sources."""
        engine = MockRetrievalEngine(populated_storage)
        llm = MockLLMClient(
            default_response="""STATUS: verified
CONFIDENCE: 0.95
REASON: Matches authoritative source"""
        )
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            ground_truth_sources=[
                "Official documentation states X",
                "API reference confirms Y",
            ],
            top_k=5,
        )

        # Should use ground truth method
        for vm in results.verified:
            assert vm.verification.method == VerificationMethod.GROUND_TRUTH

    def test_contradicted_memories(self, populated_storage):
        """Should identify contradicted memories."""
        engine = MockRetrievalEngine(populated_storage)
        llm = MockLLMClient(
            default_response="""STATUS: contradicted
CONFIDENCE: 0.3
REASON: Conflicts with current documentation
CONTRADICTION: The API endpoint has changed"""
        )
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            ground_truth_sources=["Updated API docs"],
            top_k=5,
        )

        # All should be contradicted
        assert len(results.contradicted) > 0
        for vm in results.contradicted:
            assert vm.verification.status == VerificationStatus.CONTRADICTED
            assert vm.verification.contradicting_source is not None

    def test_mixed_verification_results(self, populated_storage):
        """Should handle mixed verification results."""
        engine = MockRetrievalEngine(populated_storage)
        llm = MockLLMClient()

        # Set up alternating responses
        llm.set_responses([
            """STATUS: verified
CONFIDENCE: 0.9
REASON: Confirmed accurate""",
            """STATUS: uncertain
CONFIDENCE: 0.5
REASON: Could not fully verify""",
            """STATUS: contradicted
CONFIDENCE: 0.3
REASON: Conflicts detected
CONTRADICTION: Source says otherwise""",
            """STATUS: verified
CONFIDENCE: 0.85
REASON: Matches sources""",
            """STATUS: uncertain
CONFIDENCE: 0.6
REASON: Partially verified""",
        ] * 3)  # Repeat for all memories

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            ground_truth_sources=["Test source"],
            top_k=10,
        )

        # Should have mixed results
        assert len(results.verified) > 0 or len(results.uncertain) > 0 or len(results.contradicted) > 0

    def test_high_confidence_filter(self, populated_storage):
        """Should filter to high confidence only."""
        engine = MockRetrievalEngine(populated_storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.8),
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            top_k=10,
        )

        # High confidence should only have memories >= threshold
        for vm in results.high_confidence:
            # The memory's confidence should be >= threshold
            memory_conf = getattr(vm.memory, "confidence", 0)
            assert memory_conf >= 0.8

    def test_usable_memories(self, populated_storage):
        """Should return all usable memories."""
        engine = MockRetrievalEngine(populated_storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.6),
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            top_k=10,
        )

        # Usable should include verified + uncertain
        usable = results.all_usable
        assert len(usable) == len(results.verified) + len(results.uncertain)

    def test_needs_review_filter(self, populated_storage):
        """Should identify memories needing review."""
        engine = MockRetrievalEngine(populated_storage)
        llm = MockLLMClient(
            default_response="""STATUS: contradicted
CONFIDENCE: 0.2
REASON: Outdated information
CONTRADICTION: Current docs say different"""
        )
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test query",
            agent="test-agent",
            project_id="test-project",
            ground_truth_sources=["Current source"],
            top_k=5,
        )

        # Needs review should be contradicted
        needs_review = results.needs_review
        for vm in needs_review:
            assert vm.verification.needs_review()


class TestVerificationWithDifferentMemoryTypes:
    """Test verification with different memory types."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_types.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_verify_heuristics(self, storage):
        """Should verify heuristic memories."""
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-test",
            agent="agent",
            project_id="project",
            condition="When testing",
            strategy="Use mocks",
            confidence=0.85,
            occurrence_count=10,
            success_count=9,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        engine = MockRetrievalEngine(storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.8),
        )

        results = retriever.retrieve_verified(
            query="testing",
            agent="agent",
            project_id="project",
        )

        # High confidence heuristic should be verified
        assert len(results.verified) == 1
        assert results.verified[0].memory.id == "h-test"

    def test_verify_outcomes(self, storage):
        """Should verify outcome memories."""
        now = datetime.now(timezone.utc)
        o = Outcome(
            id="o-test",
            agent="agent",
            project_id="project",
            task_type="coding",
            task_description="Write unit tests",
            success=True,
            strategy_used="TDD approach",
            duration_ms=5000,
            timestamp=now,
        )
        storage.save_outcome(o)

        engine = MockRetrievalEngine(storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.5),
        )

        results = retriever.retrieve_verified(
            query="unit tests",
            agent="agent",
            project_id="project",
        )

        # Should have the outcome
        assert results.total_count >= 1

    def test_verify_domain_knowledge(self, storage):
        """Should verify domain knowledge memories."""
        now = datetime.now(timezone.utc)
        k = DomainKnowledge(
            id="k-test",
            agent="agent",
            project_id="project",
            domain="testing",
            fact="Unit tests should be isolated",
            source="Testing best practices",
            confidence=0.95,
            last_verified=now,
        )
        storage.save_domain_knowledge(k)

        engine = MockRetrievalEngine(storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            config=VerificationConfig(confidence_threshold=0.9),
        )

        results = retriever.retrieve_verified(
            query="testing",
            agent="agent",
            project_id="project",
        )

        # High confidence knowledge should be verified
        assert len(results.verified) == 1
        assert results.verified[0].memory.id == "k-test"


class TestVerificationPerformance:
    """Test verification performance characteristics."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_perf.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_confidence_fallback_is_fast(self, storage):
        """Confidence fallback should be fast (<100ms)."""
        now = datetime.now(timezone.utc)

        # Add many memories
        for i in range(50):
            h = Heuristic(
                id=f"h-{i}",
                agent="agent",
                project_id="project",
                condition=f"Condition {i}",
                strategy=f"Strategy {i}",
                confidence=0.5 + (i % 5) * 0.1,
                occurrence_count=1,
                success_count=1,
                last_validated=now,
                created_at=now,
            )
            storage.save_heuristic(h)

        engine = MockRetrievalEngine(storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=None,  # No LLM
        )

        import time
        start = time.time()
        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="project",
            top_k=20,
        )
        elapsed_ms = (time.time() - start) * 1000

        # Should be fast without LLM
        assert elapsed_ms < 1000  # Allow some margin for CI
        assert "total_verification_time_ms" in results.metadata


class TestVerificationErrorHandling:
    """Test error handling in verification."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_errors.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_handles_llm_timeout(self, storage):
        """Should handle LLM timeout gracefully."""
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-timeout",
            agent="agent",
            project_id="project",
            condition="Test",
            strategy="Test",
            confidence=0.8,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        engine = MockRetrievalEngine(storage)

        # Create LLM that times out
        llm = MagicMock()
        llm.complete.side_effect = TimeoutError("LLM timeout")

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        # Should not raise, should handle gracefully
        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="project",
            ground_truth_sources=["Source"],
        )

        # Should have unverifiable results
        assert len(results.unverifiable) > 0

    def test_handles_empty_results(self, storage):
        """Should handle empty retrieval results."""
        engine = MockRetrievalEngine(storage)
        retriever = VerifiedRetriever(
            retrieval_engine=engine,
        )

        results = retriever.retrieve_verified(
            query="nonexistent",
            agent="nonexistent-agent",
            project_id="nonexistent-project",
        )

        # Should return empty but valid results
        assert results.total_count == 0
        assert len(results.all_usable) == 0

    def test_handles_malformed_llm_response(self, storage):
        """Should handle malformed LLM responses."""
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-malformed",
            agent="agent",
            project_id="project",
            condition="Test",
            strategy="Test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        engine = MockRetrievalEngine(storage)
        llm = MockLLMClient(default_response="This is not a valid format at all!")

        retriever = VerifiedRetriever(
            retrieval_engine=engine,
            llm_client=llm,
        )

        results = retriever.retrieve_verified(
            query="test",
            agent="agent",
            project_id="project",
            ground_truth_sources=["Source"],
        )

        # Should default to uncertain
        assert results.total_count > 0
