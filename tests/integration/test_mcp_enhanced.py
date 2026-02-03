"""
Integration tests for Memory Wall enhanced MCP tools (v0.7.0).

Tests the new MCP tools: reinforce, get_weak_memories, smart_forget,
retrieve_verified, compress_and_learn, and extract_heuristic.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alma.mcp.tools import (
    alma_compress_and_learn,
    alma_extract_heuristic,
    alma_get_weak_memories,
    alma_reinforce,
    alma_retrieve_verified,
    alma_smart_forget,
)
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import Heuristic, Outcome


class MockALMA:
    """Mock ALMA instance for integration testing."""

    def __init__(self, storage, project_id="test-project"):
        self.storage = storage
        self.project_id = project_id
        self.retrieval = MockRetrievalEngine(storage)
        self.llm = None


class MockRetrievalEngine:
    """Mock retrieval engine for testing."""

    def __init__(self, storage):
        self.storage = storage

    def retrieve(self, query, agent, project_id, top_k=5, **kwargs):
        """Return empty list for testing."""
        return []


class TestReinforceIntegration:
    """Integration tests for alma_reinforce."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_reinforce.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_reinforce_creates_strength_record(self, alma):
        """Should create strength record when reinforcing."""
        # First, save a memory to reinforce
        now = datetime.now(timezone.utc)
        heuristic = Heuristic(
            id="heur-test-1",
            agent="helena",
            project_id="test-project",
            condition="testing",
            strategy="test strategy",
            confidence=0.8,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        alma.storage.save_heuristic(heuristic)

        # Reinforce it
        result = alma_reinforce(
            alma=alma,
            memory_id="heur-test-1",
            memory_type="heuristic",
        )

        assert result["success"] is True
        assert result["new_strength"] > 0.5
        assert result["status"] == "reinforced"

    def test_reinforce_increases_strength(self, alma):
        """Reinforcing should increase strength."""
        now = datetime.now(timezone.utc)
        outcome = Outcome(
            id="out-test-1",
            agent="helena",
            project_id="test-project",
            task_type="testing",
            task_description="Test task",
            success=True,
            strategy_used="test",
            timestamp=now,
        )
        alma.storage.save_outcome(outcome)

        # Reinforce multiple times
        result1 = alma_reinforce(alma=alma, memory_id="out-test-1")
        result2 = alma_reinforce(alma=alma, memory_id="out-test-1")

        # Second reinforcement should be higher or equal
        assert result2["new_strength"] >= result1["new_strength"]


class TestGetWeakMemoriesIntegration:
    """Integration tests for alma_get_weak_memories."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_weak.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_get_weak_memories_empty(self, alma):
        """Should return empty when no weak memories."""
        result = alma_get_weak_memories(alma=alma)

        assert result["success"] is True
        assert result["count"] == 0
        assert result["weak_memories"] == []

    def test_get_weak_memories_with_agent_filter(self, alma):
        """Should filter by agent."""
        result = alma_get_weak_memories(alma=alma, agent="nonexistent")

        assert result["success"] is True
        assert result["count"] == 0


class TestSmartForgetIntegration:
    """Integration tests for alma_smart_forget."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_forget.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_smart_forget_dry_run_empty(self, alma):
        """Dry run with no forgettable memories."""
        result = alma_smart_forget(alma=alma, dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["count"] == 0

    def test_smart_forget_with_threshold(self, alma):
        """Should respect threshold parameter."""
        result = alma_smart_forget(alma=alma, threshold=0.5, dry_run=True)

        assert result["success"] is True
        # Higher threshold means more would be forgotten


class TestRetrieveVerifiedIntegration:
    """Integration tests for alma_retrieve_verified."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_verified.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_retrieve_verified_basic(self, alma):
        """Should return verified results structure."""
        result = alma_retrieve_verified(
            alma=alma,
            query="How to handle errors?",
            agent="helena",
        )

        assert result["success"] is True
        assert "verified" in result
        assert "uncertain" in result
        assert "contradicted" in result
        assert "summary" in result

    def test_retrieve_verified_with_ground_truth(self, alma):
        """Should use ground truth for verification."""
        result = alma_retrieve_verified(
            alma=alma,
            query="API endpoint structure",
            agent="helena",
            ground_truth=["The API uses REST conventions."],
        )

        assert result["success"] is True


class TestCompressAndLearnIntegration:
    """Integration tests for alma_compress_and_learn."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_compress.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_compress_and_learn_outcome(self, alma):
        """Should compress and store as outcome."""
        verbose_content = (
            "The task involved implementing a new feature for user authentication. "
            "We started by analyzing the existing codebase and identified several "
            "areas that needed modification. The main challenge was ensuring backwards "
            "compatibility with the existing session management. We implemented a new "
            "token-based system that works alongside the old cookie-based approach. "
            "Testing revealed some edge cases that needed handling. The final solution "
            "includes rate limiting to prevent abuse."
        )

        result = alma_compress_and_learn(
            alma=alma,
            content=verbose_content,
            agent="helena",
            memory_type="outcome",
            compression_level="medium",
        )

        assert result["success"] is True
        assert result["memory_type"] == "outcome"
        assert result["compression_ratio"] > 1.0
        assert "memory_id" in result

        # Verify it was stored
        outcomes = alma.storage.get_outcomes(project_id="test-project")
        assert len(outcomes) == 1

    def test_compress_and_learn_knowledge(self, alma):
        """Should compress and store as domain knowledge."""
        content = (
            "The API uses REST conventions with JSON responses. "
            "All endpoints require authentication via bearer tokens. "
            "Rate limiting is set at 100 requests per minute per user. "
            "Pagination is handled via cursor-based pagination. "
            "Error responses follow RFC 7807 problem details format."
        )

        result = alma_compress_and_learn(
            alma=alma,
            content=content,
            agent="helena",
            memory_type="knowledge",
            task_type="api",
        )

        assert result["success"] is True
        assert result["memory_type"] == "knowledge"

        # Verify it was stored
        knowledge = alma.storage.get_domain_knowledge(project_id="test-project")
        assert len(knowledge) == 1

    def test_compress_and_learn_heuristic(self, alma):
        """Should compress and store as heuristic."""
        content = (
            "When implementing authentication, always use token-based auth. "
            "This approach is more secure than session-based authentication. "
            "Token rotation should happen every 24 hours minimum. "
            "Always store tokens securely and never in localStorage."
        )

        result = alma_compress_and_learn(
            alma=alma,
            content=content,
            agent="helena",
            memory_type="heuristic",
        )

        assert result["success"] is True
        assert result["memory_type"] == "heuristic"

        # Verify it was stored
        heuristics = alma.storage.get_heuristics(project_id="test-project")
        assert len(heuristics) == 1


class TestExtractHeuristicIntegration:
    """Integration tests for alma_extract_heuristic."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_extract.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_extract_heuristic_with_pattern(self, alma):
        """Should extract heuristic from similar experiences."""
        experiences = [
            "Deployed code first, migration failed, had to rollback",
            "Ran migration before deploy, everything worked smoothly",
            "Tried simultaneous deploy and migrate, caused race condition",
            "Migration-first approach successful again",
            "Forgot migration, app crashed on new column access",
        ]

        result = alma_extract_heuristic(
            alma=alma,
            experiences=experiences,
            agent="helena",
            auto_save=True,
        )

        assert result["success"] is True
        # Without LLM, may not find a pattern - that's ok
        if result.get("heuristic"):
            assert result["source_count"] == 5
            assert result["saved"] is True
            # Verify it was stored
            heuristics = alma.storage.get_heuristics(project_id="test-project")
            assert len(heuristics) == 1
        else:
            # No pattern found without LLM
            assert "No clear pattern" in result.get("message", "")

    def test_extract_heuristic_no_auto_save(self, alma):
        """Should not save when auto_save is False."""
        experiences = ["E1 with test", "E2 with test", "E3 with test"]

        result = alma_extract_heuristic(
            alma=alma,
            experiences=experiences,
            agent="helena",
            auto_save=False,
        )

        assert result["success"] is True
        # Without LLM, may not find pattern
        if result.get("heuristic"):
            assert result["saved"] is False
        # Verify nothing was stored regardless
        heuristics = alma.storage.get_heuristics(project_id="test-project")
        assert len(heuristics) == 0


class TestToolsEndToEnd:
    """End-to-end tests combining multiple tools."""

    @pytest.fixture
    def alma(self):
        """Create ALMA with real SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_e2e.db"
            storage = SQLiteStorage(db_path=db_path)
            yield MockALMA(storage)

    def test_compress_learn_then_reinforce(self, alma):
        """Should be able to compress, learn, and then reinforce."""
        # First compress and learn
        content = (
            "The implementation was successful. "
            "We used a caching layer to improve performance. "
            "Response times dropped from 500ms to 50ms. "
            "The cache invalidation strategy uses TTL."
        )

        learn_result = alma_compress_and_learn(
            alma=alma,
            content=content,
            agent="helena",
            memory_type="outcome",
        )

        assert learn_result["success"] is True
        memory_id = learn_result["memory_id"]

        # Then reinforce it
        reinforce_result = alma_reinforce(
            alma=alma,
            memory_id=memory_id,
            memory_type="outcome",
        )

        assert reinforce_result["success"] is True
        assert reinforce_result["new_strength"] > 0.5
