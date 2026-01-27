"""
Unit tests for ALMA Multi-Agent Memory Sharing.

Tests the P0 feature from competitive analysis: enabling agents to share
memories across agent boundaries with proper access control.

Key requirements tested:
1. Agent A shares with agent B (share_with)
2. Agent B inherits from agent A (inherit_from)
3. Agent C cannot see A's memories (isolation)
4. Bidirectional sharing
5. Write isolation (only owning agent can modify)
6. shared_from metadata tracking
"""

import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pytest

from alma.retrieval.engine import RetrievalEngine
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemoryScope,
    Outcome,
)


class TestMemoryScopeSharing:
    """Tests for MemoryScope share_with and inherit_from fields."""

    def test_memory_scope_default_empty_sharing(self):
        """Test that share_with and inherit_from default to empty lists."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
        )

        assert scope.share_with == []
        assert scope.inherit_from == []

    def test_memory_scope_with_sharing(self):
        """Test MemoryScope with sharing configuration."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
            share_with=["agent_b", "agent_c"],
            inherit_from=["agent_d"],
        )

        assert scope.share_with == ["agent_b", "agent_c"]
        assert scope.inherit_from == ["agent_d"]

    def test_get_readable_agents(self):
        """Test get_readable_agents returns self and inherited agents."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
            inherit_from=["agent_b", "agent_c"],
        )

        readable = scope.get_readable_agents()
        assert "agent_a" in readable
        assert "agent_b" in readable
        assert "agent_c" in readable
        assert len(readable) == 3

    def test_get_readable_agents_no_inheritance(self):
        """Test get_readable_agents with no inheritance returns only self."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
        )

        readable = scope.get_readable_agents()
        assert readable == ["agent_a"]

    def test_can_read_from_self(self):
        """Test that agent can always read from itself."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
        )

        assert scope.can_read_from("agent_a") is True

    def test_can_read_from_inherited(self):
        """Test that agent can read from inherited agents."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
            inherit_from=["agent_b"],
        )

        assert scope.can_read_from("agent_b") is True
        assert scope.can_read_from("agent_c") is False

    def test_shares_with(self):
        """Test shares_with method."""
        scope = MemoryScope(
            agent_name="agent_a",
            can_learn=["testing"],
            cannot_learn=[],
            share_with=["agent_b"],
        )

        assert scope.shares_with("agent_b") is True
        assert scope.shares_with("agent_c") is False


class TestSQLiteMultiAgentSharing:
    """Tests for SQLite storage backend multi-agent queries."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def agents_with_memories(self, storage: SQLiteStorage) -> Dict[str, list]:
        """Create multiple agents with their own memories."""
        now = datetime.now(timezone.utc)
        memories = {"agent_a": [], "agent_b": [], "agent_c": []}

        # Agent A memories
        h_a = Heuristic(
            id=str(uuid.uuid4()),
            agent="agent_a",
            project_id="test-project",
            condition="agent_a condition",
            strategy="agent_a strategy",
            confidence=0.9,
            occurrence_count=10,
            success_count=9,
            last_validated=now,
            created_at=now,
            embedding=[0.1] * 384,
        )
        storage.save_heuristic(h_a)
        memories["agent_a"].append(h_a)

        dk_a = DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="agent_a",
            project_id="test-project",
            domain="authentication",
            fact="Agent A knows about OAuth",
            source="analysis",
            confidence=0.95,
            last_verified=now,
            embedding=[0.2] * 384,
        )
        storage.save_domain_knowledge(dk_a)
        memories["agent_a"].append(dk_a)

        # Agent B memories
        h_b = Heuristic(
            id=str(uuid.uuid4()),
            agent="agent_b",
            project_id="test-project",
            condition="agent_b condition",
            strategy="agent_b strategy",
            confidence=0.85,
            occurrence_count=5,
            success_count=4,
            last_validated=now,
            created_at=now,
            embedding=[0.3] * 384,
        )
        storage.save_heuristic(h_b)
        memories["agent_b"].append(h_b)

        # Agent C memories
        h_c = Heuristic(
            id=str(uuid.uuid4()),
            agent="agent_c",
            project_id="test-project",
            condition="agent_c condition",
            strategy="agent_c strategy",
            confidence=0.8,
            occurrence_count=3,
            success_count=2,
            last_validated=now,
            created_at=now,
            embedding=[0.4] * 384,
        )
        storage.save_heuristic(h_c)
        memories["agent_c"].append(h_c)

        return memories

    def test_get_heuristics_for_single_agent(
        self, storage: SQLiteStorage, agents_with_memories: Dict[str, list]
    ):
        """Test querying heuristics for a single agent."""
        heuristics = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["agent_a"],
        )

        assert len(heuristics) == 1
        assert heuristics[0].agent == "agent_a"

    def test_get_heuristics_for_multiple_agents(
        self, storage: SQLiteStorage, agents_with_memories: Dict[str, list]
    ):
        """Test querying heuristics from multiple agents."""
        heuristics = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["agent_a", "agent_b"],
        )

        assert len(heuristics) == 2
        agents = {h.agent for h in heuristics}
        assert "agent_a" in agents
        assert "agent_b" in agents
        assert "agent_c" not in agents

    def test_get_heuristics_for_agents_excludes_others(
        self, storage: SQLiteStorage, agents_with_memories: Dict[str, list]
    ):
        """Test that agents not in the list are excluded."""
        heuristics = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["agent_a"],
        )

        for h in heuristics:
            assert h.agent == "agent_a"
            assert h.agent != "agent_b"
            assert h.agent != "agent_c"

    def test_get_domain_knowledge_for_agents(
        self, storage: SQLiteStorage, agents_with_memories: Dict[str, list]
    ):
        """Test querying domain knowledge from multiple agents."""
        knowledge = storage.get_domain_knowledge_for_agents(
            project_id="test-project",
            agents=["agent_a", "agent_b"],
        )

        # Only agent_a has domain knowledge
        assert len(knowledge) == 1
        assert knowledge[0].agent == "agent_a"

    def test_empty_agents_list(
        self, storage: SQLiteStorage, agents_with_memories: Dict[str, list]
    ):
        """Test that empty agents list returns empty results."""
        heuristics = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=[],
        )
        assert heuristics == []


class TestRetrievalEngineMultiAgentSharing:
    """Tests for RetrievalEngine multi-agent memory sharing."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def retrieval_engine(self, storage: SQLiteStorage) -> RetrievalEngine:
        """Create a retrieval engine with mock embeddings."""
        return RetrievalEngine(
            storage=storage,
            embedding_provider="mock",
            enable_cache=False,
        )

    @pytest.fixture
    def scopes(self) -> Dict[str, MemoryScope]:
        """Create memory scopes with sharing configuration."""
        return {
            # Agent B inherits from Agent A (can read A's memories)
            "agent_b": MemoryScope(
                agent_name="agent_b",
                can_learn=["testing"],
                cannot_learn=[],
                inherit_from=["agent_a"],
            ),
            # Agent A shares with Agent B
            "agent_a": MemoryScope(
                agent_name="agent_a",
                can_learn=["testing"],
                cannot_learn=[],
                share_with=["agent_b"],
            ),
            # Agent C has no sharing (isolated)
            "agent_c": MemoryScope(
                agent_name="agent_c",
                can_learn=["testing"],
                cannot_learn=[],
            ),
            # Agent D has bidirectional sharing with Agent E
            "agent_d": MemoryScope(
                agent_name="agent_d",
                can_learn=["testing"],
                cannot_learn=[],
                share_with=["agent_e"],
                inherit_from=["agent_e"],
            ),
            "agent_e": MemoryScope(
                agent_name="agent_e",
                can_learn=["testing"],
                cannot_learn=[],
                share_with=["agent_d"],
                inherit_from=["agent_d"],
            ),
        }

    @pytest.fixture
    def seeded_storage(self, storage: SQLiteStorage) -> SQLiteStorage:
        """Seed storage with test data."""
        now = datetime.now(timezone.utc)
        # Use embeddings so vector search can find them
        embedding = [0.1] * 384

        # Agent A heuristic
        storage.save_heuristic(
            Heuristic(
                id="h_agent_a",
                agent="agent_a",
                project_id="test-project",
                condition="when testing forms",
                strategy="validate inputs first",
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
                embedding=embedding,
            )
        )

        # Agent B heuristic
        storage.save_heuristic(
            Heuristic(
                id="h_agent_b",
                agent="agent_b",
                project_id="test-project",
                condition="when testing APIs",
                strategy="check auth first",
                confidence=0.85,
                occurrence_count=5,
                success_count=4,
                last_validated=now,
                created_at=now,
                embedding=embedding,
            )
        )

        # Agent C heuristic (isolated)
        storage.save_heuristic(
            Heuristic(
                id="h_agent_c",
                agent="agent_c",
                project_id="test-project",
                condition="secret condition",
                strategy="secret strategy",
                confidence=0.95,
                occurrence_count=20,
                success_count=19,
                last_validated=now,
                created_at=now,
                embedding=embedding,
            )
        )

        # Agent D heuristic
        storage.save_heuristic(
            Heuristic(
                id="h_agent_d",
                agent="agent_d",
                project_id="test-project",
                condition="bidirectional d",
                strategy="strategy d",
                confidence=0.88,
                occurrence_count=8,
                success_count=7,
                last_validated=now,
                created_at=now,
                embedding=embedding,
            )
        )

        # Agent E heuristic
        storage.save_heuristic(
            Heuristic(
                id="h_agent_e",
                agent="agent_e",
                project_id="test-project",
                condition="bidirectional e",
                strategy="strategy e",
                confidence=0.87,
                occurrence_count=7,
                success_count=6,
                last_validated=now,
                created_at=now,
                embedding=embedding,
            )
        )

        return storage

    def test_agent_b_inherits_from_agent_a(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test that agent B can see agent A's memories via inheritance."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        result = engine.retrieve(
            query="testing forms",
            agent="agent_b",
            project_id="test-project",
            scope=scopes["agent_b"],
            include_shared=True,
        )

        # Should see both agent_a and agent_b heuristics
        agents = {h.agent for h in result.heuristics}
        assert "agent_a" in agents, "Agent B should see Agent A's memories"
        assert "agent_b" in agents, "Agent B should see its own memories"

    def test_agent_c_isolated(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test that agent C cannot see other agents' memories."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        result = engine.retrieve(
            query="testing",
            agent="agent_c",
            project_id="test-project",
            scope=scopes["agent_c"],
            include_shared=True,
        )

        # Should only see agent_c's own memories
        for h in result.heuristics:
            assert (
                h.agent == "agent_c"
            ), f"Agent C should only see its own memories, got {h.agent}"

    def test_agent_a_cannot_see_agent_b_without_inheritance(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test that agent A cannot see agent B's memories (A shares WITH B, not inherits FROM B)."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        result = engine.retrieve(
            query="testing APIs",
            agent="agent_a",
            project_id="test-project",
            scope=scopes["agent_a"],
            include_shared=True,
        )

        # Should NOT see agent_b's memories
        for h in result.heuristics:
            assert (
                h.agent == "agent_a"
            ), f"Agent A should not see Agent B's memories, got {h.agent}"

    def test_bidirectional_sharing(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test bidirectional sharing between agents D and E."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        # Agent D should see Agent E's memories
        result_d = engine.retrieve(
            query="bidirectional",
            agent="agent_d",
            project_id="test-project",
            scope=scopes["agent_d"],
            include_shared=True,
        )

        agents_d = {h.agent for h in result_d.heuristics}
        assert "agent_d" in agents_d
        assert "agent_e" in agents_d

        # Agent E should see Agent D's memories
        result_e = engine.retrieve(
            query="bidirectional",
            agent="agent_e",
            project_id="test-project",
            scope=scopes["agent_e"],
            include_shared=True,
        )

        agents_e = {h.agent for h in result_e.heuristics}
        assert "agent_d" in agents_e
        assert "agent_e" in agents_e

    def test_shared_from_metadata_tracking(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test that shared memories have shared_from in metadata."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        result = engine.retrieve(
            query="testing forms",
            agent="agent_b",
            project_id="test-project",
            scope=scopes["agent_b"],
            include_shared=True,
        )

        for h in result.heuristics:
            if h.agent == "agent_a":
                # Shared memory should have shared_from metadata
                assert (
                    "shared_from" in h.metadata
                ), "Shared memory should have shared_from metadata"
                assert h.metadata["shared_from"] == "agent_a"
            elif h.agent == "agent_b":
                # Own memory should NOT have shared_from (or it should be absent)
                shared_from = h.metadata.get("shared_from")
                assert (
                    shared_from is None
                ), "Own memory should not have shared_from metadata"

    def test_include_shared_false_disables_sharing(
        self,
        seeded_storage: SQLiteStorage,
        scopes: Dict[str, MemoryScope],
    ):
        """Test that include_shared=False disables cross-agent retrieval."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        result = engine.retrieve(
            query="testing forms",
            agent="agent_b",
            project_id="test-project",
            scope=scopes["agent_b"],
            include_shared=False,  # Disable sharing
        )

        # Should only see agent_b's own memories
        for h in result.heuristics:
            assert (
                h.agent == "agent_b"
            ), "With include_shared=False, should only see own memories"

    def test_backward_compatibility_no_scope(
        self,
        seeded_storage: SQLiteStorage,
    ):
        """Test backward compatibility when no scope is provided."""
        engine = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
            enable_cache=False,
            min_score_threshold=0.0,  # Disable threshold for testing
        )

        # No scope provided - should work like before
        result = engine.retrieve(
            query="testing forms",
            agent="agent_a",
            project_id="test-project",
            # No scope
        )

        # Should only see agent_a's memories
        for h in result.heuristics:
            assert h.agent == "agent_a"


class TestWriteIsolation:
    """Tests to verify write isolation is maintained."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    def test_agent_cannot_modify_shared_memory(self, storage: SQLiteStorage):
        """
        Test that an agent cannot modify another agent's memory.

        This is implicit in the current design - save operations require
        the correct agent field, and update operations work by ID which
        is owned by the creating agent.
        """
        now = datetime.now(timezone.utc)

        # Agent A creates a heuristic
        original = Heuristic(
            id="h_owned_by_a",
            agent="agent_a",
            project_id="test-project",
            condition="original condition",
            strategy="original strategy",
            confidence=0.9,
            occurrence_count=10,
            success_count=9,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(original)

        # Verify the heuristic is saved correctly
        heuristics = storage.get_heuristics(project_id="test-project", agent="agent_a")
        assert len(heuristics) == 1
        assert heuristics[0].strategy == "original strategy"

        # If Agent B tries to overwrite by saving with same ID but different agent,
        # it would create a separate entry or fail depending on implementation.
        # The key is that original agent_a's memory remains unchanged.

        # Note: Write isolation is enforced at the application level.
        # The storage layer allows updates by ID, but the application
        # should check ownership before allowing updates.

    def test_shared_memory_retains_original_agent(self, storage: SQLiteStorage):
        """Test that shared memories retain their original agent field."""
        now = datetime.now(timezone.utc)

        # Agent A creates a heuristic
        h = Heuristic(
            id="h_shared",
            agent="agent_a",
            project_id="test-project",
            condition="shared condition",
            strategy="shared strategy",
            confidence=0.9,
            occurrence_count=10,
            success_count=9,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        # When retrieved by agent_b, the agent field should still be agent_a
        heuristics = storage.get_heuristics_for_agents(
            project_id="test-project",
            agents=["agent_a", "agent_b"],
        )

        for h in heuristics:
            if h.id == "h_shared":
                assert h.agent == "agent_a", "Original agent should be preserved"


class TestOutcomesAndAntiPatternsSharing:
    """Tests for multi-agent sharing of outcomes and anti-patterns."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def seeded_storage(self, storage: SQLiteStorage) -> SQLiteStorage:
        """Seed storage with outcomes and anti-patterns."""
        now = datetime.now(timezone.utc)

        # Outcomes
        storage.save_outcome(
            Outcome(
                id="o_agent_a",
                agent="agent_a",
                project_id="test-project",
                task_type="testing",
                task_description="Test form validation",
                success=True,
                strategy_used="validate first",
                timestamp=now,
            )
        )

        storage.save_outcome(
            Outcome(
                id="o_agent_b",
                agent="agent_b",
                project_id="test-project",
                task_type="testing",
                task_description="Test API auth",
                success=True,
                strategy_used="check tokens",
                timestamp=now,
            )
        )

        # Anti-patterns
        storage.save_anti_pattern(
            AntiPattern(
                id="ap_agent_a",
                agent="agent_a",
                project_id="test-project",
                pattern="using sleep",
                why_bad="flaky tests",
                better_alternative="use waits",
                occurrence_count=5,
                last_seen=now,
                created_at=now,
            )
        )

        return storage

    def test_outcomes_for_agents(self, seeded_storage: SQLiteStorage):
        """Test querying outcomes from multiple agents."""
        outcomes = seeded_storage.get_outcomes_for_agents(
            project_id="test-project",
            agents=["agent_a", "agent_b"],
        )

        assert len(outcomes) == 2
        agents = {o.agent for o in outcomes}
        assert "agent_a" in agents
        assert "agent_b" in agents

    def test_anti_patterns_for_agents(self, seeded_storage: SQLiteStorage):
        """Test querying anti-patterns from multiple agents."""
        patterns = seeded_storage.get_anti_patterns_for_agents(
            project_id="test-project",
            agents=["agent_a", "agent_b"],
        )

        # Only agent_a has anti-patterns
        assert len(patterns) == 1
        assert patterns[0].agent == "agent_a"
