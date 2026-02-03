"""
Integration tests for decay-based forgetting with storage backends.

Tests the MemoryStrength persistence with SQLite storage.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alma.learning.decay import DecayConfig, DecayManager, MemoryStrength
from alma.storage.sqlite_local import SQLiteStorage


class TestDecayStorageIntegration:
    """Integration tests for decay storage operations."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_decay.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_save_and_get_memory_strength(self, storage):
        """Should save and retrieve memory strength."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-memory-1",
            memory_type="heuristic",
            initial_strength=1.0,
            decay_half_life_days=60,
            created_at=now,
            last_accessed=now,
            access_count=5,
            reinforcement_events=[now - timedelta(days=1)],
            explicit_importance=0.8,
        )

        # Save
        saved_id = storage.save_memory_strength(strength)
        assert saved_id == "test-memory-1"

        # Retrieve
        loaded = storage.get_memory_strength("test-memory-1")
        assert loaded is not None
        assert loaded.memory_id == "test-memory-1"
        assert loaded.memory_type == "heuristic"
        assert loaded.decay_half_life_days == 60
        assert loaded.access_count == 5
        assert loaded.explicit_importance == 0.8
        assert len(loaded.reinforcement_events) == 1

    def test_update_memory_strength(self, storage):
        """Should update existing memory strength."""
        strength = MemoryStrength(
            memory_id="test-memory-2",
            memory_type="outcome",
            access_count=0,
        )

        # Save initial
        storage.save_memory_strength(strength)

        # Update
        strength.access_count = 10
        strength.explicit_importance = 0.9
        storage.save_memory_strength(strength)

        # Verify update
        loaded = storage.get_memory_strength("test-memory-2")
        assert loaded.access_count == 10
        assert loaded.explicit_importance == 0.9

    def test_get_nonexistent_strength(self, storage):
        """Should return None for nonexistent memory."""
        loaded = storage.get_memory_strength("nonexistent-id")
        assert loaded is None

    def test_delete_memory_strength(self, storage):
        """Should delete memory strength."""
        strength = MemoryStrength(memory_id="test-delete")
        storage.save_memory_strength(strength)

        # Verify exists
        assert storage.get_memory_strength("test-delete") is not None

        # Delete
        deleted = storage.delete_memory_strength("test-delete")
        assert deleted is True

        # Verify deleted
        assert storage.get_memory_strength("test-delete") is None

    def test_delete_nonexistent_returns_false(self, storage):
        """Should return False when deleting nonexistent memory."""
        deleted = storage.delete_memory_strength("nonexistent-id")
        assert deleted is False

    def test_get_all_memory_strengths_with_project(self, storage):
        """Should filter by project_id."""
        # First, create some heuristics to associate memories with
        from alma.types import Heuristic

        now = datetime.now(timezone.utc)

        # Create heuristics in different projects
        h1 = Heuristic(
            id="h1",
            agent="agent-a",
            project_id="project-1",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        h2 = Heuristic(
            id="h2",
            agent="agent-a",
            project_id="project-2",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        # Create strength records
        strength1 = MemoryStrength(memory_id="h1", memory_type="heuristic")
        strength2 = MemoryStrength(memory_id="h2", memory_type="heuristic")

        storage.save_memory_strength(strength1)
        storage.save_memory_strength(strength2)

        # Query by project
        project1_strengths = storage.get_all_memory_strengths("project-1")
        assert len(project1_strengths) == 1
        assert project1_strengths[0].memory_id == "h1"

    def test_get_all_memory_strengths_with_agent(self, storage):
        """Should filter by agent."""
        from alma.types import Heuristic

        now = datetime.now(timezone.utc)

        # Create heuristics with different agents
        h1 = Heuristic(
            id="h-a1",
            agent="agent-a",
            project_id="project-x",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        h2 = Heuristic(
            id="h-b1",
            agent="agent-b",
            project_id="project-x",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        # Create strength records
        storage.save_memory_strength(
            MemoryStrength(memory_id="h-a1", memory_type="heuristic")
        )
        storage.save_memory_strength(
            MemoryStrength(memory_id="h-b1", memory_type="heuristic")
        )

        # Query by agent
        agent_a_strengths = storage.get_all_memory_strengths(
            "project-x", agent="agent-a"
        )
        assert len(agent_a_strengths) == 1
        assert agent_a_strengths[0].memory_id == "h-a1"


class TestDecayManagerWithStorage:
    """Integration tests for DecayManager with real storage."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_decay_manager.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    @pytest.fixture
    def manager(self, storage):
        """Create DecayManager with storage."""
        config = DecayConfig(
            default_half_life_days=30,
            forget_threshold=0.1,
        )
        return DecayManager(storage=storage, config=config)

    def test_record_access_persists(self, manager, storage):
        """Access should persist to storage."""
        # Record access
        manager.record_access("mem-1", "heuristic")

        # Clear cache
        manager.invalidate_cache()

        # Get fresh from storage
        strength = manager.get_strength("mem-1")
        assert strength.access_count == 1

    def test_reinforce_persists(self, manager, storage):
        """Reinforcement should persist to storage."""
        # Reinforce
        manager.reinforce_memory("mem-2", "outcome")

        # Clear cache
        manager.invalidate_cache()

        # Get fresh from storage
        strength = manager.get_strength("mem-2")
        assert len(strength.reinforcement_events) == 1

    def test_set_importance_persists(self, manager, storage):
        """Importance setting should persist to storage."""
        # Set importance
        manager.set_importance("mem-3", 0.95, "knowledge")

        # Clear cache
        manager.invalidate_cache()

        # Get fresh from storage
        strength = manager.get_strength("mem-3")
        assert strength.explicit_importance == 0.95

    def test_multiple_accesses_accumulate(self, manager, storage):
        """Multiple accesses should accumulate."""
        for _ in range(5):
            manager.record_access("mem-4", "heuristic")

        # Clear cache
        manager.invalidate_cache()

        strength = manager.get_strength("mem-4")
        assert strength.access_count == 5

    def test_get_forgettable_with_real_data(self, manager, storage):
        """Should identify forgettable memories from storage."""
        from alma.types import Heuristic

        now = datetime.now(timezone.utc)

        # Create a heuristic
        h = Heuristic(
            id="old-heuristic",
            agent="test-agent",
            project_id="test-project",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        # Create a very old, low importance strength record
        old_strength = MemoryStrength(
            memory_id="old-heuristic",
            memory_type="heuristic",
            decay_half_life_days=30,
            created_at=now - timedelta(days=200),
            last_accessed=now - timedelta(days=200),
            explicit_importance=0.0,
        )
        storage.save_memory_strength(old_strength)

        # Should be forgettable
        forgettable = manager.get_forgettable_memories("test-project")
        forgettable_ids = [f[0] for f in forgettable]
        assert "old-heuristic" in forgettable_ids

    def test_get_memory_stats_with_real_data(self, manager, storage):
        """Should return correct stats from storage."""
        from alma.types import Heuristic

        now = datetime.now(timezone.utc)

        # Create heuristics
        for i in range(3):
            h = Heuristic(
                id=f"stats-h-{i}",
                agent="stats-agent",
                project_id="stats-project",
                condition="test",
                strategy="test",
                confidence=0.5,
                occurrence_count=1,
                success_count=1,
                last_validated=now,
                created_at=now,
            )
            storage.save_heuristic(h)
            storage.save_memory_strength(
                MemoryStrength(memory_id=f"stats-h-{i}", memory_type="heuristic")
            )

        stats = manager.get_memory_stats("stats-project")
        assert stats["total"] == 3
        assert "heuristic" in stats["by_type"]
        assert stats["by_type"]["heuristic"]["count"] == 3


class TestDecayReinforcementCycle:
    """Test the full reinforcement cycle."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cycle.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_decay_and_recover_cycle(self, storage):
        """Memory should decay over time but recover with reinforcement."""
        from alma.types import Heuristic

        now = datetime.now(timezone.utc)

        # Create a heuristic
        h = Heuristic(
            id="cycle-test",
            agent="test-agent",
            project_id="test-project",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        storage.save_heuristic(h)

        # Create aged strength record (simulate 60 days old)
        old_strength = MemoryStrength(
            memory_id="cycle-test",
            memory_type="heuristic",
            decay_half_life_days=30,
            created_at=now - timedelta(days=60),
            last_accessed=now - timedelta(days=60),
            access_count=0,
            explicit_importance=0.5,
        )
        storage.save_memory_strength(old_strength)

        # Check initial (decayed) strength
        loaded = storage.get_memory_strength("cycle-test")
        initial = loaded.current_strength()
        assert initial < 0.5  # Should be decayed

        # Reinforce
        loaded.reinforce()
        storage.save_memory_strength(loaded)

        # Check recovered strength
        reloaded = storage.get_memory_strength("cycle-test")
        recovered = reloaded.current_strength()
        assert recovered > initial  # Should be stronger after reinforcement
