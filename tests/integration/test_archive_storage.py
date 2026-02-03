"""
Integration tests for Memory Archive storage operations.

Tests archive_memory, list_archives, restore_from_archive, and purge_archives
with the SQLite storage backend.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alma.storage.archive import ArchiveReason
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import AntiPattern, DomainKnowledge, Heuristic, Outcome


class TestArchiveStorageIntegration:
    """Integration tests for archive storage operations."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_archive.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    @pytest.fixture
    def sample_heuristic(self, storage):
        """Create a sample heuristic for archiving."""
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-archive-test-1",
            agent="test-agent",
            project_id="test-project",
            condition="when testing archive",
            strategy="use sample data",
            confidence=0.8,
            occurrence_count=5,
            success_count=4,
            last_validated=now,
            created_at=now - timedelta(days=30),
        )
        storage.save_heuristic(h)
        return h

    @pytest.fixture
    def sample_outcome(self, storage):
        """Create a sample outcome for archiving."""
        now = datetime.now(timezone.utc)
        o = Outcome(
            id="o-archive-test-1",
            agent="test-agent",
            project_id="test-project",
            task_type="testing",
            task_description="Test archive functionality",
            success=True,
            strategy_used="manual testing",
            duration_ms=1000,
            timestamp=now - timedelta(days=15),
        )
        storage.save_outcome(o)
        return o

    def test_archive_heuristic(self, storage, sample_heuristic):
        """Should archive a heuristic with all data preserved."""
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason=ArchiveReason.DECAY.value,
            final_strength=0.05,
        )

        assert archived.id.startswith("archive-")
        assert archived.original_id == sample_heuristic.id
        assert archived.memory_type == "heuristic"
        assert archived.archive_reason == "decay"
        assert archived.final_strength == 0.05
        assert archived.project_id == "test-project"
        assert archived.agent == "test-agent"
        assert archived.restored is False

        # Verify content preserved
        import json

        content = json.loads(archived.content)
        assert content["condition"] == "when testing archive"
        assert content["strategy"] == "use sample data"

    def test_archive_outcome(self, storage, sample_outcome):
        """Should archive an outcome with all data preserved."""
        archived = storage.archive_memory(
            memory_id=sample_outcome.id,
            memory_type="outcome",
            reason=ArchiveReason.MANUAL.value,
            final_strength=0.1,
        )

        assert archived.original_id == sample_outcome.id
        assert archived.memory_type == "outcome"
        assert archived.archive_reason == "manual"

        import json

        content = json.loads(archived.content)
        assert content["task_type"] == "testing"
        assert content["success"] is True

    def test_archive_preserves_embedding(self, storage):
        """Should preserve embedding vector when archiving."""
        import numpy as np

        # Create heuristic with embedding
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-emb-test",
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

        # Add embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (memory_type, memory_id, embedding) VALUES (?, ?, ?)",
                ("heuristic", "h-emb-test", embedding_blob),
            )

        # Archive
        archived = storage.archive_memory(
            memory_id="h-emb-test",
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )

        assert archived.embedding is not None
        assert len(archived.embedding) == 5
        # Verify values (with floating point tolerance)
        for i, val in enumerate(embedding):
            assert abs(archived.embedding[i] - val) < 0.001

    def test_get_archive(self, storage, sample_heuristic):
        """Should retrieve archive by ID."""
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )

        loaded = storage.get_archive(archived.id)

        assert loaded is not None
        assert loaded.id == archived.id
        assert loaded.original_id == sample_heuristic.id
        assert loaded.memory_type == "heuristic"

    def test_get_nonexistent_archive(self, storage):
        """Should return None for nonexistent archive."""
        loaded = storage.get_archive("nonexistent-archive-id")
        assert loaded is None

    def test_list_archives_by_project(self, storage, sample_heuristic, sample_outcome):
        """Should list archives filtered by project."""
        # Archive both memories
        storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.archive_memory(
            memory_id=sample_outcome.id,
            memory_type="outcome",
            reason="manual",
            final_strength=0.1,
        )

        archives = storage.list_archives(project_id="test-project")

        assert len(archives) == 2

    def test_list_archives_by_agent(self, storage):
        """Should filter archives by agent."""
        now = datetime.now(timezone.utc)

        # Create memories for different agents
        h1 = Heuristic(
            id="h-agent-a",
            agent="agent-a",
            project_id="test-project",
            condition="test",
            strategy="test",
            confidence=0.5,
            occurrence_count=1,
            success_count=1,
            last_validated=now,
            created_at=now,
        )
        h2 = Heuristic(
            id="h-agent-b",
            agent="agent-b",
            project_id="test-project",
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

        storage.archive_memory("h-agent-a", "heuristic", "decay", 0.05)
        storage.archive_memory("h-agent-b", "heuristic", "decay", 0.05)

        archives = storage.list_archives(project_id="test-project", agent="agent-a")

        assert len(archives) == 1
        assert archives[0].agent == "agent-a"

    def test_list_archives_by_reason(self, storage, sample_heuristic, sample_outcome):
        """Should filter archives by reason."""
        storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.archive_memory(
            memory_id=sample_outcome.id,
            memory_type="outcome",
            reason="manual",
            final_strength=0.1,
        )

        archives = storage.list_archives(project_id="test-project", reason="decay")

        assert len(archives) == 1
        assert archives[0].archive_reason == "decay"

    def test_list_archives_by_memory_type(
        self, storage, sample_heuristic, sample_outcome
    ):
        """Should filter archives by memory type."""
        storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.archive_memory(
            memory_id=sample_outcome.id,
            memory_type="outcome",
            reason="decay",
            final_strength=0.05,
        )

        archives = storage.list_archives(
            project_id="test-project",
            memory_type="outcome",
        )

        assert len(archives) == 1
        assert archives[0].memory_type == "outcome"

    def test_list_archives_exclude_restored(self, storage, sample_heuristic):
        """Should exclude restored archives by default."""
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )

        # Restore the archive
        storage.restore_from_archive(archived.id)

        # Should not include restored
        archives = storage.list_archives(project_id="test-project")
        assert len(archives) == 0

        # Should include when requested
        archives_with_restored = storage.list_archives(
            project_id="test-project",
            include_restored=True,
        )
        assert len(archives_with_restored) == 1

    def test_list_archives_limit(self, storage):
        """Should respect limit parameter."""
        now = datetime.now(timezone.utc)

        # Create and archive 10 memories
        for i in range(10):
            h = Heuristic(
                id=f"h-limit-{i}",
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
            storage.archive_memory(f"h-limit-{i}", "heuristic", "decay", 0.05)

        archives = storage.list_archives(project_id="test-project", limit=5)

        assert len(archives) == 5

    def test_restore_from_archive(self, storage, sample_heuristic):
        """Should restore archived memory."""
        # Archive
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )

        # Delete original (simulating what would happen after archive)
        storage.delete_heuristic(sample_heuristic.id)

        # Restore
        new_id = storage.restore_from_archive(archived.id)

        assert new_id.startswith("heu-")

        # Verify new memory exists
        heuristics = storage.get_heuristics(project_id="test-project")
        assert len(heuristics) == 1
        restored = heuristics[0]
        assert restored.id == new_id
        assert restored.condition == sample_heuristic.condition
        assert restored.strategy == sample_heuristic.strategy

    def test_restore_marks_archive_as_restored(self, storage, sample_heuristic):
        """Should mark archive as restored."""
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.delete_heuristic(sample_heuristic.id)

        new_id = storage.restore_from_archive(archived.id)

        # Reload archive
        reloaded = storage.get_archive(archived.id)
        assert reloaded.restored is True
        assert reloaded.restored_at is not None
        assert reloaded.restored_as == new_id

    def test_restore_nonexistent_archive_raises(self, storage):
        """Should raise ValueError for nonexistent archive."""
        with pytest.raises(ValueError, match="Archive not found"):
            storage.restore_from_archive("nonexistent-archive")

    def test_restore_already_restored_raises(self, storage, sample_heuristic):
        """Should raise ValueError if archive already restored."""
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.delete_heuristic(sample_heuristic.id)
        storage.restore_from_archive(archived.id)

        with pytest.raises(ValueError, match="already restored"):
            storage.restore_from_archive(archived.id)

    def test_restore_preserves_embedding(self, storage):
        """Should restore embedding with memory."""
        import numpy as np

        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h-emb-restore",
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

        # Add embedding
        embedding = [0.1, 0.2, 0.3]
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (memory_type, memory_id, embedding) VALUES (?, ?, ?)",
                ("heuristic", "h-emb-restore", embedding_blob),
            )

        # Archive and restore
        archived = storage.archive_memory("h-emb-restore", "heuristic", "decay", 0.05)
        storage.delete_heuristic("h-emb-restore")
        new_id = storage.restore_from_archive(archived.id)

        # Check embedding restored
        with storage._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embeddings WHERE memory_id = ?",
                (new_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            restored_emb = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
            assert len(restored_emb) == 3

    def test_purge_archives(self, storage, sample_heuristic):
        """Should permanently delete old archives."""
        # Archive with a known timestamp
        archived = storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )

        # Purge archives older than 1 hour from now (should not delete recent)
        count = storage.purge_archives(
            older_than=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        assert count == 0

        # Archive should still exist
        assert storage.get_archive(archived.id) is not None

        # Purge archives older than 1 hour in the future (should delete all)
        count = storage.purge_archives(
            older_than=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        assert count == 1

        # Archive should be gone
        assert storage.get_archive(archived.id) is None

    def test_purge_archives_by_reason(self, storage, sample_heuristic, sample_outcome):
        """Should purge only archives with specified reason."""
        storage.archive_memory(
            memory_id=sample_heuristic.id,
            memory_type="heuristic",
            reason="decay",
            final_strength=0.05,
        )
        storage.archive_memory(
            memory_id=sample_outcome.id,
            memory_type="outcome",
            reason="manual",
            final_strength=0.1,
        )

        # Purge only decay archives
        count = storage.purge_archives(
            older_than=datetime.now(timezone.utc) + timedelta(hours=1),
            reason="decay",
        )
        assert count == 1

        # Manual archive should still exist
        archives = storage.list_archives(project_id="test-project")
        assert len(archives) == 1
        assert archives[0].archive_reason == "manual"

    def test_get_archive_stats(self, storage):
        """Should return correct archive statistics."""
        now = datetime.now(timezone.utc)

        # Create and archive some memories
        for i in range(3):
            h = Heuristic(
                id=f"h-stats-{i}",
                agent="agent-a" if i < 2 else "agent-b",
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
            storage.archive_memory(
                f"h-stats-{i}",
                "heuristic",
                "decay" if i < 2 else "manual",
                0.05,
            )

        stats = storage.get_archive_stats(project_id="test-project")

        assert stats["total_count"] == 3
        assert stats["by_reason"]["decay"] == 2
        assert stats["by_reason"]["manual"] == 1
        assert stats["by_type"]["heuristic"] == 3
        assert stats["by_agent"]["agent-a"] == 2
        assert stats["by_agent"]["agent-b"] == 1
        assert stats["restored_count"] == 0
        assert stats["oldest_archive"] is not None
        assert stats["newest_archive"] is not None


class TestArchiveWithDomainKnowledge:
    """Test archiving domain knowledge memories."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dk.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_archive_domain_knowledge(self, storage):
        """Should archive domain knowledge correctly."""
        now = datetime.now(timezone.utc)
        dk = DomainKnowledge(
            id="dk-test-1",
            agent="test-agent",
            project_id="test-project",
            domain="testing",
            fact="Archive system works",
            source="integration test",
            confidence=0.9,
            last_verified=now,
        )
        storage.save_domain_knowledge(dk)

        archived = storage.archive_memory(
            memory_id="dk-test-1",
            memory_type="domain_knowledge",
            reason="consolidation",
            final_strength=0.15,
        )

        assert archived.memory_type == "domain_knowledge"
        assert archived.archive_reason == "consolidation"

        import json

        content = json.loads(archived.content)
        assert content["domain"] == "testing"
        assert content["fact"] == "Archive system works"

    def test_restore_domain_knowledge(self, storage):
        """Should restore domain knowledge correctly."""
        now = datetime.now(timezone.utc)
        dk = DomainKnowledge(
            id="dk-restore-1",
            agent="test-agent",
            project_id="test-project",
            domain="testing",
            fact="Restore test fact",
            source="test",
            confidence=0.8,
            last_verified=now,
        )
        storage.save_domain_knowledge(dk)

        archived = storage.archive_memory(
            "dk-restore-1",
            "domain_knowledge",
            "decay",
            0.05,
        )
        storage.delete_domain_knowledge("dk-restore-1")

        new_id = storage.restore_from_archive(archived.id)

        knowledge = storage.get_domain_knowledge(project_id="test-project")
        assert len(knowledge) == 1
        assert knowledge[0].id == new_id
        assert knowledge[0].fact == "Restore test fact"


class TestArchiveWithAntiPattern:
    """Test archiving anti-pattern memories."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_ap.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_archive_anti_pattern(self, storage):
        """Should archive anti-pattern correctly."""
        now = datetime.now(timezone.utc)
        ap = AntiPattern(
            id="ap-test-1",
            agent="test-agent",
            project_id="test-project",
            pattern="Testing without archiving",
            why_bad="Data loss risk",
            better_alternative="Use archive system",
            occurrence_count=3,
            last_seen=now,
            created_at=now - timedelta(days=10),
        )
        storage.save_anti_pattern(ap)

        archived = storage.archive_memory(
            memory_id="ap-test-1",
            memory_type="anti_pattern",
            reason="superseded",
            final_strength=0.08,
        )

        assert archived.memory_type == "anti_pattern"
        assert archived.archive_reason == "superseded"

        import json

        content = json.loads(archived.content)
        assert content["pattern"] == "Testing without archiving"
        assert content["why_bad"] == "Data loss risk"

    def test_restore_anti_pattern(self, storage):
        """Should restore anti-pattern correctly."""
        now = datetime.now(timezone.utc)
        ap = AntiPattern(
            id="ap-restore-1",
            agent="test-agent",
            project_id="test-project",
            pattern="Bad pattern to restore",
            why_bad="Because test",
            better_alternative="Good pattern",
            occurrence_count=1,
            last_seen=now,
            created_at=now,
        )
        storage.save_anti_pattern(ap)

        archived = storage.archive_memory(
            "ap-restore-1",
            "anti_pattern",
            "decay",
            0.05,
        )
        storage.delete_anti_pattern("ap-restore-1")

        new_id = storage.restore_from_archive(archived.id)

        patterns = storage.get_anti_patterns(project_id="test-project")
        assert len(patterns) == 1
        assert patterns[0].id == new_id
        assert patterns[0].pattern == "Bad pattern to restore"
