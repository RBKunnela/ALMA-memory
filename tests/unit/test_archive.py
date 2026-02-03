"""
Unit tests for the Memory Archive System.

Tests ArchivedMemory, ArchiveConfig, ArchiveStats, and ArchiveReason classes.
"""

from datetime import datetime, timedelta, timezone

from alma.storage.archive import (
    ArchiveConfig,
    ArchivedMemory,
    ArchiveReason,
    ArchiveStats,
)


class TestArchiveReason:
    """Tests for ArchiveReason enum."""

    def test_decay_reason(self):
        """Should have decay reason."""
        assert ArchiveReason.DECAY.value == "decay"

    def test_manual_reason(self):
        """Should have manual reason."""
        assert ArchiveReason.MANUAL.value == "manual"

    def test_consolidation_reason(self):
        """Should have consolidation reason."""
        assert ArchiveReason.CONSOLIDATION.value == "consolidation"

    def test_superseded_reason(self):
        """Should have superseded reason."""
        assert ArchiveReason.SUPERSEDED.value == "superseded"

    def test_quota_reason(self):
        """Should have quota reason."""
        assert ArchiveReason.QUOTA.value == "quota"

    def test_cleanup_reason(self):
        """Should have cleanup reason."""
        assert ArchiveReason.CLEANUP.value == "cleanup"


class TestArchivedMemory:
    """Tests for ArchivedMemory dataclass."""

    def test_create_archived_memory(self):
        """Should create archived memory with all required fields."""
        archived = ArchivedMemory.create(
            original_id="mem-123",
            memory_type="heuristic",
            content='{"condition": "test", "strategy": "test"}',
            project_id="proj-1",
            agent="test-agent",
            archive_reason="decay",
            final_strength=0.05,
        )

        assert archived.id.startswith("archive-")
        assert archived.original_id == "mem-123"
        assert archived.memory_type == "heuristic"
        assert archived.project_id == "proj-1"
        assert archived.agent == "test-agent"
        assert archived.archive_reason == "decay"
        assert archived.final_strength == 0.05
        assert archived.restored is False
        assert archived.restored_at is None
        assert archived.restored_as is None

    def test_create_with_embedding(self):
        """Should create archived memory with embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        archived = ArchivedMemory.create(
            original_id="mem-456",
            memory_type="outcome",
            content='{"task_type": "test"}',
            project_id="proj-1",
            agent="agent-a",
            archive_reason="manual",
            final_strength=0.2,
            embedding=embedding,
        )

        assert archived.embedding == embedding

    def test_create_with_metadata(self):
        """Should create archived memory with metadata."""
        metadata = {"source": "test", "tags": ["a", "b"]}
        archived = ArchivedMemory.create(
            original_id="mem-789",
            memory_type="knowledge",
            content='{"fact": "test fact"}',
            project_id="proj-1",
            agent="agent-b",
            archive_reason="consolidation",
            final_strength=0.15,
            metadata=metadata,
        )

        assert archived.metadata == metadata

    def test_create_with_original_created_at(self):
        """Should preserve original creation timestamp."""
        original_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        archived = ArchivedMemory.create(
            original_id="mem-100",
            memory_type="heuristic",
            content="{}",
            project_id="proj-1",
            agent="agent",
            archive_reason="decay",
            final_strength=0.08,
            original_created_at=original_time,
        )

        assert archived.original_created_at == original_time

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        now = datetime.now(timezone.utc)
        archived = ArchivedMemory(
            id="archive-test123",
            original_id="mem-123",
            memory_type="heuristic",
            content='{"test": "content"}',
            embedding=[0.1, 0.2],
            metadata={"key": "value"},
            original_created_at=now - timedelta(days=30),
            archived_at=now,
            archive_reason="decay",
            final_strength=0.05,
            project_id="proj-1",
            agent="agent-a",
        )

        d = archived.to_dict()

        assert d["id"] == "archive-test123"
        assert d["original_id"] == "mem-123"
        assert d["memory_type"] == "heuristic"
        assert d["content"] == '{"test": "content"}'
        assert d["embedding"] == [0.1, 0.2]
        assert d["metadata"] == {"key": "value"}
        assert d["archive_reason"] == "decay"
        assert d["final_strength"] == 0.05
        assert d["project_id"] == "proj-1"
        assert d["agent"] == "agent-a"
        assert d["restored"] is False
        assert d["restored_at"] is None
        assert d["restored_as"] is None

    def test_from_dict(self):
        """Should create from dictionary correctly."""
        data = {
            "id": "archive-abc",
            "original_id": "mem-original",
            "memory_type": "outcome",
            "content": '{"task": "test"}',
            "embedding": [0.5, 0.5],
            "metadata": {"info": "test"},
            "original_created_at": "2024-01-01T00:00:00+00:00",
            "archived_at": "2024-06-01T00:00:00+00:00",
            "archive_reason": "manual",
            "final_strength": 0.1,
            "project_id": "test-project",
            "agent": "test-agent",
            "restored": False,
            "restored_at": None,
            "restored_as": None,
        }

        archived = ArchivedMemory.from_dict(data)

        assert archived.id == "archive-abc"
        assert archived.original_id == "mem-original"
        assert archived.memory_type == "outcome"
        assert archived.embedding == [0.5, 0.5]
        assert archived.metadata == {"info": "test"}
        assert archived.archive_reason == "manual"
        assert archived.final_strength == 0.1
        assert archived.restored is False

    def test_from_dict_with_z_suffix(self):
        """Should handle Z suffix in datetime strings."""
        data = {
            "id": "archive-xyz",
            "original_id": "mem-100",
            "memory_type": "heuristic",
            "content": "{}",
            "original_created_at": "2024-01-01T00:00:00Z",
            "archived_at": "2024-06-01T12:00:00Z",
            "archive_reason": "decay",
            "final_strength": 0.05,
            "project_id": "proj",
            "agent": "agent",
        }

        archived = ArchivedMemory.from_dict(data)

        assert archived.original_created_at.tzinfo == timezone.utc
        assert archived.archived_at.tzinfo == timezone.utc

    def test_from_dict_with_restored(self):
        """Should handle restored archives."""
        data = {
            "id": "archive-restored",
            "original_id": "mem-200",
            "memory_type": "knowledge",
            "content": "{}",
            "original_created_at": "2024-01-01T00:00:00+00:00",
            "archived_at": "2024-03-01T00:00:00+00:00",
            "archive_reason": "decay",
            "final_strength": 0.08,
            "project_id": "proj",
            "agent": "agent",
            "restored": True,
            "restored_at": "2024-04-01T00:00:00+00:00",
            "restored_as": "mem-300",
        }

        archived = ArchivedMemory.from_dict(data)

        assert archived.restored is True
        assert archived.restored_at is not None
        assert archived.restored_as == "mem-300"

    def test_mark_restored(self):
        """Should mark archive as restored."""
        archived = ArchivedMemory.create(
            original_id="mem-500",
            memory_type="heuristic",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="decay",
            final_strength=0.05,
        )

        assert archived.restored is False

        archived.mark_restored("new-mem-600")

        assert archived.restored is True
        assert archived.restored_at is not None
        assert archived.restored_as == "new-mem-600"


class TestArchiveConfig:
    """Tests for ArchiveConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = ArchiveConfig()

        assert config.enabled is True
        assert config.retention_days == 365
        assert config.auto_purge is False
        assert config.archive_on_decay is True
        assert config.archive_on_consolidation is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = ArchiveConfig(
            enabled=False,
            retention_days=90,
            auto_purge=True,
            archive_on_decay=False,
            archive_on_consolidation=False,
        )

        assert config.enabled is False
        assert config.retention_days == 90
        assert config.auto_purge is True
        assert config.archive_on_decay is False
        assert config.archive_on_consolidation is False

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "enabled": True,
            "retention_days": 180,
            "auto_purge": True,
        }

        config = ArchiveConfig.from_dict(data)

        assert config.enabled is True
        assert config.retention_days == 180
        assert config.auto_purge is True
        # Defaults for unspecified
        assert config.archive_on_decay is True
        assert config.archive_on_consolidation is True

    def test_from_dict_empty(self):
        """Should use defaults for empty dict."""
        config = ArchiveConfig.from_dict({})

        assert config.enabled is True
        assert config.retention_days == 365


class TestArchiveStats:
    """Tests for ArchiveStats dataclass."""

    def test_default_stats(self):
        """Should have zero defaults."""
        stats = ArchiveStats()

        assert stats.total_count == 0
        assert stats.by_reason == {}
        assert stats.by_type == {}
        assert stats.by_agent == {}
        assert stats.restored_count == 0
        assert stats.oldest_archive is None
        assert stats.newest_archive is None

    def test_stats_with_data(self):
        """Should hold statistics data."""
        now = datetime.now(timezone.utc)
        stats = ArchiveStats(
            total_count=100,
            by_reason={"decay": 70, "manual": 20, "consolidation": 10},
            by_type={"heuristic": 50, "outcome": 30, "knowledge": 20},
            by_agent={"agent-a": 60, "agent-b": 40},
            restored_count=5,
            oldest_archive=now - timedelta(days=365),
            newest_archive=now,
        )

        assert stats.total_count == 100
        assert stats.by_reason["decay"] == 70
        assert stats.by_type["heuristic"] == 50
        assert stats.by_agent["agent-a"] == 60
        assert stats.restored_count == 5
        assert stats.oldest_archive is not None
        assert stats.newest_archive is not None

    def test_to_dict(self):
        """Should convert to dictionary."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)
        stats = ArchiveStats(
            total_count=50,
            by_reason={"decay": 40, "manual": 10},
            by_type={"heuristic": 30, "outcome": 20},
            by_agent={"agent-1": 50},
            restored_count=2,
            oldest_archive=old,
            newest_archive=now,
        )

        d = stats.to_dict()

        assert d["total_count"] == 50
        assert d["by_reason"] == {"decay": 40, "manual": 10}
        assert d["by_type"] == {"heuristic": 30, "outcome": 20}
        assert d["by_agent"] == {"agent-1": 50}
        assert d["restored_count"] == 2
        assert d["oldest_archive"] == old.isoformat()
        assert d["newest_archive"] == now.isoformat()

    def test_to_dict_with_none_dates(self):
        """Should handle None dates in to_dict."""
        stats = ArchiveStats(total_count=0)

        d = stats.to_dict()

        assert d["oldest_archive"] is None
        assert d["newest_archive"] is None


class TestArchiveReasonUsage:
    """Tests for using ArchiveReason enum values."""

    def test_reason_in_archived_memory(self):
        """Should use ArchiveReason value in ArchivedMemory."""
        archived = ArchivedMemory.create(
            original_id="mem-1",
            memory_type="heuristic",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason=ArchiveReason.DECAY.value,
            final_strength=0.05,
        )

        assert archived.archive_reason == "decay"

    def test_reason_string_comparison(self):
        """Should allow string comparison with enum values."""
        reason = "decay"
        assert reason == ArchiveReason.DECAY.value

    def test_all_reasons_are_strings(self):
        """All reason values should be strings."""
        for reason in ArchiveReason:
            assert isinstance(reason.value, str)


class TestArchivedMemoryEdgeCases:
    """Edge case tests for ArchivedMemory."""

    def test_empty_content(self):
        """Should handle empty content string."""
        archived = ArchivedMemory.create(
            original_id="mem-empty",
            memory_type="heuristic",
            content="",
            project_id="proj",
            agent="agent",
            archive_reason="manual",
            final_strength=0.0,
        )

        assert archived.content == ""

    def test_null_embedding(self):
        """Should handle None embedding."""
        archived = ArchivedMemory.create(
            original_id="mem-no-emb",
            memory_type="outcome",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="decay",
            final_strength=0.05,
            embedding=None,
        )

        assert archived.embedding is None

    def test_empty_metadata(self):
        """Should default to empty dict for metadata."""
        archived = ArchivedMemory.create(
            original_id="mem-no-meta",
            memory_type="knowledge",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="consolidation",
            final_strength=0.1,
        )

        assert archived.metadata == {}

    def test_zero_final_strength(self):
        """Should handle zero final strength."""
        archived = ArchivedMemory.create(
            original_id="mem-zero",
            memory_type="anti_pattern",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="decay",
            final_strength=0.0,
        )

        assert archived.final_strength == 0.0

    def test_large_embedding(self):
        """Should handle large embedding vectors."""
        embedding = [0.1] * 1536  # Common embedding size
        archived = ArchivedMemory.create(
            original_id="mem-large",
            memory_type="heuristic",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="manual",
            final_strength=0.5,
            embedding=embedding,
        )

        assert len(archived.embedding) == 1536

    def test_complex_metadata(self):
        """Should handle complex nested metadata."""
        metadata = {
            "source": "test",
            "tags": ["a", "b", "c"],
            "nested": {"key": "value", "list": [1, 2, 3]},
            "unicode": "Hello \u4e16\u754c",
        }
        archived = ArchivedMemory.create(
            original_id="mem-complex",
            memory_type="knowledge",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="manual",
            final_strength=0.3,
            metadata=metadata,
        )

        assert archived.metadata == metadata


class TestArchiveIdGeneration:
    """Tests for archive ID generation."""

    def test_unique_ids(self):
        """Should generate unique IDs."""
        ids = set()
        for _ in range(100):
            archived = ArchivedMemory.create(
                original_id="mem-same",
                memory_type="heuristic",
                content="{}",
                project_id="proj",
                agent="agent",
                archive_reason="decay",
                final_strength=0.05,
            )
            ids.add(archived.id)

        # All 100 should be unique
        assert len(ids) == 100

    def test_id_format(self):
        """Should follow expected format."""
        archived = ArchivedMemory.create(
            original_id="mem-test",
            memory_type="outcome",
            content="{}",
            project_id="proj",
            agent="agent",
            archive_reason="manual",
            final_strength=0.1,
        )

        assert archived.id.startswith("archive-")
        # Should be archive- plus 12 hex chars
        assert len(archived.id) == len("archive-") + 12
