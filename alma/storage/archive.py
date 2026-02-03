"""
ALMA Memory Archive System.

Provides a safety net for memories before permanent deletion, supporting:
- Recovery of accidentally deleted memories
- Compliance and audit requirements
- Analysis of forgotten memories

Archives preserve full memory data with metadata about why/when archived.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ArchiveReason(Enum):
    """Reasons for archiving a memory."""

    DECAY = "decay"  # Natural decay below threshold
    MANUAL = "manual"  # User-initiated archival
    CONSOLIDATION = "consolidation"  # Merged into another memory
    SUPERSEDED = "superseded"  # Replaced by better memory
    QUOTA = "quota"  # Agent quota exceeded
    CLEANUP = "cleanup"  # General cleanup operation


@dataclass
class ArchivedMemory:
    """
    A memory that has been archived (soft-deleted).

    Archives preserve the full memory content along with metadata
    about when and why the memory was archived, enabling recovery
    and audit capabilities.

    Attributes:
        id: Unique archive identifier
        original_id: Original memory ID before archival
        memory_type: Type of memory (heuristic, outcome, knowledge, etc.)
        content: Serialized memory content (JSON)
        embedding: Original embedding vector (if available)
        metadata: Additional memory metadata
        original_created_at: When the memory was originally created
        archived_at: When the memory was archived
        archive_reason: Why the memory was archived
        final_strength: Memory strength at time of archival
        project_id: Project the memory belonged to
        agent: Agent the memory belonged to
        restored: Whether this archive has been restored
        restored_at: When the archive was restored (if applicable)
        restored_as: New memory ID after restoration (if applicable)
    """

    id: str
    original_id: str
    memory_type: str
    content: str  # JSON serialized memory content
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    original_created_at: datetime
    archived_at: datetime
    archive_reason: str
    final_strength: float
    project_id: str
    agent: str
    restored: bool = False
    restored_at: Optional[datetime] = None
    restored_as: Optional[str] = None

    @classmethod
    def create(
        cls,
        original_id: str,
        memory_type: str,
        content: str,
        project_id: str,
        agent: str,
        archive_reason: str,
        final_strength: float,
        original_created_at: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ArchivedMemory":
        """
        Create a new archived memory.

        Args:
            original_id: Original memory ID
            memory_type: Type of memory
            content: Serialized memory content
            project_id: Project ID
            agent: Agent name
            archive_reason: Reason for archival
            final_strength: Strength at archival
            original_created_at: Original creation time
            embedding: Optional embedding vector
            metadata: Optional additional metadata

        Returns:
            New ArchivedMemory instance
        """
        now = datetime.now(timezone.utc)
        return cls(
            id=f"archive-{uuid.uuid4().hex[:12]}",
            original_id=original_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            original_created_at=original_created_at or now,
            archived_at=now,
            archive_reason=archive_reason,
            final_strength=final_strength,
            project_id=project_id,
            agent=agent,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "original_id": self.original_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "original_created_at": self.original_created_at.isoformat(),
            "archived_at": self.archived_at.isoformat(),
            "archive_reason": self.archive_reason,
            "final_strength": self.final_strength,
            "project_id": self.project_id,
            "agent": self.agent,
            "restored": self.restored,
            "restored_at": self.restored_at.isoformat() if self.restored_at else None,
            "restored_as": self.restored_as,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchivedMemory":
        """Create from dictionary."""
        original_created_at = data.get("original_created_at")
        if isinstance(original_created_at, str):
            original_created_at = datetime.fromisoformat(
                original_created_at.replace("Z", "+00:00")
            )

        archived_at = data.get("archived_at")
        if isinstance(archived_at, str):
            archived_at = datetime.fromisoformat(archived_at.replace("Z", "+00:00"))
        elif archived_at is None:
            archived_at = datetime.now(timezone.utc)

        restored_at = data.get("restored_at")
        if isinstance(restored_at, str):
            restored_at = datetime.fromisoformat(restored_at.replace("Z", "+00:00"))

        return cls(
            id=data["id"],
            original_id=data["original_id"],
            memory_type=data["memory_type"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            original_created_at=original_created_at,
            archived_at=archived_at,
            archive_reason=data["archive_reason"],
            final_strength=data.get("final_strength", 0.0),
            project_id=data["project_id"],
            agent=data["agent"],
            restored=data.get("restored", False),
            restored_at=restored_at,
            restored_as=data.get("restored_as"),
        )

    def mark_restored(self, new_memory_id: str) -> None:
        """Mark this archive as restored."""
        self.restored = True
        self.restored_at = datetime.now(timezone.utc)
        self.restored_as = new_memory_id


@dataclass
class ArchiveConfig:
    """Configuration for memory archiving."""

    enabled: bool = True
    retention_days: int = 365  # Keep archives for 1 year by default
    auto_purge: bool = False  # Manual purge only by default
    archive_on_decay: bool = True  # Archive when decaying memories
    archive_on_consolidation: bool = True  # Archive when consolidating

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchiveConfig":
        """Create from configuration dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            retention_days=data.get("retention_days", 365),
            auto_purge=data.get("auto_purge", False),
            archive_on_decay=data.get("archive_on_decay", True),
            archive_on_consolidation=data.get("archive_on_consolidation", True),
        )


@dataclass
class ArchiveStats:
    """Statistics about archived memories."""

    total_count: int = 0
    by_reason: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_agent: Dict[str, int] = field(default_factory=dict)
    restored_count: int = 0
    oldest_archive: Optional[datetime] = None
    newest_archive: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "by_reason": self.by_reason,
            "by_type": self.by_type,
            "by_agent": self.by_agent,
            "restored_count": self.restored_count,
            "oldest_archive": self.oldest_archive.isoformat()
            if self.oldest_archive
            else None,
            "newest_archive": self.newest_archive.isoformat()
            if self.newest_archive
            else None,
        }
