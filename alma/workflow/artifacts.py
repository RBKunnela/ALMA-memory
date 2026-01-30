"""
ALMA Workflow Artifacts.

Provides artifact reference dataclass for linking external artifacts
(files, screenshots, logs, etc.) to memories.

Sprint 1 Task 1.4
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


class ArtifactType(Enum):
    """Types of artifacts that can be linked to memories."""

    # Files and documents
    FILE = "file"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"

    # Development artifacts
    SCREENSHOT = "screenshot"
    LOG = "log"
    TRACE = "trace"
    DIFF = "diff"

    # Test artifacts
    TEST_RESULT = "test_result"
    COVERAGE_REPORT = "coverage_report"

    # Analysis artifacts
    REPORT = "report"
    METRICS = "metrics"

    # Generic
    OTHER = "other"


@dataclass
class ArtifactRef:
    """
    Reference to an external artifact linked to a memory.

    Artifacts are stored externally (e.g., Cloudflare R2, S3, local filesystem)
    and referenced by URL/path. This allows memories to reference large files
    without bloating the memory database.

    Attributes:
        id: Unique artifact reference identifier
        memory_id: The memory this artifact is linked to
        artifact_type: Type of artifact (screenshot, log, etc.)
        storage_url: URL or path to the artifact in storage
        filename: Original filename
        mime_type: MIME type of the artifact
        size_bytes: Size of the artifact in bytes
        checksum: SHA256 checksum for integrity verification
        metadata: Additional artifact metadata
        created_at: When this reference was created
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    memory_id: str = ""
    artifact_type: ArtifactType = ArtifactType.OTHER
    storage_url: str = ""
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> None:
        """
        Validate the artifact reference.

        Raises:
            ValueError: If validation fails.
        """
        if not self.memory_id:
            raise ValueError("memory_id is required")
        if not self.storage_url:
            raise ValueError("storage_url is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "artifact_type": self.artifact_type.value,
            "storage_url": self.storage_url,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRef":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        artifact_type = data.get("artifact_type", "other")
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)

        return cls(
            id=data.get("id", str(uuid4())),
            memory_id=data.get("memory_id", ""),
            artifact_type=artifact_type,
            storage_url=data.get("storage_url", ""),
            filename=data.get("filename"),
            mime_type=data.get("mime_type"),
            size_bytes=data.get("size_bytes"),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


def link_artifact(
    memory_id: str,
    artifact_type: ArtifactType,
    storage_url: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ArtifactRef:
    """
    Create an artifact reference linked to a memory.

    This is a convenience function for creating ArtifactRef instances.

    Args:
        memory_id: The memory to link the artifact to.
        artifact_type: Type of artifact.
        storage_url: URL or path to the artifact.
        filename: Original filename.
        mime_type: MIME type.
        size_bytes: Size in bytes.
        checksum: SHA256 checksum.
        metadata: Additional metadata.

    Returns:
        A validated ArtifactRef instance.
    """
    ref = ArtifactRef(
        memory_id=memory_id,
        artifact_type=artifact_type,
        storage_url=storage_url,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        checksum=checksum,
        metadata=metadata or {},
    )
    ref.validate()
    return ref
