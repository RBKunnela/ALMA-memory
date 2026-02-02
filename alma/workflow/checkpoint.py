"""
ALMA Workflow Checkpoints.

Provides checkpoint dataclass and CheckpointManager for crash recovery
and state persistence in workflow orchestration.

Sprint 1 Task 1.3, Sprint 3 Tasks 3.1-3.4
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Default maximum state size (1MB)
DEFAULT_MAX_STATE_SIZE = 1024 * 1024


@dataclass
class Checkpoint:
    """
    Represents a workflow execution checkpoint.

    Checkpoints enable crash recovery by persisting state at key points
    during workflow execution. They support parallel execution through
    branch tracking and parent references.

    Attributes:
        id: Unique checkpoint identifier
        run_id: The workflow run this checkpoint belongs to
        node_id: The node that created this checkpoint
        state: The serializable state data
        sequence_number: Ordering within the run (monotonically increasing)
        branch_id: Identifier for parallel branch (None for main branch)
        parent_checkpoint_id: Previous checkpoint in the chain
        state_hash: SHA256 hash for change detection
        metadata: Additional checkpoint metadata
        created_at: When this checkpoint was created
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    node_id: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    branch_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    state_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Compute state hash if not provided."""
        if not self.state_hash and self.state:
            self.state_hash = self._compute_hash(self.state)

    @staticmethod
    def _compute_hash(state: Dict[str, Any]) -> str:
        """Compute SHA256 hash of state for change detection."""
        # Sort keys for consistent hashing
        state_json = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(state_json.encode()).hexdigest()

    def has_changed(self, other_state: Dict[str, Any]) -> bool:
        """Check if state has changed compared to another state."""
        other_hash = self._compute_hash(other_state)
        return self.state_hash != other_hash

    def get_state_size(self) -> int:
        """Get the size of the state in bytes."""
        return len(json.dumps(self.state, default=str).encode())

    def validate(self, max_state_size: int = DEFAULT_MAX_STATE_SIZE) -> None:
        """
        Validate the checkpoint.

        Args:
            max_state_size: Maximum allowed state size in bytes.

        Raises:
            ValueError: If validation fails.
        """
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.node_id:
            raise ValueError("node_id is required")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")

        state_size = self.get_state_size()
        if state_size > max_state_size:
            raise ValueError(
                f"State size ({state_size} bytes) exceeds maximum "
                f"({max_state_size} bytes). Consider storing large data "
                "in artifact storage and linking via ArtifactRef."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "state": self.state,
            "sequence_number": self.sequence_number,
            "branch_id": self.branch_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "state_hash": self.state_hash,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", str(uuid4())),
            run_id=data.get("run_id", ""),
            node_id=data.get("node_id", ""),
            state=data.get("state", {}),
            sequence_number=data.get("sequence_number", 0),
            branch_id=data.get("branch_id"),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            state_hash=data.get("state_hash", ""),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )


class CheckpointManager:
    """
    Manages checkpoint operations for workflow execution.

    Provides methods for creating, retrieving, and cleaning up checkpoints.
    Supports concurrent access patterns and branch management.

    Sprint 3 Tasks 3.1-3.4
    """

    def __init__(
        self,
        storage: Any,  # StorageBackend
        max_state_size: int = DEFAULT_MAX_STATE_SIZE,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            storage: Storage backend for persisting checkpoints.
            max_state_size: Maximum allowed state size in bytes.
        """
        self._storage = storage
        self._max_state_size = max_state_size
        self._sequence_cache: Dict[str, int] = {}  # run_id -> last sequence

    def create_checkpoint(
        self,
        run_id: str,
        node_id: str,
        state: Dict[str, Any],
        branch_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_if_unchanged: bool = True,
    ) -> Optional[Checkpoint]:
        """
        Create a new checkpoint.

        Args:
            run_id: The workflow run identifier.
            node_id: The node creating this checkpoint.
            state: The state to persist.
            branch_id: Optional branch identifier for parallel execution.
            parent_checkpoint_id: Previous checkpoint in the chain.
            metadata: Additional checkpoint metadata.
            skip_if_unchanged: If True, skip creating checkpoint if state
                              hasn't changed from the last checkpoint.

        Returns:
            The created Checkpoint, or None if skipped due to no changes.

        Raises:
            ValueError: If state exceeds max_state_size.
        """
        # Get next sequence number
        sequence_number = self._get_next_sequence(run_id)

        # Create checkpoint
        checkpoint = Checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            sequence_number=sequence_number,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata or {},
        )

        # Validate
        checkpoint.validate(self._max_state_size)

        # Check if state has changed (optional optimization)
        if skip_if_unchanged and parent_checkpoint_id:
            parent = self.get_checkpoint(parent_checkpoint_id)
            if parent and not parent.has_changed(state):
                return None

        # Persist
        self._storage.save_checkpoint(checkpoint)

        # Update sequence cache
        self._sequence_cache[run_id] = sequence_number

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._storage.get_checkpoint(checkpoint_id)

    def get_latest_checkpoint(
        self,
        run_id: str,
        branch_id: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint for a run.

        Args:
            run_id: The workflow run identifier.
            branch_id: Optional branch to filter by.

        Returns:
            The latest checkpoint, or None if no checkpoints exist.
        """
        return self._storage.get_latest_checkpoint(run_id, branch_id)

    def get_resume_point(self, run_id: str) -> Optional[Checkpoint]:
        """
        Get the checkpoint to resume from after a crash.

        This is an alias for get_latest_checkpoint for clarity.

        Args:
            run_id: The workflow run identifier.

        Returns:
            The checkpoint to resume from, or None if no checkpoints.
        """
        return self.get_latest_checkpoint(run_id)

    def get_branch_checkpoints(
        self,
        run_id: str,
        branch_ids: List[str],
    ) -> Dict[str, Checkpoint]:
        """
        Get the latest checkpoint for each branch.

        Used for parallel merge operations.

        Args:
            run_id: The workflow run identifier.
            branch_ids: List of branch identifiers.

        Returns:
            Dictionary mapping branch_id to its latest checkpoint.
        """
        result = {}
        for branch_id in branch_ids:
            checkpoint = self.get_latest_checkpoint(run_id, branch_id)
            if checkpoint:
                result[branch_id] = checkpoint
        return result

    def cleanup_checkpoints(
        self,
        run_id: str,
        keep_latest: int = 1,
    ) -> int:
        """
        Clean up old checkpoints for a completed run.

        Args:
            run_id: The workflow run identifier.
            keep_latest: Number of latest checkpoints to keep.

        Returns:
            Number of checkpoints deleted.
        """
        return self._storage.cleanup_checkpoints(run_id, keep_latest)

    def _get_next_sequence(self, run_id: str) -> int:
        """Get the next sequence number for a run."""
        if run_id in self._sequence_cache:
            return self._sequence_cache[run_id] + 1

        # Query storage for latest sequence
        latest = self.get_latest_checkpoint(run_id)
        if latest:
            self._sequence_cache[run_id] = latest.sequence_number
            return latest.sequence_number + 1

        return 0
