"""
Progress Tracking Types.

Data models for tracking work items and progress.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
import uuid


WorkItemStatus = Literal[
    "pending",      # Not started
    "in_progress",  # Currently being worked on
    "blocked",      # Waiting on something
    "review",       # Completed, awaiting review
    "done",         # Completed and verified
    "failed",       # Could not complete
]


@dataclass
class WorkItem:
    """
    A trackable unit of work.

    Can represent features, bugs, tasks, research questions,
    or any domain-specific work unit.
    """

    id: str
    project_id: str
    agent: Optional[str]

    # Work item details
    title: str
    description: str
    item_type: str  # "feature", "bug", "task", "research_question", etc.
    status: WorkItemStatus = "pending"
    priority: int = 50  # 0-100, higher = more important

    # Progress tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    time_spent_ms: int = 0
    attempt_count: int = 0

    # Relationships
    parent_id: Optional[str] = None
    blocks: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)

    # Validation
    tests: List[str] = field(default_factory=list)
    tests_passing: bool = False
    acceptance_criteria: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_id: str,
        title: str,
        description: str,
        item_type: str = "task",
        agent: Optional[str] = None,
        priority: int = 50,
        parent_id: Optional[str] = None,
        **kwargs,
    ) -> "WorkItem":
        """Factory method to create a new work item."""
        return cls(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent=agent,
            title=title,
            description=description,
            item_type=item_type,
            priority=priority,
            parent_id=parent_id,
            **kwargs,
        )

    def start(self) -> None:
        """Mark work item as started."""
        self.status = "in_progress"
        self.started_at = datetime.now(timezone.utc)
        self.attempt_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def complete(self, tests_passing: bool = True) -> None:
        """Mark work item as completed."""
        self.status = "done"
        self.completed_at = datetime.now(timezone.utc)
        self.tests_passing = tests_passing
        if self.started_at:
            self.time_spent_ms += int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )
        self.updated_at = datetime.now(timezone.utc)

    def block(self, blocked_by: Optional[str] = None, reason: str = "") -> None:
        """Mark work item as blocked."""
        self.status = "blocked"
        if blocked_by:
            self.blocked_by.append(blocked_by)
        if reason:
            self.metadata["block_reason"] = reason
        self.updated_at = datetime.now(timezone.utc)

    def fail(self, reason: str = "") -> None:
        """Mark work item as failed."""
        self.status = "failed"
        if reason:
            self.metadata["failure_reason"] = reason
        self.updated_at = datetime.now(timezone.utc)

    def is_actionable(self) -> bool:
        """Check if work item can be worked on."""
        return (
            self.status in ("pending", "in_progress")
            and len(self.blocked_by) == 0
        )


@dataclass
class ProgressLog:
    """
    Session-level progress snapshot.

    Records the state of progress at a point in time.
    """

    id: str
    project_id: str
    agent: str
    session_id: str

    # Progress counts
    items_total: int
    items_done: int
    items_in_progress: int
    items_blocked: int
    items_pending: int

    # Current focus
    current_item_id: Optional[str]
    current_action: str

    # Session metrics
    session_start: datetime
    actions_taken: int = 0
    outcomes_recorded: int = 0

    # Timestamp
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        project_id: str,
        agent: str,
        session_id: str,
        items_total: int,
        items_done: int,
        items_in_progress: int,
        items_blocked: int,
        items_pending: int,
        current_item_id: Optional[str] = None,
        current_action: str = "",
        session_start: Optional[datetime] = None,
    ) -> "ProgressLog":
        """Factory method to create progress log."""
        return cls(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent=agent,
            session_id=session_id,
            items_total=items_total,
            items_done=items_done,
            items_in_progress=items_in_progress,
            items_blocked=items_blocked,
            items_pending=items_pending,
            current_item_id=current_item_id,
            current_action=current_action,
            session_start=session_start or datetime.now(timezone.utc),
        )


@dataclass
class ProgressSummary:
    """
    Summary of progress for display/reporting.

    A simplified view of progress state.
    """

    project_id: str
    agent: Optional[str]

    # Counts
    total: int
    done: int
    in_progress: int
    blocked: int
    pending: int
    failed: int

    # Percentages
    completion_rate: float  # 0-1
    success_rate: float  # done / (done + failed)

    # Current focus
    current_item: Optional[WorkItem]
    next_suggested: Optional[WorkItem]

    # Blockers
    blockers: List[WorkItem]

    # Time tracking
    total_time_ms: int
    avg_time_per_item_ms: float

    # Timestamps
    last_activity: Optional[datetime]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def completion_percentage(self) -> int:
        """Get completion as percentage."""
        return int(self.completion_rate * 100)

    def format_summary(self) -> str:
        """Format as human-readable string."""
        lines = [
            f"Progress: {self.done}/{self.total} ({self.completion_percentage}%)",
            f"  In Progress: {self.in_progress}",
            f"  Blocked: {self.blocked}",
            f"  Pending: {self.pending}",
        ]
        if self.current_item:
            lines.append(f"  Current: {self.current_item.title}")
        if self.next_suggested:
            lines.append(f"  Next: {self.next_suggested.title}")
        if self.blockers:
            lines.append(f"  Blockers: {len(self.blockers)} items")
        return "\n".join(lines)
