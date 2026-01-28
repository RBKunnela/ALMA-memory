"""
Progress Tracker.

Manages work items and provides progress tracking functionality.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from alma.progress.types import (
    ProgressLog,
    ProgressSummary,
    WorkItem,
    WorkItemStatus,
)
from alma.storage.base import StorageBackend

logger = logging.getLogger(__name__)


SelectionStrategy = Literal[
    "priority",  # Highest priority first
    "blocked_unblock",  # Items that unblock others
    "quick_win",  # Smallest/easiest first
    "fifo",  # First in, first out
]


class ProgressTracker:
    """
    Track work item progress.

    Provides methods for creating, updating, and querying work items,
    as well as generating progress summaries and suggesting next actions.
    """

    def __init__(
        self,
        project_id: str,
        storage: Optional[StorageBackend] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            project_id: Project to track progress for
            storage: Optional storage backend for persistence.
                     If not provided, uses in-memory storage only.
        """
        self.storage = storage
        self.project_id = project_id
        self._work_items: Dict[str, WorkItem] = {}
        self._progress_logs: List[ProgressLog] = []

    # ==================== WORK ITEM CRUD ====================

    def create_work_item(
        self,
        title: str,
        description: str,
        item_type: str = "task",
        agent: Optional[str] = None,
        priority: int = 50,
        parent_id: Optional[str] = None,
        **kwargs,
    ) -> WorkItem:
        """
        Create a new work item.

        Args:
            title: Short title for the work item
            description: Detailed description
            item_type: Type of work (task, feature, bug, etc.)
            agent: Agent responsible for this item
            priority: Priority 0-100 (higher = more important)
            parent_id: Parent work item ID for hierarchies
            **kwargs: Additional fields for WorkItem

        Returns:
            Created WorkItem
        """
        item = WorkItem.create(
            project_id=self.project_id,
            title=title,
            description=description,
            item_type=item_type,
            agent=agent,
            priority=priority,
            parent_id=parent_id,
            **kwargs,
        )
        self._work_items[item.id] = item
        logger.info(f"Created work item: {item.id} - {item.title}")
        return item

    def get_work_item(self, item_id: str) -> Optional[WorkItem]:
        """Get a work item by ID."""
        return self._work_items.get(item_id)

    def update_work_item(
        self,
        item_id: str,
        **updates,
    ) -> Optional[WorkItem]:
        """
        Update a work item's fields.

        Args:
            item_id: ID of work item to update
            **updates: Fields to update

        Returns:
            Updated WorkItem or None if not found
        """
        item = self._work_items.get(item_id)
        if not item:
            logger.warning(f"Work item not found: {item_id}")
            return None

        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)

        item.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updated work item: {item_id}")
        return item

    def delete_work_item(self, item_id: str) -> bool:
        """Delete a work item."""
        if item_id in self._work_items:
            del self._work_items[item_id]
            logger.info(f"Deleted work item: {item_id}")
            return True
        return False

    # ==================== STATUS UPDATES ====================

    def update_status(
        self,
        item_id: str,
        status: WorkItemStatus,
        notes: Optional[str] = None,
    ) -> Optional[WorkItem]:
        """
        Update work item status.

        Args:
            item_id: ID of work item
            status: New status
            notes: Optional notes about the status change

        Returns:
            Updated WorkItem or None if not found
        """
        item = self._work_items.get(item_id)
        if not item:
            return None

        old_status = item.status
        item.status = status
        item.updated_at = datetime.now(timezone.utc)

        # Handle status-specific updates
        if status == "in_progress" and old_status != "in_progress":
            item.start()
        elif status == "done":
            item.complete()
        elif status == "blocked":
            item.block(reason=notes or "")
        elif status == "failed":
            item.fail(reason=notes or "")

        if notes:
            if "status_notes" not in item.metadata:
                item.metadata["status_notes"] = []
            item.metadata["status_notes"].append(
                {
                    "from": old_status,
                    "to": status,
                    "notes": notes,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        logger.info(f"Status updated: {item_id} {old_status} -> {status}")
        return item

    def start_item(self, item_id: str) -> Optional[WorkItem]:
        """Start working on an item."""
        return self.update_status(item_id, "in_progress")

    def complete_item(
        self,
        item_id: str,
        tests_passing: bool = True,
    ) -> Optional[WorkItem]:
        """Complete an item."""
        item = self.update_status(item_id, "done")
        if item:
            item.tests_passing = tests_passing
        return item

    def block_item(
        self,
        item_id: str,
        blocked_by: Optional[str] = None,
        reason: str = "",
    ) -> Optional[WorkItem]:
        """Block an item."""
        item = self._work_items.get(item_id)
        if item:
            item.block(blocked_by=blocked_by, reason=reason)
        return item

    def unblock_item(self, item_id: str) -> Optional[WorkItem]:
        """Unblock an item."""
        item = self._work_items.get(item_id)
        if item and item.status == "blocked":
            item.status = "pending"
            item.blocked_by = []
            item.updated_at = datetime.now(timezone.utc)
        return item

    # ==================== QUERIES ====================

    def get_items(
        self,
        status: Optional[WorkItemStatus] = None,
        agent: Optional[str] = None,
        item_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> List[WorkItem]:
        """
        Get work items with optional filters.

        Args:
            status: Filter by status
            agent: Filter by agent
            item_type: Filter by type
            parent_id: Filter by parent ID

        Returns:
            List of matching work items
        """
        results = []
        for item in self._work_items.values():
            if status and item.status != status:
                continue
            if agent and item.agent != agent:
                continue
            if item_type and item.item_type != item_type:
                continue
            if parent_id is not None and item.parent_id != parent_id:
                continue
            results.append(item)
        return results

    def get_actionable_items(
        self,
        agent: Optional[str] = None,
    ) -> List[WorkItem]:
        """Get items that can be worked on (not blocked, not done)."""
        return [
            item
            for item in self._work_items.values()
            if item.is_actionable()
            and (agent is None or item.agent == agent or item.agent is None)
        ]

    def get_blocked_items(
        self,
        agent: Optional[str] = None,
    ) -> List[WorkItem]:
        """Get blocked items."""
        return self.get_items(status="blocked", agent=agent)

    def get_in_progress_items(
        self,
        agent: Optional[str] = None,
    ) -> List[WorkItem]:
        """Get items currently in progress."""
        return self.get_items(status="in_progress", agent=agent)

    # ==================== NEXT ITEM SELECTION ====================

    def get_next_item(
        self,
        agent: Optional[str] = None,
        strategy: SelectionStrategy = "priority",
    ) -> Optional[WorkItem]:
        """
        Get next work item to focus on.

        Args:
            agent: Filter to specific agent
            strategy: Selection strategy

        Returns:
            Suggested next work item
        """
        actionable = self.get_actionable_items(agent=agent)
        if not actionable:
            return None

        if strategy == "priority":
            # Highest priority first
            actionable.sort(key=lambda x: -x.priority)
            return actionable[0]

        elif strategy == "blocked_unblock":
            # Items that unblock the most other items
            unblock_counts = {}
            for item in actionable:
                count = sum(
                    1
                    for other in self._work_items.values()
                    if item.id in other.blocked_by
                )
                unblock_counts[item.id] = count
            actionable.sort(key=lambda x: -unblock_counts.get(x.id, 0))
            return actionable[0]

        elif strategy == "quick_win":
            # Prefer items with fewer acceptance criteria (proxy for complexity)
            actionable.sort(key=lambda x: len(x.acceptance_criteria))
            return actionable[0]

        elif strategy == "fifo":
            # First created first
            actionable.sort(key=lambda x: x.created_at)
            return actionable[0]

        return actionable[0] if actionable else None

    # ==================== PROGRESS SUMMARY ====================

    def get_progress_summary(
        self,
        agent: Optional[str] = None,
    ) -> ProgressSummary:
        """
        Get progress summary.

        Args:
            agent: Filter to specific agent

        Returns:
            ProgressSummary with counts and statistics
        """
        items = list(self._work_items.values())
        if agent:
            items = [i for i in items if i.agent == agent or i.agent is None]

        total = len(items)
        done = len([i for i in items if i.status == "done"])
        in_progress = len([i for i in items if i.status == "in_progress"])
        blocked = len([i for i in items if i.status == "blocked"])
        pending = len([i for i in items if i.status == "pending"])
        failed = len([i for i in items if i.status == "failed"])

        completion_rate = done / total if total > 0 else 0.0
        success_rate = done / (done + failed) if (done + failed) > 0 else 0.0

        total_time = sum(i.time_spent_ms for i in items)
        avg_time = total_time / done if done > 0 else 0.0

        current_item = None
        in_progress_items = [i for i in items if i.status == "in_progress"]
        if in_progress_items:
            current_item = in_progress_items[0]

        last_activity = None
        if items:
            last_activity = max(i.updated_at for i in items)

        return ProgressSummary(
            project_id=self.project_id,
            agent=agent,
            total=total,
            done=done,
            in_progress=in_progress,
            blocked=blocked,
            pending=pending,
            failed=failed,
            completion_rate=completion_rate,
            success_rate=success_rate,
            current_item=current_item,
            next_suggested=self.get_next_item(agent=agent),
            blockers=self.get_blocked_items(agent=agent),
            total_time_ms=total_time,
            avg_time_per_item_ms=avg_time,
            last_activity=last_activity,
        )

    # ==================== PROGRESS LOGGING ====================

    def log_progress(
        self,
        agent: str,
        session_id: str,
        current_action: str = "",
    ) -> ProgressLog:
        """
        Create a progress snapshot for the current session.

        Args:
            agent: Agent recording progress
            session_id: Current session ID
            current_action: What is currently being done

        Returns:
            ProgressLog snapshot
        """
        summary = self.get_progress_summary(agent=agent)

        log = ProgressLog.create(
            project_id=self.project_id,
            agent=agent,
            session_id=session_id,
            items_total=summary.total,
            items_done=summary.done,
            items_in_progress=summary.in_progress,
            items_blocked=summary.blocked,
            items_pending=summary.pending,
            current_item_id=summary.current_item.id if summary.current_item else None,
            current_action=current_action,
        )

        self._progress_logs.append(log)
        logger.debug(f"Progress logged: {log.id}")
        return log

    def get_progress_history(
        self,
        agent: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[ProgressLog]:
        """Get progress log history."""
        logs = self._progress_logs

        if agent:
            logs = [log for log in logs if log.agent == agent]
        if session_id:
            logs = [log for log in logs if log.session_id == session_id]

        # Sort by created_at descending and limit
        logs.sort(key=lambda x: x.created_at, reverse=True)
        return logs[:limit]

    # ==================== BULK OPERATIONS ====================

    def create_from_list(
        self,
        items: List[Dict[str, Any]],
        agent: Optional[str] = None,
    ) -> List[WorkItem]:
        """
        Create multiple work items from a list of dicts.

        Args:
            items: List of item definitions
            agent: Default agent for items

        Returns:
            List of created WorkItems
        """
        created = []
        for item_def in items:
            if "agent" not in item_def and agent:
                item_def["agent"] = agent
            item = self.create_work_item(**item_def)
            created.append(item)
        return created

    def expand_to_items(
        self,
        text: str,
        item_type: str = "task",
        agent: Optional[str] = None,
    ) -> List[WorkItem]:
        """
        Parse text into work items (simple line-based parsing).

        Each line starting with "- " becomes a work item.

        Args:
            text: Text to parse
            item_type: Type for created items
            agent: Agent for items

        Returns:
            List of created WorkItems
        """
        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                title = line[2:].strip()
                if title:
                    item = self.create_work_item(
                        title=title,
                        description=title,  # Same as title for now
                        item_type=item_type,
                        agent=agent,
                    )
                    items.append(item)
        return items

    # ==================== SERIALIZATION ====================

    def to_dict(self) -> Dict[str, Any]:
        """Export tracker state to dict."""
        return {
            "project_id": self.project_id,
            "work_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "description": item.description,
                    "item_type": item.item_type,
                    "status": item.status,
                    "priority": item.priority,
                    "agent": item.agent,
                    "parent_id": item.parent_id,
                    "blocks": item.blocks,
                    "blocked_by": item.blocked_by,
                    "tests": item.tests,
                    "tests_passing": item.tests_passing,
                    "time_spent_ms": item.time_spent_ms,
                    "attempt_count": item.attempt_count,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "started_at": (
                        item.started_at.isoformat() if item.started_at else None
                    ),
                    "completed_at": (
                        item.completed_at.isoformat() if item.completed_at else None
                    ),
                    "metadata": item.metadata,
                }
                for item in self._work_items.values()
            ],
            "progress_logs": [
                {
                    "id": log.id,
                    "session_id": log.session_id,
                    "agent": log.agent,
                    "items_total": log.items_total,
                    "items_done": log.items_done,
                    "current_action": log.current_action,
                    "created_at": log.created_at.isoformat(),
                }
                for log in self._progress_logs
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        storage: StorageBackend,
    ) -> "ProgressTracker":
        """Load tracker state from dict."""
        tracker = cls(storage=storage, project_id=data["project_id"])

        for item_data in data.get("work_items", []):
            # Parse dates
            created_at = datetime.fromisoformat(item_data["created_at"])
            updated_at = datetime.fromisoformat(item_data["updated_at"])
            started_at = (
                datetime.fromisoformat(item_data["started_at"])
                if item_data.get("started_at")
                else None
            )
            completed_at = (
                datetime.fromisoformat(item_data["completed_at"])
                if item_data.get("completed_at")
                else None
            )

            item = WorkItem(
                id=item_data["id"],
                project_id=data["project_id"],
                title=item_data["title"],
                description=item_data["description"],
                item_type=item_data["item_type"],
                status=item_data["status"],
                priority=item_data["priority"],
                agent=item_data.get("agent"),
                parent_id=item_data.get("parent_id"),
                blocks=item_data.get("blocks", []),
                blocked_by=item_data.get("blocked_by", []),
                tests=item_data.get("tests", []),
                tests_passing=item_data.get("tests_passing", False),
                time_spent_ms=item_data.get("time_spent_ms", 0),
                attempt_count=item_data.get("attempt_count", 0),
                created_at=created_at,
                updated_at=updated_at,
                started_at=started_at,
                completed_at=completed_at,
                metadata=item_data.get("metadata", {}),
            )
            tracker._work_items[item.id] = item

        return tracker
