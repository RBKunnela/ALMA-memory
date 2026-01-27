"""
Initializer Types.

Data structures for the Session Initializer pattern.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CodebaseOrientation:
    """Codebase orientation information."""

    # Git state
    current_branch: str
    has_uncommitted_changes: bool
    recent_commits: List[str]  # Last N commit messages

    # File structure
    root_path: str
    key_directories: List[str]  # src/, tests/, etc.
    config_files: List[str]  # package.json, pyproject.toml, etc.

    # Summary
    summary: Optional[str] = None

    def to_prompt(self) -> str:
        """Format orientation for prompt injection."""
        lines = [
            f"Branch: {self.current_branch}",
            f"Uncommitted changes: {'Yes' if self.has_uncommitted_changes else 'No'}",
        ]

        if self.recent_commits:
            lines.append("Recent commits:")
            for commit in self.recent_commits[:5]:
                lines.append(f"  - {commit}")

        if self.key_directories:
            lines.append(f"Key directories: {', '.join(self.key_directories)}")

        if self.summary:
            lines.append(f"Summary: {self.summary}")

        return "\n".join(lines)


@dataclass
class RulesOfEngagement:
    """Rules governing agent behavior during session."""

    # What agent CAN do
    scope_rules: List[str] = field(default_factory=list)

    # What agent CANNOT do
    constraints: List[str] = field(default_factory=list)

    # Must pass before marking "done"
    quality_gates: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format rules for prompt injection."""
        lines = []

        if self.scope_rules:
            lines.append("You CAN:")
            for rule in self.scope_rules:
                lines.append(f"  - {rule}")

        if self.constraints:
            lines.append("You CANNOT:")
            for constraint in self.constraints:
                lines.append(f"  - {constraint}")

        if self.quality_gates:
            lines.append("Before marking DONE, verify:")
            for gate in self.quality_gates:
                lines.append(f"  - {gate}")

        return "\n".join(lines)


@dataclass
class InitializationResult:
    """
    Result of session initialization.

    Contains everything an agent needs to start work:
    - Expanded goal and work items
    - Codebase orientation
    - Relevant memories
    - Rules of engagement
    - Recommended starting point
    """

    id: str
    session_id: str
    project_id: str
    agent: str

    # Original and expanded goal
    original_prompt: str
    goal: str

    # Work items extracted from goal
    work_items: List[Any] = field(default_factory=list)  # WorkItem objects

    # Codebase orientation
    orientation: Optional[CodebaseOrientation] = None

    # Recent activity
    recent_activity: List[str] = field(default_factory=list)

    # Relevant memories (MemorySlice)
    relevant_memories: Optional[Any] = None

    # Rules of engagement
    rules: RulesOfEngagement = field(default_factory=RulesOfEngagement)

    # Suggested first action
    recommended_start: Optional[Any] = None  # WorkItem

    # Metadata
    initialized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_id: str,
        agent: str,
        original_prompt: str,
        goal: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "InitializationResult":
        """Create a new initialization result."""
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
            project_id=project_id,
            agent=agent,
            original_prompt=original_prompt,
            goal=goal or original_prompt,
        )

    def to_prompt(self) -> str:
        """
        Format initialization result for prompt injection.

        This is the "briefing" that prepares the agent.
        """
        sections = []

        # Header
        sections.append(f"## Session Initialization for {self.agent}")
        sections.append(f"Project: {self.project_id}")
        sections.append(f"Session: {self.session_id}")
        sections.append("")

        # Goal
        sections.append("### Goal")
        sections.append(self.goal)
        sections.append("")

        # Work items
        if self.work_items:
            sections.append("### Work Items")
            for i, item in enumerate(self.work_items, 1):
                title = getattr(item, "title", str(item))
                status = getattr(item, "status", "pending")
                sections.append(f"{i}. [{status}] {title}")
            sections.append("")

        # Orientation
        if self.orientation:
            sections.append("### Codebase Orientation")
            sections.append(self.orientation.to_prompt())
            sections.append("")

        # Relevant memories
        if self.relevant_memories:
            sections.append("### Relevant Knowledge from Past Runs")
            if hasattr(self.relevant_memories, "to_prompt"):
                sections.append(self.relevant_memories.to_prompt())
            else:
                sections.append(str(self.relevant_memories))
            sections.append("")

        # Rules of engagement
        if self.rules.scope_rules or self.rules.constraints or self.rules.quality_gates:
            sections.append("### Rules of Engagement")
            sections.append(self.rules.to_prompt())
            sections.append("")

        # Recommended start
        if self.recommended_start:
            sections.append("### Recommended First Action")
            title = getattr(
                self.recommended_start, "title", str(self.recommended_start)
            )
            sections.append(f"Start with: {title}")
            sections.append("")

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "agent": self.agent,
            "original_prompt": self.original_prompt,
            "goal": self.goal,
            "work_items": [
                item.to_dict() if hasattr(item, "to_dict") else str(item)
                for item in self.work_items
            ],
            "orientation": {
                "current_branch": self.orientation.current_branch,
                "has_uncommitted_changes": self.orientation.has_uncommitted_changes,
                "recent_commits": self.orientation.recent_commits,
                "root_path": self.orientation.root_path,
                "key_directories": self.orientation.key_directories,
                "config_files": self.orientation.config_files,
                "summary": self.orientation.summary,
            }
            if self.orientation
            else None,
            "recent_activity": self.recent_activity,
            "rules": {
                "scope_rules": self.rules.scope_rules,
                "constraints": self.rules.constraints,
                "quality_gates": self.rules.quality_gates,
            },
            "recommended_start": (
                self.recommended_start.to_dict()
                if self.recommended_start and hasattr(self.recommended_start, "to_dict")
                else str(self.recommended_start)
                if self.recommended_start
                else None
            ),
            "initialized_at": self.initialized_at.isoformat(),
            "metadata": self.metadata,
        }
