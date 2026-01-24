"""
Session Management Types.

Data models for session continuity and handoffs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
import uuid


SessionOutcome = Literal["success", "failure", "interrupted", "unknown"]


@dataclass
class SessionHandoff:
    """
    Compressed context for session continuity.

    Captures the essential state at session end so the next session
    can quickly resume without full context reconstruction.
    """

    id: str
    project_id: str
    agent: str
    session_id: str

    # Where we left off
    last_action: str
    last_outcome: SessionOutcome
    current_goal: str

    # Quick context (not full history)
    key_decisions: List[str] = field(default_factory=list)  # Max 10 most important
    active_files: List[str] = field(default_factory=list)  # Files being worked on
    blockers: List[str] = field(default_factory=list)  # Current blockers
    next_steps: List[str] = field(default_factory=list)  # Planned next actions

    # Test/validation state
    test_status: Dict[str, bool] = field(default_factory=dict)  # test_name -> passing

    # Confidence signals
    confidence_level: float = 0.5  # 0-1, how well is this going
    risk_flags: List[str] = field(default_factory=list)  # Concerns noted

    # Timing
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_end: Optional[datetime] = None
    duration_ms: int = 0

    # For semantic retrieval
    embedding: Optional[List[float]] = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_id: str,
        agent: str,
        session_id: str,
        last_action: str,
        current_goal: str,
        last_outcome: SessionOutcome = "unknown",
        session_start: Optional[datetime] = None,
        **kwargs,
    ) -> "SessionHandoff":
        """Factory method to create a new session handoff."""
        return cls(
            id=str(uuid.uuid4()),
            project_id=project_id,
            agent=agent,
            session_id=session_id,
            last_action=last_action,
            last_outcome=last_outcome,
            current_goal=current_goal,
            session_start=session_start or datetime.now(timezone.utc),
            **kwargs,
        )

    def finalize(
        self,
        last_action: str,
        last_outcome: SessionOutcome,
        next_steps: Optional[List[str]] = None,
    ) -> None:
        """Finalize the handoff at session end."""
        self.last_action = last_action
        self.last_outcome = last_outcome
        self.session_end = datetime.now(timezone.utc)
        if self.session_start:
            self.duration_ms = int(
                (self.session_end - self.session_start).total_seconds() * 1000
            )
        if next_steps:
            self.next_steps = next_steps[:10]  # Cap at 10

    def add_decision(self, decision: str) -> None:
        """Record a key decision (max 10)."""
        if len(self.key_decisions) >= 10:
            self.key_decisions.pop(0)  # Remove oldest
        self.key_decisions.append(decision)

    def add_blocker(self, blocker: str) -> None:
        """Record a blocker."""
        if blocker not in self.blockers:
            self.blockers.append(blocker)

    def remove_blocker(self, blocker: str) -> None:
        """Remove a resolved blocker."""
        if blocker in self.blockers:
            self.blockers.remove(blocker)

    def set_test_status(self, test_name: str, passing: bool) -> None:
        """Update test status."""
        self.test_status[test_name] = passing

    def format_quick_reload(self) -> str:
        """
        Format handoff as a quick reload string.

        Returns a compact string that can be parsed by the next session
        for rapid context restoration.
        """
        lines = [
            f"## Session Handoff: {self.session_id}",
            f"Agent: {self.agent}",
            f"Goal: {self.current_goal}",
            f"Last Action: {self.last_action} ({self.last_outcome})",
            f"Confidence: {int(self.confidence_level * 100)}%",
        ]

        if self.next_steps:
            lines.append("\n### Next Steps:")
            for step in self.next_steps[:5]:
                lines.append(f"- {step}")

        if self.blockers:
            lines.append("\n### Blockers:")
            for blocker in self.blockers:
                lines.append(f"- {blocker}")

        if self.key_decisions:
            lines.append("\n### Key Decisions:")
            for decision in self.key_decisions[-5:]:  # Last 5 decisions
                lines.append(f"- {decision}")

        if self.active_files:
            lines.append("\n### Active Files:")
            for f in self.active_files[:5]:
                lines.append(f"- {f}")

        if self.risk_flags:
            lines.append("\n### Risks:")
            for risk in self.risk_flags:
                lines.append(f"- {risk}")

        # Test summary
        if self.test_status:
            passing = sum(1 for v in self.test_status.values() if v)
            total = len(self.test_status)
            lines.append(f"\n### Tests: {passing}/{total} passing")

        return "\n".join(lines)


@dataclass
class SessionContext:
    """
    Full context for starting/resuming a session.

    Provides everything an agent needs to orient itself at session start.
    """

    project_id: str
    agent: str
    session_id: str

    # Handoff from previous session
    previous_handoff: Optional[SessionHandoff] = None

    # Current progress state (from ProgressSummary)
    progress: Optional[Any] = None  # Will be ProgressSummary when integrated

    # Recent outcomes (from ALMA)
    recent_outcomes: List[Any] = field(default_factory=list)

    # Relevant heuristics (from ALMA)
    relevant_heuristics: List[Any] = field(default_factory=list)

    # Environment orientation
    codebase_state: Optional[Dict[str, Any]] = None  # git status, recent commits
    environment_state: Optional[Dict[str, Any]] = None  # running services, etc.

    # Suggested focus
    suggested_focus: Optional[Any] = None  # WorkItem when integrated

    # Rules of engagement
    rules_of_engagement: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        project_id: str,
        agent: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> "SessionContext":
        """Factory method to create new session context."""
        return cls(
            project_id=project_id,
            agent=agent,
            session_id=session_id or str(uuid.uuid4()),
            **kwargs,
        )

    def format_orientation(self) -> str:
        """
        Format context as an orientation briefing.

        Returns a structured string for quick agent orientation.
        """
        lines = [
            "# Session Orientation",
            f"Project: {self.project_id}",
            f"Agent: {self.agent}",
            f"Session: {self.session_id}",
        ]

        # Previous session summary
        if self.previous_handoff:
            lines.append("\n## Previous Session")
            lines.append(f"Outcome: {self.previous_handoff.last_outcome}")
            lines.append(f"Last Action: {self.previous_handoff.last_action}")
            lines.append(f"Goal: {self.previous_handoff.current_goal}")
            if self.previous_handoff.next_steps:
                lines.append("Next Steps from Last Session:")
                for step in self.previous_handoff.next_steps[:3]:
                    lines.append(f"  - {step}")

        # Current progress
        if self.progress:
            lines.append("\n## Progress")
            # Assuming progress has a format_summary method
            if hasattr(self.progress, "format_summary"):
                lines.append(self.progress.format_summary())
            else:
                lines.append(str(self.progress))

        # Blockers from previous session
        if self.previous_handoff and self.previous_handoff.blockers:
            lines.append("\n## Outstanding Blockers")
            for blocker in self.previous_handoff.blockers:
                lines.append(f"- {blocker}")

        # Suggested focus
        if self.suggested_focus:
            lines.append("\n## Suggested Focus")
            if hasattr(self.suggested_focus, "title"):
                lines.append(f"Task: {self.suggested_focus.title}")
            else:
                lines.append(str(self.suggested_focus))

        # Environment state
        if self.codebase_state:
            lines.append("\n## Codebase State")
            if "branch" in self.codebase_state:
                lines.append(f"Branch: {self.codebase_state['branch']}")
            if "uncommitted" in self.codebase_state:
                lines.append(f"Uncommitted Changes: {self.codebase_state['uncommitted']}")

        # Rules
        if self.rules_of_engagement:
            lines.append("\n## Rules of Engagement")
            for rule in self.rules_of_engagement:
                lines.append(f"- {rule}")

        return "\n".join(lines)
