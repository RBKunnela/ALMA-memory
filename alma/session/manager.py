"""
Session Manager.

Manages session continuity, handoffs, and quick context reload.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable
import uuid
import logging

from alma.session.types import (
    SessionHandoff,
    SessionContext,
    SessionOutcome,
)

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manage session continuity for AI agents.

    Provides:
    - Session start/end handling
    - Handoff creation and retrieval
    - Quick reload formatting
    - Integration with progress tracking
    """

    def __init__(
        self,
        project_id: str,
        storage: Optional[Any] = None,  # Will be StorageBackend when integrated
        progress_tracker: Optional[Any] = None,  # Will be ProgressTracker
        max_handoffs: int = 50,
    ):
        """
        Initialize session manager.

        Args:
            project_id: Project identifier
            storage: Optional storage backend for persistence
            progress_tracker: Optional progress tracker integration
            max_handoffs: Maximum handoffs to keep in memory
        """
        self.project_id = project_id
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.max_handoffs = max_handoffs

        # In-memory handoff storage (keyed by agent)
        self._handoffs: Dict[str, List[SessionHandoff]] = {}

        # Active sessions
        self._active_sessions: Dict[str, SessionHandoff] = {}

        # Context enrichers (callables that add context to sessions)
        self._context_enrichers: List[Callable[[SessionContext], SessionContext]] = []

    def register_enricher(
        self,
        enricher: Callable[[SessionContext], SessionContext],
    ) -> None:
        """
        Register a context enricher.

        Enrichers are called during session start to add context
        (e.g., git status, running services, etc.)
        """
        self._context_enrichers.append(enricher)

    def start_session(
        self,
        agent: str,
        goal: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SessionContext:
        """
        Start a new session, loading previous context.

        Args:
            agent: Agent identifier
            goal: Current session goal
            session_id: Optional session ID (generated if not provided)

        Returns:
            SessionContext with all relevant orientation data
        """
        session_id = session_id or str(uuid.uuid4())

        # Get previous handoff
        previous = self.get_latest_handoff(agent)

        # Create session context
        context = SessionContext.create(
            project_id=self.project_id,
            agent=agent,
            session_id=session_id,
            previous_handoff=previous,
        )

        # Get progress if tracker available
        if self.progress_tracker:
            try:
                context.progress = self.progress_tracker.get_progress_summary(agent)
            except Exception as e:
                logger.warning(f"Could not get progress: {e}")

        # Apply enrichers
        for enricher in self._context_enrichers:
            try:
                context = enricher(context)
            except Exception as e:
                logger.warning(f"Context enricher failed: {e}")

        # Create active session handoff for this session
        current_goal = goal or (previous.current_goal if previous else "Unknown")
        active_handoff = SessionHandoff.create(
            project_id=self.project_id,
            agent=agent,
            session_id=session_id,
            last_action="session_start",
            current_goal=current_goal,
            last_outcome="unknown",
        )

        # Carry over blockers from previous session
        if previous and previous.blockers:
            active_handoff.blockers = previous.blockers.copy()

        self._active_sessions[f"{agent}:{session_id}"] = active_handoff

        logger.info(
            f"Started session {session_id} for agent {agent} "
            f"(previous: {'yes' if previous else 'no'})"
        )

        return context

    def get_active_handoff(
        self,
        agent: str,
        session_id: str,
    ) -> Optional[SessionHandoff]:
        """Get the active handoff for current session."""
        return self._active_sessions.get(f"{agent}:{session_id}")

    def update_session(
        self,
        agent: str,
        session_id: str,
        action: Optional[str] = None,
        outcome: Optional[SessionOutcome] = None,
        decision: Optional[str] = None,
        blocker: Optional[str] = None,
        resolved_blocker: Optional[str] = None,
        active_file: Optional[str] = None,
        test_result: Optional[Dict[str, bool]] = None,
        confidence: Optional[float] = None,
        risk: Optional[str] = None,
    ) -> Optional[SessionHandoff]:
        """
        Update the active session with new information.

        This should be called periodically during a session to track state.

        Args:
            agent: Agent identifier
            session_id: Session identifier
            action: Latest action taken
            outcome: Outcome of latest action
            decision: Key decision made
            blocker: New blocker encountered
            resolved_blocker: Blocker that was resolved
            active_file: File currently being worked on
            test_result: Test name -> passing status
            confidence: Updated confidence level
            risk: New risk identified

        Returns:
            Updated SessionHandoff or None if session not found
        """
        key = f"{agent}:{session_id}"
        handoff = self._active_sessions.get(key)

        if not handoff:
            logger.warning(f"No active session found for {key}")
            return None

        if action:
            handoff.last_action = action
        if outcome:
            handoff.last_outcome = outcome
        if decision:
            handoff.add_decision(decision)
        if blocker:
            handoff.add_blocker(blocker)
        if resolved_blocker:
            handoff.remove_blocker(resolved_blocker)
        if active_file and active_file not in handoff.active_files:
            handoff.active_files.append(active_file)
        if test_result:
            for test_name, passing in test_result.items():
                handoff.set_test_status(test_name, passing)
        if confidence is not None:
            handoff.confidence_level = max(0.0, min(1.0, confidence))
        if risk and risk not in handoff.risk_flags:
            handoff.risk_flags.append(risk)

        return handoff

    def create_handoff(
        self,
        agent: str,
        session_id: str,
        last_action: str,
        last_outcome: SessionOutcome,
        next_steps: Optional[List[str]] = None,
        **context,
    ) -> SessionHandoff:
        """
        Create handoff at session end.

        This finalizes the session and stores the handoff for the next session.

        Args:
            agent: Agent identifier
            session_id: Session identifier
            last_action: Final action taken
            last_outcome: Outcome of the session
            next_steps: Planned next actions
            **context: Additional context to store

        Returns:
            Finalized SessionHandoff
        """
        key = f"{agent}:{session_id}"
        handoff = self._active_sessions.get(key)

        if handoff:
            # Finalize existing handoff
            handoff.finalize(last_action, last_outcome, next_steps)
            # Add any additional context
            handoff.metadata.update(context)
        else:
            # Create new handoff (session started without start_session call)
            handoff = SessionHandoff.create(
                project_id=self.project_id,
                agent=agent,
                session_id=session_id,
                last_action=last_action,
                current_goal=context.get("goal", "Unknown"),
                last_outcome=last_outcome,
                next_steps=next_steps or [],
            )
            handoff.session_end = datetime.now(timezone.utc)
            handoff.metadata.update(context)

        # Store handoff
        self._store_handoff(agent, handoff)

        # Clear active session
        if key in self._active_sessions:
            del self._active_sessions[key]

        logger.info(
            f"Created handoff for session {session_id}, "
            f"outcome: {last_outcome}, next_steps: {len(next_steps or [])}"
        )

        return handoff

    def _store_handoff(self, agent: str, handoff: SessionHandoff) -> None:
        """Store a handoff internally and optionally to persistent storage."""
        if agent not in self._handoffs:
            self._handoffs[agent] = []

        self._handoffs[agent].append(handoff)

        # Trim to max
        if len(self._handoffs[agent]) > self.max_handoffs:
            self._handoffs[agent] = self._handoffs[agent][-self.max_handoffs:]

        # TODO: Persist to storage backend when integrated

    def get_latest_handoff(self, agent: str) -> Optional[SessionHandoff]:
        """Get the most recent handoff for an agent."""
        handoffs = self._handoffs.get(agent, [])
        return handoffs[-1] if handoffs else None

    def get_previous_sessions(
        self,
        agent: str,
        limit: int = 5,
    ) -> List[SessionHandoff]:
        """
        Get recent session handoffs for an agent.

        Args:
            agent: Agent identifier
            limit: Maximum number of handoffs to return

        Returns:
            List of SessionHandoff, most recent first
        """
        handoffs = self._handoffs.get(agent, [])
        return list(reversed(handoffs[-limit:]))

    def get_quick_reload(
        self,
        agent: str,
    ) -> str:
        """
        Get compressed context string for quick reload.

        This is a formatted string that can be quickly parsed by an agent
        for rapid context restoration.

        Args:
            agent: Agent identifier

        Returns:
            Formatted quick reload string
        """
        handoff = self.get_latest_handoff(agent)
        if not handoff:
            return f"No previous session found for agent {agent}"

        return handoff.format_quick_reload()

    def get_all_agents(self) -> List[str]:
        """Get list of all agents with session history."""
        return list(self._handoffs.keys())

    def get_agent_stats(self, agent: str) -> Dict[str, Any]:
        """
        Get session statistics for an agent.

        Returns summary of session history including:
        - Total sessions
        - Success rate
        - Average duration
        - Common blockers
        """
        handoffs = self._handoffs.get(agent, [])
        if not handoffs:
            return {
                "agent": agent,
                "total_sessions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0,
                "common_blockers": [],
            }

        # Calculate stats
        total = len(handoffs)
        successes = sum(1 for h in handoffs if h.last_outcome == "success")
        success_rate = successes / total if total > 0 else 0.0

        durations = [h.duration_ms for h in handoffs if h.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Count blockers
        blocker_counts: Dict[str, int] = {}
        for h in handoffs:
            for blocker in h.blockers:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        common_blockers = sorted(
            blocker_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "agent": agent,
            "total_sessions": total,
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
            "common_blockers": [b[0] for b in common_blockers],
        }

    def clear_history(self, agent: Optional[str] = None) -> int:
        """
        Clear session history.

        Args:
            agent: If provided, only clear history for this agent

        Returns:
            Number of handoffs cleared
        """
        if agent:
            count = len(self._handoffs.get(agent, []))
            self._handoffs[agent] = []
            return count
        else:
            count = sum(len(h) for h in self._handoffs.values())
            self._handoffs.clear()
            return count
