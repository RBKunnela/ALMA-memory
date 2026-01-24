"""
Tests for the Session Management module.
"""

import pytest
from datetime import datetime, timezone

from alma.session import (
    SessionHandoff,
    SessionContext,
    SessionOutcome,
    SessionManager,
)


class TestSessionHandoff:
    """Tests for SessionHandoff dataclass."""

    def test_create_basic(self):
        """Test basic handoff creation."""
        handoff = SessionHandoff.create(
            project_id="proj-1",
            agent="Helena",
            session_id="session-123",
            last_action="test_login",
            current_goal="Test authentication flow",
        )

        assert handoff.id is not None
        assert handoff.project_id == "proj-1"
        assert handoff.agent == "Helena"
        assert handoff.session_id == "session-123"
        assert handoff.last_action == "test_login"
        assert handoff.current_goal == "Test authentication flow"
        assert handoff.last_outcome == "unknown"

    def test_finalize(self):
        """Test finalizing a handoff."""
        handoff = SessionHandoff.create(
            project_id="proj-1",
            agent="Helena",
            session_id="session-123",
            last_action="started",
            current_goal="Testing",
        )

        handoff.finalize(
            last_action="completed_tests",
            last_outcome="success",
            next_steps=["Review results", "Create report"],
        )

        assert handoff.last_action == "completed_tests"
        assert handoff.last_outcome == "success"
        assert handoff.session_end is not None
        assert len(handoff.next_steps) == 2

    def test_add_decision(self):
        """Test adding decisions."""
        handoff = SessionHandoff.create("proj", "agent", "sess", "action", "goal")

        for i in range(15):
            handoff.add_decision(f"Decision {i}")

        # Should cap at 10
        assert len(handoff.key_decisions) == 10
        # Latest should be last
        assert handoff.key_decisions[-1] == "Decision 14"

    def test_blockers(self):
        """Test blocker management."""
        handoff = SessionHandoff.create("proj", "agent", "sess", "action", "goal")

        handoff.add_blocker("API unavailable")
        handoff.add_blocker("Missing credentials")

        assert len(handoff.blockers) == 2

        handoff.remove_blocker("API unavailable")
        assert len(handoff.blockers) == 1
        assert "Missing credentials" in handoff.blockers

    def test_format_quick_reload(self):
        """Test formatting for quick reload."""
        handoff = SessionHandoff.create(
            project_id="proj-1",
            agent="Helena",
            session_id="session-123",
            last_action="test_login",
            current_goal="Test authentication",
        )
        handoff.add_decision("Use OAuth for testing")
        handoff.add_blocker("API rate limit")
        handoff.next_steps = ["Wait for rate limit", "Retry tests"]
        handoff.confidence_level = 0.7

        formatted = handoff.format_quick_reload()

        assert "Session Handoff" in formatted
        assert "Helena" in formatted
        assert "Test authentication" in formatted
        assert "70%" in formatted
        assert "Use OAuth for testing" in formatted
        assert "API rate limit" in formatted


class TestSessionContext:
    """Tests for SessionContext dataclass."""

    def test_create_basic(self):
        """Test basic context creation."""
        context = SessionContext.create(
            project_id="proj-1",
            agent="Helena",
        )

        assert context.project_id == "proj-1"
        assert context.agent == "Helena"
        assert context.session_id is not None
        assert context.previous_handoff is None

    def test_create_with_handoff(self):
        """Test context with previous handoff."""
        previous = SessionHandoff.create("proj", "Helena", "prev", "action", "goal")
        previous.add_blocker("Known issue")

        context = SessionContext.create(
            project_id="proj",
            agent="Helena",
            previous_handoff=previous,
        )

        assert context.previous_handoff is not None
        assert len(context.previous_handoff.blockers) == 1

    def test_format_orientation(self):
        """Test formatting orientation briefing."""
        previous = SessionHandoff.create("proj", "Helena", "prev", "test_ui", "UI Testing")
        previous.next_steps = ["Complete form tests"]

        context = SessionContext.create(
            project_id="proj",
            agent="Helena",
            previous_handoff=previous,
        )

        formatted = context.format_orientation()

        assert "Session Orientation" in formatted
        assert "Helena" in formatted
        assert "Previous Session" in formatted
        assert "UI Testing" in formatted


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh session manager."""
        return SessionManager(project_id="test-project")

    def test_start_session_new(self, manager):
        """Test starting a new session without previous."""
        context = manager.start_session(agent="Helena", goal="Test login")

        assert context.session_id is not None
        assert context.agent == "Helena"
        assert context.previous_handoff is None

    def test_start_session_with_previous(self, manager):
        """Test starting session with previous handoff."""
        # Create first session
        ctx1 = manager.start_session(agent="Helena", goal="First goal")

        # End first session
        manager.create_handoff(
            agent="Helena",
            session_id=ctx1.session_id,
            last_action="completed",
            last_outcome="success",
            next_steps=["Continue testing"],
        )

        # Start second session
        ctx2 = manager.start_session(agent="Helena", goal="Second goal")

        assert ctx2.previous_handoff is not None
        assert ctx2.previous_handoff.last_outcome == "success"

    def test_update_session(self, manager):
        """Test updating active session."""
        ctx = manager.start_session(agent="Helena")

        handoff = manager.update_session(
            agent="Helena",
            session_id=ctx.session_id,
            action="testing_form",
            decision="Use Playwright",
            confidence=0.8,
        )

        assert handoff is not None
        assert handoff.last_action == "testing_form"
        assert "Use Playwright" in handoff.key_decisions
        assert handoff.confidence_level == 0.8

    def test_create_handoff(self, manager):
        """Test creating handoff at session end."""
        ctx = manager.start_session(agent="Helena", goal="Testing")

        handoff = manager.create_handoff(
            agent="Helena",
            session_id=ctx.session_id,
            last_action="finished_tests",
            last_outcome="success",
            next_steps=["Review report"],
        )

        assert handoff.last_action == "finished_tests"
        assert handoff.last_outcome == "success"
        assert handoff.session_end is not None

    def test_get_quick_reload(self, manager):
        """Test getting quick reload string."""
        ctx = manager.start_session(agent="Helena", goal="Testing")
        manager.create_handoff(
            agent="Helena",
            session_id=ctx.session_id,
            last_action="completed",
            last_outcome="success",
        )

        reload_str = manager.get_quick_reload("Helena")

        assert "Session Handoff" in reload_str
        assert "Helena" in reload_str
        assert "success" in reload_str

    def test_get_previous_sessions(self, manager):
        """Test getting session history."""
        # Create multiple sessions
        for i in range(3):
            ctx = manager.start_session(agent="Helena", goal=f"Goal {i}")
            manager.create_handoff(
                agent="Helena",
                session_id=ctx.session_id,
                last_action=f"action_{i}",
                last_outcome="success",
            )

        history = manager.get_previous_sessions("Helena", limit=2)

        assert len(history) == 2
        # Most recent first
        assert history[0].last_action == "action_2"

    def test_agent_stats(self, manager):
        """Test getting agent statistics."""
        # Create sessions with mixed outcomes
        for outcome in ["success", "success", "failure"]:
            ctx = manager.start_session(agent="Helena")
            manager.create_handoff(
                agent="Helena",
                session_id=ctx.session_id,
                last_action="action",
                last_outcome=outcome,
            )

        stats = manager.get_agent_stats("Helena")

        assert stats["total_sessions"] == 3
        assert stats["success_rate"] == pytest.approx(2 / 3)

    def test_multi_agent(self, manager):
        """Test manager with multiple agents."""
        # Create sessions for different agents
        ctx_h = manager.start_session(agent="Helena")
        ctx_v = manager.start_session(agent="Victor")

        manager.create_handoff("Helena", ctx_h.session_id, "done", "success")
        manager.create_handoff("Victor", ctx_v.session_id, "done", "success")

        agents = manager.get_all_agents()
        assert "Helena" in agents
        assert "Victor" in agents

        # Quick reload should be agent-specific
        reload_h = manager.get_quick_reload("Helena")
        reload_v = manager.get_quick_reload("Victor")
        assert reload_h != reload_v

    def test_blocker_carryover(self, manager):
        """Test that blockers carry over to new sessions."""
        ctx1 = manager.start_session(agent="Helena")
        manager.update_session("Helena", ctx1.session_id, blocker="API down")
        manager.create_handoff("Helena", ctx1.session_id, "blocked", "interrupted")

        ctx2 = manager.start_session(agent="Helena")
        active = manager.get_active_handoff("Helena", ctx2.session_id)

        assert "API down" in active.blockers
