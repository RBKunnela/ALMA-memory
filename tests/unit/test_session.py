"""
Tests for the Session Management module.
"""

import pytest

from alma.session import (
    SessionContext,
    SessionHandoff,
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
        previous = SessionHandoff.create(
            "proj", "Helena", "prev", "test_ui", "UI Testing"
        )
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


class TestSessionPersistence:
    """Tests for session persistence with storage backend."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary SQLite database."""
        from alma.storage.sqlite_local import SQLiteStorage

        db_path = tmp_path / "test_sessions.db"
        return SQLiteStorage(db_path=db_path)

    @pytest.fixture
    def persistent_manager(self, temp_db):
        """Create a session manager with storage backend."""
        return SessionManager(
            project_id="test-project",
            storage=temp_db,
        )

    def test_handoff_persisted_to_storage(self, persistent_manager, temp_db):
        """Test that handoffs are saved to storage."""
        ctx = persistent_manager.start_session(agent="Helena", goal="Testing")
        persistent_manager.create_handoff(
            agent="Helena",
            session_id=ctx.session_id,
            last_action="finished",
            last_outcome="success",
        )

        # Verify it's in storage
        handoffs = temp_db.get_session_handoffs("test-project", "Helena")
        assert len(handoffs) == 1
        assert handoffs[0].last_action == "finished"
        assert handoffs[0].last_outcome == "success"

    def test_handoffs_loaded_from_storage(self, temp_db):
        """Test that handoffs are loaded from storage on new manager instance."""
        # Create manager and add a handoff
        manager1 = SessionManager(project_id="test-project", storage=temp_db)
        ctx = manager1.start_session(agent="Helena", goal="Testing")
        manager1.create_handoff(
            agent="Helena",
            session_id=ctx.session_id,
            last_action="finished",
            last_outcome="success",
            next_steps=["Review results"],
        )

        # Create a new manager instance (simulating process restart)
        manager2 = SessionManager(project_id="test-project", storage=temp_db)

        # Should load from storage
        latest = manager2.get_latest_handoff("Helena")
        assert latest is not None
        assert latest.last_action == "finished"
        assert latest.last_outcome == "success"
        assert "Review results" in latest.next_steps

    def test_get_previous_sessions_from_storage(self, temp_db):
        """Test that get_previous_sessions loads from storage."""
        # Create manager and add multiple handoffs
        manager1 = SessionManager(project_id="test-project", storage=temp_db)
        for i in range(3):
            ctx = manager1.start_session(agent="Helena", goal=f"Goal {i}")
            manager1.create_handoff(
                agent="Helena",
                session_id=ctx.session_id,
                last_action=f"action_{i}",
                last_outcome="success",
            )

        # Create new manager
        manager2 = SessionManager(project_id="test-project", storage=temp_db)

        # Should load from storage
        history = manager2.get_previous_sessions("Helena", limit=2)
        assert len(history) == 2
        # Most recent first
        assert history[0].last_action == "action_2"

    def test_clear_history_clears_storage(self, persistent_manager, temp_db):
        """Test that clear_history also clears from storage."""
        # Add handoffs
        for i in range(3):
            ctx = persistent_manager.start_session(agent="Helena")
            persistent_manager.create_handoff(
                agent="Helena",
                session_id=ctx.session_id,
                last_action=f"action_{i}",
                last_outcome="success",
            )

        # Clear history
        count = persistent_manager.clear_history(agent="Helena")
        assert count == 3

        # Verify storage is also cleared
        handoffs = temp_db.get_session_handoffs("test-project", "Helena")
        assert len(handoffs) == 0

    def test_clear_all_history_clears_storage(self, persistent_manager, temp_db):
        """Test that clear_history without agent clears all from storage."""
        # Add handoffs for different agents
        for agent in ["Helena", "Victor"]:
            ctx = persistent_manager.start_session(agent=agent)
            persistent_manager.create_handoff(
                agent=agent,
                session_id=ctx.session_id,
                last_action="done",
                last_outcome="success",
            )

        # Clear all history
        persistent_manager.clear_history()

        # Verify storage is cleared for all agents
        assert len(temp_db.get_session_handoffs("test-project", "Helena")) == 0
        assert len(temp_db.get_session_handoffs("test-project", "Victor")) == 0

    def test_backward_compatibility_without_storage(self):
        """Test that manager works without storage (backward compatibility)."""
        manager = SessionManager(project_id="test-project", storage=None)

        ctx = manager.start_session(agent="Helena", goal="Testing")
        manager.create_handoff(
            agent="Helena",
            session_id=ctx.session_id,
            last_action="finished",
            last_outcome="success",
        )

        # Should work with in-memory only
        latest = manager.get_latest_handoff("Helena")
        assert latest is not None
        assert latest.last_action == "finished"

    def test_handoff_serialization_roundtrip(self, temp_db):
        """Test that all handoff fields survive storage roundtrip."""
        handoff = SessionHandoff.create(
            project_id="test-project",
            agent="Helena",
            session_id="sess-1",
            last_action="testing",
            current_goal="Test goal",
            last_outcome="success",
        )
        handoff.key_decisions = ["Decision 1", "Decision 2"]
        handoff.active_files = ["file1.py", "file2.py"]
        handoff.blockers = ["Blocker 1"]
        handoff.next_steps = ["Step 1", "Step 2"]
        handoff.test_status = {"test1": True, "test2": False}
        handoff.confidence_level = 0.75
        handoff.risk_flags = ["Risk 1"]
        handoff.metadata = {"key": "value"}

        # Save and retrieve
        temp_db.save_session_handoff(handoff)
        retrieved = temp_db.get_latest_session_handoff("test-project", "Helena")

        assert retrieved is not None
        assert retrieved.id == handoff.id
        assert retrieved.key_decisions == ["Decision 1", "Decision 2"]
        assert retrieved.active_files == ["file1.py", "file2.py"]
        assert retrieved.blockers == ["Blocker 1"]
        assert retrieved.next_steps == ["Step 1", "Step 2"]
        assert retrieved.test_status == {"test1": True, "test2": False}
        assert retrieved.confidence_level == 0.75
        assert retrieved.risk_flags == ["Risk 1"]
        assert retrieved.metadata == {"key": "value"}
