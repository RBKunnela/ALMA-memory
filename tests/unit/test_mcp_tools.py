"""
Unit tests for ALMA MCP tools.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from alma import ALMA, MemorySlice
from alma.exceptions import ScopeViolationError
from alma.mcp.tools import (
    alma_add_knowledge,
    alma_add_preference,
    alma_compress_and_learn,
    alma_extract_heuristic,
    alma_forget,
    alma_health,
    alma_learn,
    alma_reinforce,
    alma_retrieve,
    alma_retrieve_for_mode,
    alma_retrieve_smart,
    alma_retrieve_verified,
    alma_smart_forget,
    alma_stats,
)
from alma.retrieval.modes import RetrievalMode
from alma.types import DomainKnowledge, Heuristic, Outcome, UserPreference


class TestAlmaRetrieve:
    """Tests for alma_retrieve tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.project_id = "test-project"
        now = datetime.now(timezone.utc)

        # Mock retrieve to return a populated MemorySlice
        alma.retrieve.return_value = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="helena",
                    project_id="test-project",
                    condition="form testing",
                    strategy="validate inputs first",
                    confidence=0.85,
                    occurrence_count=10,
                    success_count=9,
                    last_validated=now,
                    created_at=now,
                ),
            ],
            domain_knowledge=[
                DomainKnowledge(
                    id="dk1",
                    agent="helena",
                    project_id="test-project",
                    domain="testing",
                    fact="Use data-testid",
                    source="test",
                    last_verified=now,
                ),
            ],
            preferences=[],
            anti_patterns=[],
            outcomes=[],
            query="test query",
            agent="helena",
            retrieval_time_ms=10,
        )

        return alma

    def test_retrieve_success(self, mock_alma):
        """Test successful retrieval."""
        result = alma_retrieve(
            alma=mock_alma,
            task="Test login form",
            agent="helena",
            top_k=5,
        )

        assert result["success"] is True
        assert "memories" in result
        assert "prompt_injection" in result
        assert len(result["memories"]["heuristics"]) == 1
        mock_alma.retrieve.assert_called_once()

    def test_retrieve_with_user_id(self, mock_alma):
        """Test retrieval with user ID."""
        result = alma_retrieve(
            alma=mock_alma,
            task="Test form",
            agent="helena",
            user_id="user-123",
            top_k=10,
        )

        assert result["success"] is True
        mock_alma.retrieve.assert_called_with(
            task="Test form",
            agent="helena",
            user_id="user-123",
            top_k=10,
        )

    def test_retrieve_error_handling(self, mock_alma):
        """Test error handling in retrieval."""
        mock_alma.retrieve.side_effect = Exception("Database error")

        result = alma_retrieve(
            alma=mock_alma,
            task="Test form",
            agent="helena",
        )

        assert result["success"] is False
        assert "error" in result
        assert "Database error" in result["error"]


class TestAlmaLearn:
    """Tests for alma_learn tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with Outcome return value."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        # learn() now returns an Outcome object
        alma.learn.return_value = Outcome(
            id="out-test-1",
            agent="helena",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test login form",
            success=True,
            strategy_used="validate inputs first",
            timestamp=now,
        )
        return alma

    def test_learn_success(self, mock_alma):
        """Test successful learning."""
        result = alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Test login form",
            outcome="success",
            strategy_used="validate inputs first",
            task_type="form_testing",
        )

        assert result["success"] is True
        assert result["learned"] is True
        assert "outcome" in result
        assert result["outcome"]["id"] == "out-test-1"
        mock_alma.learn.assert_called_once()

    def test_learn_failure_outcome(self, mock_alma):
        """Test learning with failure outcome."""
        now = datetime.now(timezone.utc)
        # Update mock to return failure outcome
        mock_alma.learn.return_value = Outcome(
            id="out-test-2",
            agent="helena",
            project_id="test-project",
            task_type="general",
            task_description="Test modal",
            success=False,
            strategy_used="click without wait",
            error_message="Element not found",
            timestamp=now,
        )

        result = alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Test modal",
            outcome="failure",
            strategy_used="click without wait",
            error_message="Element not found",
        )

        assert result["success"] is True
        assert result["outcome"]["success"] is False
        mock_alma.learn.assert_called_with(
            agent="helena",
            task="Test modal",
            outcome="failure",
            strategy_used="click without wait",
            task_type=None,
            duration_ms=None,
            error_message="Element not found",
            feedback=None,
        )

    def test_learn_scope_rejection(self, mock_alma):
        """Test learning rejected due to scope - now raises ScopeViolationError."""
        mock_alma.learn.side_effect = ScopeViolationError(
            "Agent 'helena' is not allowed in this scope"
        )

        result = alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Backend task",
            outcome="success",
            strategy_used="some strategy",
        )

        assert result["success"] is False
        assert "error" in result
        assert (
            "scope" in result["error"].lower()
            or "not allowed" in result["error"].lower()
        )


class TestAlmaAddPreference:
    """Tests for alma_add_preference tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        alma.add_user_preference.return_value = UserPreference(
            id="pref-1",
            user_id="user-123",
            category="code_style",
            preference="No emojis",
            source="explicit_instruction",
            timestamp=now,
        )

        return alma

    def test_add_preference_success(self, mock_alma):
        """Test successful preference addition."""
        result = alma_add_preference(
            alma=mock_alma,
            user_id="user-123",
            category="code_style",
            preference="No emojis",
        )

        assert result["success"] is True
        assert result["preference"]["id"] == "pref-1"
        assert result["preference"]["preference"] == "No emojis"

    def test_add_preference_custom_source(self, mock_alma):
        """Test preference with custom source."""
        result = alma_add_preference(
            alma=mock_alma,
            user_id="user-123",
            category="workflow",
            preference="Always run tests",
            source="inferred",
        )

        assert result["success"] is True
        mock_alma.add_user_preference.assert_called_with(
            user_id="user-123",
            category="workflow",
            preference="Always run tests",
            source="inferred",
        )


class TestAlmaAddKnowledge:
    """Tests for alma_add_knowledge tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        alma.add_domain_knowledge.return_value = DomainKnowledge(
            id="dk-1",
            agent="helena",
            project_id="test-project",
            domain="testing",
            fact="Use data-testid",
            source="user_stated",
            last_verified=now,
        )

        return alma

    def test_add_knowledge_success(self, mock_alma):
        """Test successful knowledge addition."""
        result = alma_add_knowledge(
            alma=mock_alma,
            agent="helena",
            domain="testing",
            fact="Use data-testid",
        )

        assert result["success"] is True
        assert result["knowledge"]["id"] == "dk-1"
        assert result["knowledge"]["fact"] == "Use data-testid"

    def test_add_knowledge_scope_rejection(self, mock_alma):
        """Test knowledge rejected due to scope - now raises ScopeViolationError."""
        mock_alma.add_domain_knowledge.side_effect = ScopeViolationError(
            "Agent 'helena' is not allowed to learn in domain 'backend'"
        )

        result = alma_add_knowledge(
            alma=mock_alma,
            agent="helena",
            domain="backend",
            fact="Some backend fact",
        )

        assert result["success"] is False
        assert "not allowed" in result["error"]


class TestAlmaForget:
    """Tests for alma_forget tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.forget.return_value = 5
        return alma

    def test_forget_success(self, mock_alma):
        """Test successful forget operation."""
        result = alma_forget(
            alma=mock_alma,
            agent="helena",
            older_than_days=30,
            below_confidence=0.5,
        )

        assert result["success"] is True
        assert result["pruned_count"] == 5

    def test_forget_all_agents(self, mock_alma):
        """Test forget for all agents."""
        result = alma_forget(
            alma=mock_alma,
            older_than_days=90,
            below_confidence=0.3,
        )

        assert result["success"] is True
        mock_alma.forget.assert_called_with(
            agent=None,
            older_than_days=90,
            below_confidence=0.3,
        )


class TestAlmaStats:
    """Tests for alma_stats tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.get_stats.return_value = {
            "project_id": "test-project",
            "heuristics_count": 10,
            "outcomes_count": 50,
            "domain_knowledge_count": 15,
            "total_count": 75,
        }
        return alma

    def test_stats_success(self, mock_alma):
        """Test successful stats retrieval."""
        result = alma_stats(alma=mock_alma, agent="helena")

        assert result["success"] is True
        assert result["stats"]["total_count"] == 75

    def test_stats_all_agents(self, mock_alma):
        """Test stats for all agents."""
        result = alma_stats(alma=mock_alma)

        assert result["success"] is True
        mock_alma.get_stats.assert_called_with(agent=None)


class TestAlmaHealth:
    """Tests for alma_health tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.project_id = "test-project"
        alma.scopes = {"helena": MagicMock(), "victor": MagicMock()}
        alma.get_stats.return_value = {"total_count": 100}
        return alma

    def test_health_success(self, mock_alma):
        """Test successful health check."""
        result = alma_health(alma=mock_alma)

        assert result["success"] is True
        assert result["status"] == "healthy"
        assert result["project_id"] == "test-project"
        assert "helena" in result["registered_agents"]
        assert "victor" in result["registered_agents"]

    def test_health_includes_timestamp(self, mock_alma):
        """Test that health check includes timestamp."""
        result = alma_health(alma=mock_alma)

        assert "timestamp" in result
        # Should be ISO format
        datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))

    def test_health_error_handling(self, mock_alma):
        """Test health check error handling."""
        mock_alma.get_stats.side_effect = Exception("Connection failed")

        result = alma_health(alma=mock_alma)

        assert result["success"] is False
        assert result["status"] == "unhealthy"


class TestAlmaRetrieveForMode:
    """Tests for alma_retrieve_for_mode tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with retrieval engine."""
        alma = MagicMock()  # Don't use spec to allow nested attributes
        alma.project_id = "test-project"
        alma.scopes = {"helena": MagicMock()}
        now = datetime.now(timezone.utc)

        # Create a mock MemorySlice
        memory_slice = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="helena",
                    project_id="test-project",
                    condition="debugging",
                    strategy="check logs first",
                    confidence=0.9,
                    occurrence_count=5,
                    success_count=4,
                    last_validated=now,
                    created_at=now,
                ),
            ],
            domain_knowledge=[],
            preferences=[],
            anti_patterns=[],
            outcomes=[],
            query="test query",
            agent="helena",
            retrieval_time_ms=15,
        )

        # Mock retrieval engine
        alma.retrieval = MagicMock()
        alma.retrieval.retrieve_with_mode.return_value = (
            memory_slice,
            RetrievalMode.DIAGNOSTIC,
            "Query contains diagnostic terms: error",
        )

        return alma

    def test_retrieve_for_mode_success(self, mock_alma):
        """Test successful mode-aware retrieval."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Why is the login failing?",
            mode="diagnostic",
            agent="helena",
        )

        assert result["success"] is True
        assert "memories" in result
        assert "prompt_injection" in result
        assert result["mode"] == "diagnostic"
        assert "diagnostic" in result["mode_reason"].lower()
        mock_alma.retrieval.retrieve_with_mode.assert_called_once()

    def test_retrieve_for_mode_with_top_k(self, mock_alma):
        """Test retrieval with custom top_k."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Debug the API",
            mode="diagnostic",
            agent="helena",
            top_k=10,
        )

        assert result["success"] is True
        mock_alma.retrieval.retrieve_with_mode.assert_called_with(
            query="Debug the API",
            agent="helena",
            project_id="test-project",
            mode=RetrievalMode.DIAGNOSTIC,
            user_id=None,
            top_k=10,
            scope=mock_alma.scopes.get("helena"),
        )

    def test_retrieve_for_mode_all_modes(self, mock_alma):
        """Test all valid retrieval modes."""
        valid_modes = ["broad", "precise", "diagnostic", "learning", "recall"]

        for mode in valid_modes:
            mock_alma.retrieval.retrieve_with_mode.return_value = (
                mock_alma.retrieval.retrieve_with_mode.return_value[0],
                RetrievalMode(mode),
                f"Mode: {mode}",
            )
            result = alma_retrieve_for_mode(
                alma=mock_alma,
                query="Test query",
                mode=mode,
                agent="helena",
            )
            assert result["success"] is True
            assert result["mode"] == mode

    def test_retrieve_for_mode_invalid_mode(self, mock_alma):
        """Test error handling for invalid mode."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Test query",
            mode="invalid_mode",
            agent="helena",
        )

        assert result["success"] is False
        assert "mode must be one of" in result["error"]

    def test_retrieve_for_mode_empty_query(self, mock_alma):
        """Test error handling for empty query."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="",
            mode="precise",
            agent="helena",
        )

        assert result["success"] is False
        assert "query cannot be empty" in result["error"]

    def test_retrieve_for_mode_empty_agent(self, mock_alma):
        """Test error handling for empty agent."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Test query",
            mode="precise",
            agent="",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]

    def test_retrieve_for_mode_empty_mode(self, mock_alma):
        """Test error handling for empty mode."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Test query",
            mode="",
            agent="helena",
        )

        assert result["success"] is False
        assert "mode cannot be empty" in result["error"]

    def test_retrieve_for_mode_case_insensitive(self, mock_alma):
        """Test that mode is case-insensitive."""
        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Test query",
            mode="DIAGNOSTIC",
            agent="helena",
        )

        assert result["success"] is True
        mock_alma.retrieval.retrieve_with_mode.assert_called()

    def test_retrieve_for_mode_error_handling(self, mock_alma):
        """Test error handling in retrieval."""
        mock_alma.retrieval.retrieve_with_mode.side_effect = Exception("Engine error")

        result = alma_retrieve_for_mode(
            alma=mock_alma,
            query="Test query",
            mode="diagnostic",
            agent="helena",
        )

        assert result["success"] is False
        assert "Engine error" in result["error"]


class TestAlmaRetrieveSmart:
    """Tests for alma_retrieve_smart tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with retrieval engine."""
        alma = MagicMock()  # Don't use spec to allow nested attributes
        alma.project_id = "test-project"
        alma.scopes = {"helena": MagicMock()}
        now = datetime.now(timezone.utc)

        # Create a mock MemorySlice
        memory_slice = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="helena",
                    project_id="test-project",
                    condition="planning",
                    strategy="explore options",
                    confidence=0.8,
                    occurrence_count=3,
                    success_count=3,
                    last_validated=now,
                    created_at=now,
                ),
            ],
            domain_knowledge=[],
            preferences=[],
            anti_patterns=[],
            outcomes=[],
            query="test query",
            agent="helena",
            retrieval_time_ms=12,
        )

        # Mock retrieval engine
        alma.retrieval = MagicMock()
        alma.retrieval.retrieve_with_mode.return_value = (
            memory_slice,
            RetrievalMode.BROAD,
            "Query contains planning/exploration terms: options",
        )

        return alma

    def test_retrieve_smart_success(self, mock_alma):
        """Test successful smart retrieval with auto-inferred mode."""
        result = alma_retrieve_smart(
            alma=mock_alma,
            query="What are our options for authentication?",
            agent="helena",
        )

        assert result["success"] is True
        assert "memories" in result
        assert "prompt_injection" in result
        assert "mode" in result
        assert "mode_reason" in result
        mock_alma.retrieval.retrieve_with_mode.assert_called_once()

    def test_retrieve_smart_infers_diagnostic(self, mock_alma):
        """Test smart retrieval infers DIAGNOSTIC mode."""
        mock_alma.retrieval.retrieve_with_mode.return_value = (
            mock_alma.retrieval.retrieve_with_mode.return_value[0],
            RetrievalMode.DIAGNOSTIC,
            "Query contains diagnostic terms: error",
        )

        result = alma_retrieve_smart(
            alma=mock_alma,
            query="Why is the system throwing an error?",
            agent="helena",
        )

        assert result["success"] is True
        # Verify mode=None was passed for auto-inference
        call_kwargs = mock_alma.retrieval.retrieve_with_mode.call_args.kwargs
        assert call_kwargs["mode"] is None

    def test_retrieve_smart_with_user_id(self, mock_alma):
        """Test smart retrieval with user ID."""
        result = alma_retrieve_smart(
            alma=mock_alma,
            query="How should we design the API?",
            agent="helena",
            user_id="user-123",
        )

        assert result["success"] is True
        call_kwargs = mock_alma.retrieval.retrieve_with_mode.call_args.kwargs
        assert call_kwargs["user_id"] == "user-123"

    def test_retrieve_smart_with_top_k(self, mock_alma):
        """Test smart retrieval with custom top_k."""
        result = alma_retrieve_smart(
            alma=mock_alma,
            query="Find similar patterns",
            agent="helena",
            top_k=20,
        )

        assert result["success"] is True
        call_kwargs = mock_alma.retrieval.retrieve_with_mode.call_args.kwargs
        assert call_kwargs["top_k"] == 20

    def test_retrieve_smart_empty_query(self, mock_alma):
        """Test error handling for empty query."""
        result = alma_retrieve_smart(
            alma=mock_alma,
            query="",
            agent="helena",
        )

        assert result["success"] is False
        assert "query cannot be empty" in result["error"]

    def test_retrieve_smart_empty_agent(self, mock_alma):
        """Test error handling for empty agent."""
        result = alma_retrieve_smart(
            alma=mock_alma,
            query="Test query",
            agent="",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]

    def test_retrieve_smart_error_handling(self, mock_alma):
        """Test error handling in retrieval."""
        mock_alma.retrieval.retrieve_with_mode.side_effect = Exception("Storage error")

        result = alma_retrieve_smart(
            alma=mock_alma,
            query="Test query",
            agent="helena",
        )

        assert result["success"] is False
        assert "Storage error" in result["error"]

    def test_retrieve_smart_returns_mode_reason(self, mock_alma):
        """Test that mode reason is returned."""
        mock_alma.retrieval.retrieve_with_mode.return_value = (
            mock_alma.retrieval.retrieve_with_mode.return_value[0],
            RetrievalMode.RECALL,
            "Query contains recall terms: what was, previously",
        )

        result = alma_retrieve_smart(
            alma=mock_alma,
            query="What was the previous approach?",
            agent="helena",
        )

        assert result["success"] is True
        assert result["mode"] == "recall"
        assert "recall" in result["mode_reason"].lower()


# =============================================================================
# MEMORY WALL ENHANCEMENT TOOLS TESTS (v0.7.0)
# =============================================================================


class TestAlmaReinforce:
    """Tests for alma_reinforce tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with storage."""
        alma = MagicMock()
        alma.storage = MagicMock()
        return alma

    def test_reinforce_empty_memory_id(self, mock_alma):
        """Test error for empty memory_id."""
        result = alma_reinforce(alma=mock_alma, memory_id="")

        assert result["success"] is False
        assert "memory_id cannot be empty" in result["error"]

    def test_reinforce_whitespace_memory_id(self, mock_alma):
        """Test error for whitespace-only memory_id."""
        result = alma_reinforce(alma=mock_alma, memory_id="   ")

        assert result["success"] is False
        assert "memory_id cannot be empty" in result["error"]


class TestAlmaGetWeakMemories:
    """Tests for alma_get_weak_memories tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.storage = MagicMock()
        return alma

    # No empty input validation needed - all params are optional


class TestAlmaSmartForget:
    """Tests for alma_smart_forget tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.storage = MagicMock()
        return alma

    def test_smart_forget_invalid_threshold_high(self, mock_alma):
        """Test invalid threshold (too high)."""
        result = alma_smart_forget(alma=mock_alma, threshold=1.5)

        assert result["success"] is False
        assert "threshold must be between" in result["error"]

    def test_smart_forget_invalid_threshold_negative(self, mock_alma):
        """Test invalid threshold (negative)."""
        result = alma_smart_forget(alma=mock_alma, threshold=-0.5)

        assert result["success"] is False
        assert "threshold must be between" in result["error"]


class TestAlmaRetrieveVerified:
    """Tests for alma_retrieve_verified tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.retrieval = MagicMock()
        alma.llm = None
        return alma

    def test_retrieve_verified_empty_query(self, mock_alma):
        """Test error for empty query."""
        result = alma_retrieve_verified(alma=mock_alma, query="", agent="helena")

        assert result["success"] is False
        assert "query cannot be empty" in result["error"]

    def test_retrieve_verified_empty_agent(self, mock_alma):
        """Test error for empty agent."""
        result = alma_retrieve_verified(alma=mock_alma, query="test", agent="")

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]

    def test_retrieve_verified_whitespace_query(self, mock_alma):
        """Test error for whitespace-only query."""
        result = alma_retrieve_verified(alma=mock_alma, query="   ", agent="helena")

        assert result["success"] is False
        assert "query cannot be empty" in result["error"]


class TestAlmaCompressAndLearn:
    """Tests for alma_compress_and_learn tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.storage = MagicMock()
        alma.llm = None
        return alma

    def test_compress_and_learn_empty_content(self, mock_alma):
        """Test error for empty content."""
        result = alma_compress_and_learn(alma=mock_alma, content="", agent="helena")

        assert result["success"] is False
        assert "content cannot be empty" in result["error"]

    def test_compress_and_learn_empty_agent(self, mock_alma):
        """Test error for empty agent."""
        result = alma_compress_and_learn(
            alma=mock_alma, content="test content", agent=""
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]

    def test_compress_and_learn_invalid_level(self, mock_alma):
        """Test error for invalid compression level."""
        result = alma_compress_and_learn(
            alma=mock_alma,
            content="test content",
            agent="helena",
            compression_level="invalid",
        )

        assert result["success"] is False
        assert "compression_level must be one of" in result["error"]

    def test_compress_and_learn_invalid_type(self, mock_alma):
        """Test error for invalid memory type."""
        result = alma_compress_and_learn(
            alma=mock_alma,
            content="test content",
            agent="helena",
            memory_type="invalid",
        )

        assert result["success"] is False
        assert "memory_type must be one of" in result["error"]


class TestAlmaExtractHeuristic:
    """Tests for alma_extract_heuristic tool - input validation."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.project_id = "test-project"
        alma.storage = MagicMock()
        alma.llm = None
        return alma

    def test_extract_heuristic_too_few_experiences(self, mock_alma):
        """Test error when too few experiences."""
        result = alma_extract_heuristic(
            alma=mock_alma,
            experiences=["Event 1", "Event 2"],
            agent="helena",
        )

        assert result["success"] is False
        assert "Need at least 3 experiences" in result["error"]
        assert result["provided"] == 2

    def test_extract_heuristic_empty_experiences(self, mock_alma):
        """Test error when no experiences."""
        result = alma_extract_heuristic(
            alma=mock_alma,
            experiences=[],
            agent="helena",
        )

        assert result["success"] is False
        assert "Need at least 3 experiences" in result["error"]

    def test_extract_heuristic_none_experiences(self, mock_alma):
        """Test error when experiences is None."""
        result = alma_extract_heuristic(
            alma=mock_alma,
            experiences=None,
            agent="helena",
        )

        assert result["success"] is False
        assert "Need at least 3 experiences" in result["error"]

    def test_extract_heuristic_empty_agent(self, mock_alma):
        """Test error for empty agent."""
        result = alma_extract_heuristic(
            alma=mock_alma,
            experiences=["E1", "E2", "E3"],
            agent="",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]
