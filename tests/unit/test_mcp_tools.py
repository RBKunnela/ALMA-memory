"""
Unit tests for ALMA MCP tools.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import uuid

from alma import ALMA, MemorySlice
from alma.types import Heuristic, DomainKnowledge, UserPreference
from alma.mcp.tools import (
    alma_retrieve,
    alma_learn,
    alma_add_preference,
    alma_add_knowledge,
    alma_forget,
    alma_stats,
    alma_health,
)


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
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.learn.return_value = True
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
        mock_alma.learn.assert_called_once()

    def test_learn_failure_outcome(self, mock_alma):
        """Test learning with failure outcome."""
        result = alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Test modal",
            outcome="failure",
            strategy_used="click without wait",
            error_message="Element not found",
        )

        assert result["success"] is True
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
        """Test learning rejected due to scope."""
        mock_alma.learn.return_value = False

        result = alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Backend task",
            outcome="success",
            strategy_used="some strategy",
        )

        assert result["success"] is True
        assert result["learned"] is False
        assert "scope" in result["message"].lower()


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
        """Test knowledge rejected due to scope."""
        mock_alma.add_domain_knowledge.return_value = None

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
