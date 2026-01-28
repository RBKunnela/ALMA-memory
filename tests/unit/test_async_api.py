"""
Unit tests for ALMA async API.

Tests async_retrieve, async_learn, async_add_domain_knowledge,
async_add_user_preference, async_forget, and async_get_stats.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alma import ALMA, MemorySlice
from alma.exceptions import ScopeViolationError
from alma.mcp.tools import (
    async_alma_add_knowledge,
    async_alma_add_preference,
    async_alma_forget,
    async_alma_health,
    async_alma_learn,
    async_alma_retrieve,
    async_alma_stats,
)
from alma.types import DomainKnowledge, Heuristic, Outcome, UserPreference


class TestAsyncRetrieve:
    """Tests for ALMA.async_retrieve()."""

    @pytest.mark.asyncio
    async def test_async_retrieve_returns_memory_slice(self, alma_instance):
        """Test that async_retrieve returns a MemorySlice."""
        result = await alma_instance.async_retrieve(
            task="Test login form validation",
            agent="helena",
            top_k=5,
        )

        assert isinstance(result, MemorySlice)
        assert result.agent == "helena"

    @pytest.mark.asyncio
    async def test_async_retrieve_with_user_id(self, alma_instance):
        """Test async_retrieve with user_id parameter."""
        result = await alma_instance.async_retrieve(
            task="Test form",
            agent="helena",
            user_id="test-user",
            top_k=3,
        )

        assert isinstance(result, MemorySlice)

    @pytest.mark.asyncio
    async def test_async_retrieve_equivalent_to_sync(self, alma_instance):
        """Test that async_retrieve returns same results as sync retrieve."""
        task = "Test API endpoint"
        agent = "victor"

        sync_result = alma_instance.retrieve(task=task, agent=agent, top_k=5)
        async_result = await alma_instance.async_retrieve(
            task=task, agent=agent, top_k=5
        )

        # Should have same structure
        assert sync_result.agent == async_result.agent
        assert len(sync_result.heuristics) == len(async_result.heuristics)


class TestAsyncLearn:
    """Tests for ALMA.async_learn()."""

    @pytest.mark.asyncio
    async def test_async_learn_success(self, alma_instance):
        """Test successful async learning."""
        result = await alma_instance.async_learn(
            agent="helena",
            task="Test login form",
            outcome="success",
            strategy_used="validate inputs first",
            task_type="form_testing",
            duration_ms=1500,
        )

        assert isinstance(result, Outcome)
        assert result.agent == "helena"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_async_learn_failure_outcome(self, alma_instance):
        """Test async learning with failure outcome."""
        result = await alma_instance.async_learn(
            agent="helena",
            task="Test modal dialog",
            outcome="failure",
            strategy_used="click without wait",
            error_message="Element not found",
        )

        assert isinstance(result, Outcome)
        assert result.success is False
        assert result.error_message == "Element not found"

    @pytest.mark.asyncio
    async def test_async_learn_with_feedback(self, alma_instance):
        """Test async learning with user feedback."""
        result = await alma_instance.async_learn(
            agent="victor",
            task="Test auth endpoint",
            outcome="success",
            strategy_used="validate tokens first",
            feedback="Good approach",
        )

        assert isinstance(result, Outcome)
        assert result.user_feedback == "Good approach"


class TestAsyncAddUserPreference:
    """Tests for ALMA.async_add_user_preference()."""

    @pytest.mark.asyncio
    async def test_async_add_preference(self, alma_instance):
        """Test adding user preference asynchronously."""
        result = await alma_instance.async_add_user_preference(
            user_id="test-user",
            category="code_style",
            preference="No emojis in code",
            source="explicit_instruction",
        )

        assert isinstance(result, UserPreference)
        assert result.user_id == "test-user"
        assert result.preference == "No emojis in code"

    @pytest.mark.asyncio
    async def test_async_add_preference_inferred_source(self, alma_instance):
        """Test adding preference with inferred source."""
        result = await alma_instance.async_add_user_preference(
            user_id="test-user",
            category="workflow",
            preference="Run tests before commit",
            source="inferred",
        )

        assert result.source == "inferred"


class TestAsyncAddDomainKnowledge:
    """Tests for ALMA.async_add_domain_knowledge()."""

    @pytest.mark.asyncio
    async def test_async_add_knowledge(self, alma_instance):
        """Test adding domain knowledge asynchronously."""
        result = await alma_instance.async_add_domain_knowledge(
            agent="helena",
            domain="selector_patterns",
            fact="data-testid selectors are most stable",
            source="user_stated",
        )

        assert isinstance(result, DomainKnowledge)
        assert result.agent == "helena"
        assert result.domain == "selector_patterns"

    @pytest.mark.asyncio
    async def test_async_add_knowledge_scope_violation(self, alma_instance):
        """Test that scope violations are raised for async knowledge."""
        # helena's scope doesn't allow backend_logic domain
        with pytest.raises(ScopeViolationError):
            await alma_instance.async_add_domain_knowledge(
                agent="helena",
                domain="backend_logic",
                fact="Some backend fact",
                source="user_stated",
            )


class TestAsyncForget:
    """Tests for ALMA.async_forget()."""

    @pytest.mark.asyncio
    async def test_async_forget(self, alma_instance):
        """Test async forget operation."""
        # First add some data to forget
        await alma_instance.async_learn(
            agent="helena",
            task="Old task",
            outcome="success",
            strategy_used="old strategy",
        )

        result = await alma_instance.async_forget(
            agent="helena",
            older_than_days=0,  # Forget everything
            below_confidence=1.0,  # High threshold to catch low-confidence items
        )

        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_async_forget_all_agents(self, alma_instance):
        """Test async forget for all agents."""
        result = await alma_instance.async_forget(
            older_than_days=90,
            below_confidence=0.3,
        )

        assert isinstance(result, int)


class TestAsyncGetStats:
    """Tests for ALMA.async_get_stats()."""

    @pytest.mark.asyncio
    async def test_async_get_stats(self, alma_instance):
        """Test getting stats asynchronously."""
        result = await alma_instance.async_get_stats(agent="helena")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_async_get_stats_all_agents(self, alma_instance):
        """Test getting stats for all agents."""
        result = await alma_instance.async_get_stats()

        assert isinstance(result, dict)


class TestAsyncConcurrency:
    """Tests for concurrent async operations."""

    @pytest.mark.asyncio
    async def test_concurrent_retrieves(self, alma_instance):
        """Test multiple concurrent async_retrieve calls."""
        tasks = [
            alma_instance.async_retrieve(task=f"Task {i}", agent="helena", top_k=3)
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, MemorySlice)

    @pytest.mark.asyncio
    async def test_concurrent_learns(self, alma_instance):
        """Test multiple concurrent async_learn calls."""
        tasks = [
            alma_instance.async_learn(
                agent="helena",
                task=f"Task {i}",
                outcome="success",
                strategy_used=f"Strategy {i}",
            )
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, Outcome)

    @pytest.mark.asyncio
    async def test_mixed_async_operations(self, alma_instance):
        """Test mixing different async operations concurrently."""
        tasks = [
            alma_instance.async_retrieve(task="Test task", agent="helena"),
            alma_instance.async_learn(
                agent="victor",
                task="API test",
                outcome="success",
                strategy_used="validate first",
            ),
            alma_instance.async_get_stats(),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert isinstance(results[0], MemorySlice)
        assert isinstance(results[1], Outcome)
        assert isinstance(results[2], dict)


# =============================================================================
# Tests for Async MCP Tools
# =============================================================================


class TestAsyncMCPRetrieve:
    """Tests for async_alma_retrieve MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with async methods."""
        alma = MagicMock(spec=ALMA)
        alma.project_id = "test-project"
        now = datetime.now(timezone.utc)

        # Create a proper MemorySlice
        memory_slice = MemorySlice(
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
            domain_knowledge=[],
            preferences=[],
            anti_patterns=[],
            outcomes=[],
            query="test query",
            agent="helena",
            retrieval_time_ms=10,
        )

        # Mock async_retrieve as a coroutine
        async def mock_async_retrieve(**kwargs):
            return memory_slice

        alma.async_retrieve = mock_async_retrieve
        return alma

    @pytest.mark.asyncio
    async def test_async_retrieve_success(self, mock_alma):
        """Test successful async retrieval via MCP tool."""
        result = await async_alma_retrieve(
            alma=mock_alma,
            task="Test login form",
            agent="helena",
            top_k=5,
        )

        assert result["success"] is True
        assert "memories" in result
        assert "prompt_injection" in result

    @pytest.mark.asyncio
    async def test_async_retrieve_empty_task(self, mock_alma):
        """Test async retrieval with empty task."""
        result = await async_alma_retrieve(
            alma=mock_alma,
            task="",
            agent="helena",
        )

        assert result["success"] is False
        assert "task cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_async_retrieve_empty_agent(self, mock_alma):
        """Test async retrieval with empty agent."""
        result = await async_alma_retrieve(
            alma=mock_alma,
            task="Test form",
            agent="",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]


class TestAsyncMCPLearn:
    """Tests for async_alma_learn MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance with async learn."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        outcome = Outcome(
            id="out-test-1",
            agent="helena",
            project_id="test-project",
            task_type="form_testing",
            task_description="Test login form",
            success=True,
            strategy_used="validate inputs first",
            timestamp=now,
        )

        async def mock_async_learn(**kwargs):
            return outcome

        alma.async_learn = mock_async_learn
        return alma

    @pytest.mark.asyncio
    async def test_async_learn_success(self, mock_alma):
        """Test successful async learning via MCP tool."""
        result = await async_alma_learn(
            alma=mock_alma,
            agent="helena",
            task="Test login form",
            outcome="success",
            strategy_used="validate inputs first",
        )

        assert result["success"] is True
        assert result["learned"] is True
        assert result["outcome"]["id"] == "out-test-1"

    @pytest.mark.asyncio
    async def test_async_learn_missing_required_fields(self, mock_alma):
        """Test async learn with missing required fields."""
        result = await async_alma_learn(
            alma=mock_alma,
            agent="",
            task="Test form",
            outcome="success",
            strategy_used="test",
        )

        assert result["success"] is False
        assert "agent cannot be empty" in result["error"]


class TestAsyncMCPAddPreference:
    """Tests for async_alma_add_preference MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        pref = UserPreference(
            id="pref-1",
            user_id="user-123",
            category="code_style",
            preference="No emojis",
            source="explicit_instruction",
            timestamp=now,
        )

        async def mock_async_add_preference(**kwargs):
            return pref

        alma.async_add_user_preference = mock_async_add_preference
        return alma

    @pytest.mark.asyncio
    async def test_async_add_preference_success(self, mock_alma):
        """Test successful async preference addition."""
        result = await async_alma_add_preference(
            alma=mock_alma,
            user_id="user-123",
            category="code_style",
            preference="No emojis",
        )

        assert result["success"] is True
        assert result["preference"]["id"] == "pref-1"


class TestAsyncMCPAddKnowledge:
    """Tests for async_alma_add_knowledge MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        now = datetime.now(timezone.utc)

        knowledge = DomainKnowledge(
            id="dk-1",
            agent="helena",
            project_id="test-project",
            domain="testing",
            fact="Use data-testid",
            source="user_stated",
            last_verified=now,
        )

        async def mock_async_add_knowledge(**kwargs):
            return knowledge

        alma.async_add_domain_knowledge = mock_async_add_knowledge
        return alma

    @pytest.mark.asyncio
    async def test_async_add_knowledge_success(self, mock_alma):
        """Test successful async knowledge addition."""
        result = await async_alma_add_knowledge(
            alma=mock_alma,
            agent="helena",
            domain="testing",
            fact="Use data-testid",
        )

        assert result["success"] is True
        assert result["knowledge"]["id"] == "dk-1"


class TestAsyncMCPForget:
    """Tests for async_alma_forget MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)

        async def mock_async_forget(**kwargs):
            return 5

        alma.async_forget = mock_async_forget
        return alma

    @pytest.mark.asyncio
    async def test_async_forget_success(self, mock_alma):
        """Test successful async forget."""
        result = await async_alma_forget(
            alma=mock_alma,
            agent="helena",
            older_than_days=30,
            below_confidence=0.5,
        )

        assert result["success"] is True
        assert result["pruned_count"] == 5


class TestAsyncMCPStats:
    """Tests for async_alma_stats MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)

        async def mock_async_get_stats(**kwargs):
            return {
                "project_id": "test-project",
                "heuristics_count": 10,
                "outcomes_count": 50,
                "total_count": 75,
            }

        alma.async_get_stats = mock_async_get_stats
        return alma

    @pytest.mark.asyncio
    async def test_async_stats_success(self, mock_alma):
        """Test successful async stats."""
        result = await async_alma_stats(alma=mock_alma, agent="helena")

        assert result["success"] is True
        assert result["stats"]["total_count"] == 75


class TestAsyncMCPHealth:
    """Tests for async_alma_health MCP tool."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock(spec=ALMA)
        alma.project_id = "test-project"
        alma.scopes = {"helena": MagicMock(), "victor": MagicMock()}

        async def mock_async_get_stats(**kwargs):
            return {"total_count": 100}

        alma.async_get_stats = mock_async_get_stats
        return alma

    @pytest.mark.asyncio
    async def test_async_health_success(self, mock_alma):
        """Test successful async health check."""
        result = await async_alma_health(alma=mock_alma)

        assert result["success"] is True
        assert result["status"] == "healthy"
        assert result["project_id"] == "test-project"
        assert "helena" in result["registered_agents"]
