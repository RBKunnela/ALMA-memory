"""
Integration tests for Victor with ALMA.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from alma.integration import (
    VictorHooks,
    APITestContext,
    APITestOutcome,
    create_victor_hooks,
    victor_pre_task,
    victor_post_task,
    VICTOR_CATEGORIES,
    VICTOR_FORBIDDEN,
)
from alma.types import MemorySlice, Heuristic, DomainKnowledge


class TestVictorHooks:
    """Tests for VictorHooks integration."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()

        # Mock retrieve to return a MemorySlice
        alma.retrieve.return_value = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="victor",
                    project_id="test",
                    condition="api endpoint testing",
                    strategy="check auth before payload validation",
                    confidence=0.90,
                    occurrence_count=15,
                    success_count=14,
                    last_validated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[
                DomainKnowledge(
                    id="dk1",
                    agent="victor",
                    project_id="test",
                    domain="error_handling",
                    fact="Always return structured error responses with error codes",
                    source="api_test:success=True",
                    last_verified=datetime.now(timezone.utc),
                ),
            ],
            preferences=[],
        )

        alma.learn.return_value = True
        alma.add_domain_knowledge.return_value = MagicMock(id="dk_new")
        alma.get_stats.return_value = {"heuristics": 8, "outcomes": 30}

        return alma

    @pytest.fixture
    def victor_hooks(self, mock_alma):
        """Create VictorHooks with mock ALMA."""
        with patch("alma.integration.victor.CodingDomain") as mock_domain:
            mock_domain.create_victor.return_value = MagicMock()
            hooks = VictorHooks(alma=mock_alma)
            hooks.alma = mock_alma  # Ensure mock is used
            hooks.harness = None  # Disable harness to test direct ALMA retrieval
            return hooks

    def test_pre_task_retrieves_memories(self, victor_hooks, mock_alma):
        """Test that pre_task retrieves relevant memories."""
        context = APITestContext(
            task_description="Test user authentication endpoint",
            task_type="authentication_patterns",
            agent_name="victor",
            project_id="test-project",
            endpoint="/api/v1/auth/login",
            method="POST",
            is_auth_test=True,
        )

        memories = victor_hooks.pre_task(context)

        assert memories is not None
        mock_alma.retrieve.assert_called()

    def test_post_task_records_learning(self, victor_hooks, mock_alma):
        """Test that post_task records learning."""
        context = APITestContext(
            task_description="Test user creation",
            task_type="api_design_patterns",
            agent_name="victor",
            project_id="test-project",
            endpoint="/api/v1/users",
            method="POST",
        )

        outcome = APITestOutcome(
            success=True,
            strategy_used="validate request, check auth, process",
            response_status=201,
            response_time_ms=150,
        )

        result = victor_hooks.post_task(context, outcome)

        assert result is True

    def test_get_api_patterns(self, victor_hooks, mock_alma):
        """Test getting API patterns."""
        patterns = victor_hooks.get_api_patterns("CRUD")

        assert isinstance(patterns, list)
        mock_alma.retrieve.assert_called()

    def test_get_error_handling_patterns(self, victor_hooks, mock_alma):
        """Test getting error handling patterns."""
        patterns = victor_hooks.get_error_handling_patterns()

        assert isinstance(patterns, list)

    def test_get_performance_strategies(self, victor_hooks, mock_alma):
        """Test getting performance strategies."""
        strategies = victor_hooks.get_performance_strategies()

        assert isinstance(strategies, list)

    def test_get_auth_patterns(self, victor_hooks, mock_alma):
        """Test getting auth patterns."""
        patterns = victor_hooks.get_auth_patterns()

        assert isinstance(patterns, list)

    def test_format_api_test_prompt(self, victor_hooks, mock_alma):
        """Test prompt formatting for API tests."""
        memories = mock_alma.retrieve.return_value
        context = APITestContext(
            task_description="Test rate limiting",
            task_type="performance_optimization",
            agent_name="victor",
            project_id="test",
            endpoint="/api/v1/search",
            method="GET",
            is_performance_test=True,
        )

        prompt = victor_hooks.format_api_test_prompt(memories, context)

        assert "Test rate limiting" in prompt
        assert "performance_optimization" in prompt
        assert "GET /api/v1/search" in prompt

    def test_record_api_pattern(self, victor_hooks, mock_alma):
        """Test recording an API pattern."""
        result = victor_hooks.record_api_pattern(
            endpoint="/api/v1/users",
            method="POST",
            pattern_type="error_handling",
            description="Return 422 for validation errors with field details",
            success=True,
        )

        assert result is True
        mock_alma.add_domain_knowledge.assert_called()

    def test_record_performance_metric(self, victor_hooks, mock_alma):
        """Test recording a performance metric."""
        result = victor_hooks.record_performance_metric(
            endpoint="/api/v1/search",
            response_time_ms=250,
            query_count=3,
            threshold_ms=500,
        )

        assert result is True
        mock_alma.add_domain_knowledge.assert_called()


class TestAPITestContext:
    """Tests for APITestContext."""

    def test_infers_auth_type(self):
        """Test that auth flag sets correct type."""
        context = APITestContext(
            task_description="Test JWT validation",
            task_type="",
            agent_name="victor",
            project_id="test",
            is_auth_test=True,
        )

        assert context.task_type == "authentication_patterns"

    def test_infers_performance_type(self):
        """Test that performance flag sets correct type."""
        context = APITestContext(
            task_description="Check response times",
            task_type="",
            agent_name="victor",
            project_id="test",
            is_performance_test=True,
        )

        assert context.task_type == "performance_optimization"

    def test_infers_database_type(self):
        """Test that database flag sets correct type."""
        context = APITestContext(
            task_description="Validate query results",
            task_type="",
            agent_name="victor",
            project_id="test",
            is_database_test=True,
        )

        assert context.task_type == "database_query_patterns"

    def test_infers_api_type_from_endpoint(self):
        """Test that endpoint presence sets api type."""
        context = APITestContext(
            task_description="Test endpoint",
            task_type="",
            agent_name="victor",
            project_id="test",
            endpoint="/api/v1/users",
        )

        assert context.task_type == "api_design_patterns"


class TestAPITestOutcome:
    """Tests for APITestOutcome."""

    def test_response_time_used_as_duration(self):
        """Test that response_time_ms is used as duration."""
        outcome = APITestOutcome(
            success=True,
            strategy_used="test strategy",
            response_time_ms=150,
        )

        assert outcome.duration_ms == 150


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_victor_pre_task(self):
        """Test victor_pre_task convenience function."""
        mock_alma = MagicMock()
        mock_alma.retrieve.return_value = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )

        with patch("alma.integration.victor.CodingDomain"):
            result = victor_pre_task(
                alma=mock_alma,
                task="Test user API",
                endpoint="/api/v1/users",
                method="GET",
            )

        assert "memories" in result
        assert "prompt" in result
        assert "context" in result

    def test_victor_post_task(self):
        """Test victor_post_task convenience function."""
        mock_alma = MagicMock()
        mock_alma.learn.return_value = True

        with patch("alma.integration.victor.CodingDomain"):
            result = victor_post_task(
                alma=mock_alma,
                task="Test user API",
                success=True,
                strategy_used="validate request, check auth",
                endpoint="/api/v1/users",
                response_status=200,
                response_time_ms=150,
            )

        assert result is True


class TestVictorCategoryConstants:
    """Tests for Victor category constants."""

    def test_victor_categories_defined(self):
        """Test that Victor categories are defined."""
        assert "api_design_patterns" in VICTOR_CATEGORIES
        assert "authentication_patterns" in VICTOR_CATEGORIES
        assert "error_handling" in VICTOR_CATEGORIES
        assert "database_query_patterns" in VICTOR_CATEGORIES

    def test_victor_forbidden_defined(self):
        """Test that Victor forbidden categories are defined."""
        assert "frontend_styling" in VICTOR_FORBIDDEN
        assert "ui_testing" in VICTOR_FORBIDDEN
        assert "marketing_content" in VICTOR_FORBIDDEN
