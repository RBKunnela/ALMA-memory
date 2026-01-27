"""
Integration tests for Helena with ALMA.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from alma.integration import (
    HelenaHooks,
    UITestContext,
    UITestOutcome,
    create_helena_hooks,
    helena_pre_task,
    helena_post_task,
    HELENA_CATEGORIES,
    HELENA_FORBIDDEN,
)
from alma.types import MemorySlice, Heuristic, DomainKnowledge


class TestHelenaHooks:
    """Tests for HelenaHooks integration."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()

        # Mock retrieve to return a MemorySlice
        alma.retrieve.return_value = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="helena",
                    project_id="test",
                    condition="form testing",
                    strategy="validate inputs before submit",
                    confidence=0.85,
                    occurrence_count=10,
                    success_count=9,
                    last_validated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[
                DomainKnowledge(
                    id="dk1",
                    agent="helena",
                    project_id="test",
                    domain="selector_patterns",
                    fact="data-testid selectors are most stable",
                    source="test_run:stability=0.95",
                    last_verified=datetime.now(timezone.utc),
                ),
            ],
            preferences=[],
        )

        alma.learn.return_value = True
        alma.add_domain_knowledge.return_value = MagicMock(id="dk_new")
        alma.get_stats.return_value = {"heuristics": 5, "outcomes": 20}

        return alma

    @pytest.fixture
    def helena_hooks(self, mock_alma):
        """Create HelenaHooks with mock ALMA."""
        with patch("alma.integration.helena.CodingDomain") as mock_domain:
            mock_domain.create_helena.return_value = MagicMock()
            hooks = HelenaHooks(alma=mock_alma)
            hooks.alma = mock_alma  # Ensure mock is used
            hooks.harness = None  # Disable harness to test direct ALMA retrieval
            return hooks

    def test_pre_task_retrieves_memories(self, helena_hooks, mock_alma):
        """Test that pre_task retrieves relevant memories."""
        context = UITestContext(
            task_description="Test login form validation",
            task_type="form_testing",
            agent_name="helena",
            project_id="test-project",
            component_type="form",
            page_url="/login",
        )

        memories = helena_hooks.pre_task(context)

        assert memories is not None
        mock_alma.retrieve.assert_called()

    def test_post_task_records_learning(self, helena_hooks, mock_alma):
        """Test that post_task records learning."""
        context = UITestContext(
            task_description="Test login form",
            task_type="form_testing",
            agent_name="helena",
            project_id="test-project",
        )

        outcome = UITestOutcome(
            success=True,
            strategy_used="validate inputs first, then submit",
            selectors_used=["[data-testid='email']", "[data-testid='password']"],
            duration_ms=1500,
        )

        result = helena_hooks.post_task(context, outcome)

        assert result is True

    def test_get_selector_patterns(self, helena_hooks, mock_alma):
        """Test getting selector patterns."""
        patterns = helena_hooks.get_selector_patterns("button")

        assert isinstance(patterns, list)
        mock_alma.retrieve.assert_called()

    def test_get_form_testing_strategies(self, helena_hooks, mock_alma):
        """Test getting form testing strategies."""
        strategies = helena_hooks.get_form_testing_strategies()

        assert isinstance(strategies, list)

    def test_format_ui_test_prompt(self, helena_hooks, mock_alma):
        """Test prompt formatting for UI tests."""
        memories = mock_alma.retrieve.return_value
        context = UITestContext(
            task_description="Test form accessibility",
            task_type="accessibility_testing",
            agent_name="helena",
            project_id="test",
            component_type="form",
            page_url="/contact",
            is_accessibility_test=True,
        )

        prompt = helena_hooks.format_ui_test_prompt(memories, context)

        assert "Test form accessibility" in prompt
        assert "accessibility_testing" in prompt
        assert "form" in prompt

    def test_record_selector_pattern(self, helena_hooks, mock_alma):
        """Test recording a selector pattern."""
        result = helena_hooks.record_selector_pattern(
            selector="[data-testid='submit-btn']",
            component_type="button",
            success=True,
            stability_score=0.9,
        )

        assert result is True
        mock_alma.add_domain_knowledge.assert_called()


class TestUITestContext:
    """Tests for UITestContext."""

    def test_infers_form_testing_type(self):
        """Test that form component infers form_testing type."""
        context = UITestContext(
            task_description="Test contact form",
            task_type="",
            agent_name="helena",
            project_id="test",
            component_type="form",
        )

        assert context.task_type == "form_testing"

    def test_infers_accessibility_type(self):
        """Test that accessibility flag sets correct type."""
        context = UITestContext(
            task_description="Check ARIA labels",
            task_type="",
            agent_name="helena",
            project_id="test",
            is_accessibility_test=True,
        )

        assert context.task_type == "accessibility_testing"

    def test_infers_visual_type(self):
        """Test that visual flag sets correct type."""
        context = UITestContext(
            task_description="Check button styling",
            task_type="",
            agent_name="helena",
            project_id="test",
            is_visual_test=True,
        )

        assert context.task_type == "visual_testing"


class TestUITestOutcome:
    """Tests for UITestOutcome."""

    def test_selectors_added_to_tools(self):
        """Test that selectors are added to tools_used."""
        outcome = UITestOutcome(
            success=True,
            strategy_used="test strategy",
            selectors_used=["#login", ".submit-btn"],
        )

        assert "#login" in outcome.tools_used
        assert ".submit-btn" in outcome.tools_used


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_helena_pre_task(self):
        """Test helena_pre_task convenience function."""
        mock_alma = MagicMock()
        mock_alma.retrieve.return_value = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )

        with patch("alma.integration.helena.CodingDomain"):
            result = helena_pre_task(
                alma=mock_alma,
                task="Test login form",
                component_type="form",
                page_url="/login",
            )

        assert "memories" in result
        assert "prompt" in result
        assert "context" in result

    def test_helena_post_task(self):
        """Test helena_post_task convenience function."""
        mock_alma = MagicMock()
        mock_alma.learn.return_value = True

        with patch("alma.integration.helena.CodingDomain"):
            result = helena_post_task(
                alma=mock_alma,
                task="Test login form",
                success=True,
                strategy_used="validate then submit",
                selectors_used=["#email", "#password"],
            )

        assert result is True


class TestHelenaCategoryConstants:
    """Tests for Helena category constants."""

    def test_helena_categories_defined(self):
        """Test that Helena categories are defined."""
        assert "testing_strategies" in HELENA_CATEGORIES
        assert "selector_patterns" in HELENA_CATEGORIES
        assert "form_testing" in HELENA_CATEGORIES
        assert "accessibility_testing" in HELENA_CATEGORIES

    def test_helena_forbidden_defined(self):
        """Test that Helena forbidden categories are defined."""
        assert "backend_logic" in HELENA_FORBIDDEN
        assert "database_queries" in HELENA_FORBIDDEN
        assert "api_design" in HELENA_FORBIDDEN
