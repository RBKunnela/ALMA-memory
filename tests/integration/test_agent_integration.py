"""
End-to-end integration tests for Claude Code agents with ALMA.

Tests the full flow from agent initialization through memory operations.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from alma.integration import (
    AgentType,
    TaskContext,
    TaskOutcome,
    ClaudeAgentHooks,
    AgentIntegration,
    create_integration,
)
from alma.types import MemorySlice, Heuristic, Outcome


class TestAgentIntegration:
    """Tests for AgentIntegration manager."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.retrieve.return_value = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )
        alma.learn.return_value = True
        alma.get_stats.return_value = {"heuristics": 0, "outcomes": 0}
        return alma

    def test_register_agent(self, mock_alma):
        """Test registering an agent."""
        integration = AgentIntegration(mock_alma)
        hooks = integration.register_agent(AgentType.HELENA)

        assert hooks is not None
        assert hooks.agent_name == "helena"
        assert "helena" in integration.list_agents()

    def test_get_hooks(self, mock_alma):
        """Test getting hooks for an agent."""
        integration = AgentIntegration(mock_alma)
        integration.register_agent(AgentType.VICTOR)

        hooks = integration.get_hooks("victor")

        assert hooks is not None
        assert hooks.agent_name == "victor"

    def test_get_hooks_unknown_agent(self, mock_alma):
        """Test getting hooks for unknown agent."""
        integration = AgentIntegration(mock_alma)

        hooks = integration.get_hooks("unknown")

        assert hooks is None

    def test_list_agents(self, mock_alma):
        """Test listing all registered agents."""
        integration = AgentIntegration(mock_alma)
        integration.register_agent(AgentType.HELENA)
        integration.register_agent(AgentType.VICTOR)

        agents = integration.list_agents()

        assert "helena" in agents
        assert "victor" in agents
        assert len(agents) == 2

    def test_get_all_stats(self, mock_alma):
        """Test getting stats for all agents."""
        integration = AgentIntegration(mock_alma)
        integration.register_agent(AgentType.HELENA)
        integration.register_agent(AgentType.VICTOR)

        stats = integration.get_all_stats()

        assert "helena" in stats
        assert "victor" in stats


class TestClaudeAgentHooks:
    """Tests for ClaudeAgentHooks base class."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA instance."""
        alma = MagicMock()
        alma.retrieve.return_value = MemorySlice(
            heuristics=[
                Heuristic(
                    id="h1",
                    agent="test",
                    project_id="test",
                    condition="testing",
                    strategy="test first",
                    confidence=0.8,
                    occurrence_count=5,
                    success_count=4,
                    last_validated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                ),
            ],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )
        alma.learn.return_value = True
        alma.add_domain_knowledge.return_value = MagicMock(id="dk1")
        alma.get_stats.return_value = {"heuristics": 1}
        return alma

    def test_pre_task_retrieves_memories(self, mock_alma):
        """Test pre_task hook."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)

        context = TaskContext(
            task_description="Test something",
            task_type="testing",
            agent_name="custom",
            project_id="test",
        )

        memories = hooks.pre_task(context)

        assert memories is not None
        mock_alma.retrieve.assert_called_once()

    def test_post_task_records_learning(self, mock_alma):
        """Test post_task hook."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)

        context = TaskContext(
            task_description="Test something",
            task_type="testing",
            agent_name="custom",
            project_id="test",
        )

        outcome = TaskOutcome(
            success=True,
            strategy_used="test first",
            duration_ms=1000,
        )

        result = hooks.post_task(context, outcome)

        assert result is True
        mock_alma.learn.assert_called_once()

    def test_auto_learn_disabled(self, mock_alma):
        """Test that auto_learn=False skips learning."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM, auto_learn=False)

        context = TaskContext(
            task_description="Test something",
            task_type="testing",
            agent_name="custom",
            project_id="test",
        )

        outcome = TaskOutcome(success=True, strategy_used="test")

        result = hooks.post_task(context, outcome)

        assert result is False
        mock_alma.learn.assert_not_called()

    def test_format_memories_for_prompt(self, mock_alma):
        """Test memory formatting for prompt."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)
        memories = mock_alma.retrieve.return_value

        prompt = hooks.format_memories_for_prompt(memories)

        assert "Relevant Memory" in prompt
        assert "Proven Strategies" in prompt
        assert "test first" in prompt
        assert "80%" in prompt  # 0.8 confidence

    def test_format_memories_empty(self, mock_alma):
        """Test formatting empty memories."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)
        memories = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )

        prompt = hooks.format_memories_for_prompt(memories)

        assert prompt == ""

    def test_add_knowledge(self, mock_alma):
        """Test adding domain knowledge."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)

        result = hooks.add_knowledge(
            domain="testing",
            fact="This is a fact",
            source="discovered",
        )

        assert result is True
        mock_alma.add_domain_knowledge.assert_called_once()

    def test_get_agent_stats(self, mock_alma):
        """Test getting agent stats."""
        hooks = ClaudeAgentHooks(mock_alma, AgentType.CUSTOM)

        stats = hooks.get_agent_stats()

        assert "heuristics" in stats
        mock_alma.get_stats.assert_called_once()


class TestCreateIntegration:
    """Tests for create_integration convenience function."""

    def test_creates_default_agents(self):
        """Test creating integration with default agents."""
        mock_alma = MagicMock()
        mock_alma.retrieve.return_value = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )

        with patch("alma.harness.domains.CodingDomain") as mock_domain:
            mock_domain.create_helena.return_value = MagicMock()
            mock_domain.create_victor.return_value = MagicMock()

            integration = create_integration(mock_alma)

        agents = integration.list_agents()
        assert "helena" in agents
        assert "victor" in agents

    def test_creates_specific_agents(self):
        """Test creating integration with specific agents."""
        mock_alma = MagicMock()
        mock_alma.retrieve.return_value = MemorySlice(
            heuristics=[],
            anti_patterns=[],
            outcomes=[],
            domain_knowledge=[],
            preferences=[],
        )

        with patch("alma.harness.domains.CodingDomain") as mock_domain:
            mock_domain.create_helena.return_value = MagicMock()

            integration = create_integration(mock_alma, agents=[AgentType.HELENA])

        agents = integration.list_agents()
        assert "helena" in agents
        assert "victor" not in agents


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.fixture
    def mock_alma(self):
        """Create mock ALMA with realistic behavior."""
        alma = MagicMock()

        # Track outcomes for learning
        outcomes = []

        def learn_side_effect(*args, **kwargs):
            outcomes.append(kwargs)
            return True

        alma.learn.side_effect = learn_side_effect

        # Return memories based on agent
        def retrieve_side_effect(task, agent, **kwargs):
            if agent == "helena":
                return MemorySlice(
                    heuristics=[
                        Heuristic(
                            id="h1",
                            agent="helena",
                            project_id="test",
                            condition="form testing",
                            strategy="validate inputs first",
                            confidence=0.85,
                            occurrence_count=10,
                            success_count=9,
                            last_validated=datetime.now(timezone.utc),
                            created_at=datetime.now(timezone.utc),
                        ),
                    ],
                    anti_patterns=[],
                    outcomes=[],
                    domain_knowledge=[],
                    preferences=[],
                )
            else:
                return MemorySlice(
                    heuristics=[
                        Heuristic(
                            id="h2",
                            agent="victor",
                            project_id="test",
                            condition="api testing",
                            strategy="check auth first",
                            confidence=0.90,
                            occurrence_count=15,
                            success_count=14,
                            last_validated=datetime.now(timezone.utc),
                            created_at=datetime.now(timezone.utc),
                        ),
                    ],
                    anti_patterns=[],
                    outcomes=[],
                    domain_knowledge=[],
                    preferences=[],
                )

        alma.retrieve.side_effect = retrieve_side_effect
        alma.get_stats.return_value = {"heuristics": 2, "outcomes": 10}
        alma.add_domain_knowledge.return_value = MagicMock(id="dk1")

        return alma

    def test_full_helena_flow(self, mock_alma):
        """Test full Helena workflow: pre-task, execute, post-task."""
        integration = AgentIntegration(mock_alma)
        hooks = integration.register_agent(AgentType.HELENA)

        # Pre-task: Get memories
        context = TaskContext(
            task_description="Test login form validation",
            task_type="form_testing",
            agent_name="helena",
            project_id="test",
        )

        memories = hooks.pre_task(context)
        assert len(memories.heuristics) == 1
        assert "validate inputs first" in memories.heuristics[0].strategy

        # Format for prompt
        prompt = hooks.format_memories_for_prompt(memories)
        assert "validate inputs first" in prompt

        # Post-task: Record learning
        outcome = TaskOutcome(
            success=True,
            strategy_used="validated inputs, then tested submission",
            tools_used=["playwright"],
            duration_ms=2000,
        )

        result = hooks.post_task(context, outcome)
        assert result is True

    def test_full_victor_flow(self, mock_alma):
        """Test full Victor workflow: pre-task, execute, post-task."""
        integration = AgentIntegration(mock_alma)
        hooks = integration.register_agent(AgentType.VICTOR)

        # Pre-task: Get memories
        context = TaskContext(
            task_description="Test user authentication endpoint",
            task_type="authentication_patterns",
            agent_name="victor",
            project_id="test",
        )

        memories = hooks.pre_task(context)
        assert len(memories.heuristics) == 1
        assert "check auth first" in memories.heuristics[0].strategy

        # Format for prompt
        prompt = hooks.format_memories_for_prompt(memories)
        assert "check auth first" in prompt

        # Post-task: Record learning
        outcome = TaskOutcome(
            success=True,
            strategy_used="checked auth, then validated payload",
            tools_used=["api_client"],
            duration_ms=500,
        )

        result = hooks.post_task(context, outcome)
        assert result is True

    def test_multi_agent_workflow(self, mock_alma):
        """Test workflow with multiple agents."""
        integration = AgentIntegration(mock_alma)
        integration.register_agent(AgentType.HELENA)
        integration.register_agent(AgentType.VICTOR)

        # Helena tests UI
        helena_hooks = integration.get_hooks("helena")
        helena_context = TaskContext(
            task_description="Test form submission",
            task_type="form_testing",
            agent_name="helena",
            project_id="test",
        )
        helena_memories = helena_hooks.pre_task(helena_context)
        assert "validate inputs first" in helena_memories.heuristics[0].strategy

        # Victor tests API
        victor_hooks = integration.get_hooks("victor")
        victor_context = TaskContext(
            task_description="Test form submission API",
            task_type="api_design_patterns",
            agent_name="victor",
            project_id="test",
        )
        victor_memories = victor_hooks.pre_task(victor_context)
        assert "check auth first" in victor_memories.heuristics[0].strategy

        # Both complete their tasks
        helena_outcome = TaskOutcome(success=True, strategy_used="validated form")
        victor_outcome = TaskOutcome(success=True, strategy_used="checked API")

        helena_hooks.post_task(helena_context, helena_outcome)
        victor_hooks.post_task(victor_context, victor_outcome)

        # Verify both learned
        assert mock_alma.learn.call_count == 2
