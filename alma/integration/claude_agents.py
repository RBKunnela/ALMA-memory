"""
ALMA Claude Code Integration.

Provides hooks for integrating ALMA with Claude Code agents (Helena, Victor, etc).
These hooks enable agents to:
- Retrieve relevant memories before executing tasks
- Learn from task outcomes automatically
- Access domain-specific heuristics and patterns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from alma.core import ALMA
from alma.harness.base import Context, Harness, RunResult
from alma.types import MemorySlice

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Supported Claude Code agent types."""

    HELENA = "helena"
    VICTOR = "victor"
    CLARA = "clara"
    ALEX = "alex"
    CUSTOM = "custom"


@dataclass
class TaskContext:
    """
    Context for a Claude Code agent task.

    Captures all relevant information for memory retrieval and learning.
    """

    task_description: str
    task_type: str
    agent_name: str
    project_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_harness_context(self) -> Context:
        """Convert to Harness Context."""
        return Context(
            task=self.task_description,
            user_id=self.user_id,
            project_id=self.project_id,
            session_id=self.session_id,
            inputs=self.inputs,
            constraints=self.constraints,
            timestamp=self.timestamp,
        )


@dataclass
class TaskOutcome:
    """
    Outcome of a Claude Code agent task.

    Used for learning from task results.
    """

    success: bool
    strategy_used: str
    output: Any = None
    tools_used: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    feedback: Optional[str] = None
    reflections: List[str] = field(default_factory=list)

    def to_run_result(self) -> RunResult:
        """Convert to Harness RunResult."""
        return RunResult(
            success=self.success,
            output=self.output,
            reflections=self.reflections,
            tools_used=self.tools_used,
            duration_ms=self.duration_ms,
            error=self.error_message,
        )


class ClaudeAgentHooks:
    """
    Integration hooks for Claude Code agents.

    Provides a consistent interface for agents to interact with ALMA.

    Usage:
        hooks = ClaudeAgentHooks(alma, agent_type=AgentType.HELENA)

        # Before task execution
        memories = hooks.pre_task(task_context)
        prompt_enhancement = hooks.format_memories_for_prompt(memories)

        # After task execution
        hooks.post_task(task_context, task_outcome)
    """

    def __init__(
        self,
        alma: ALMA,
        agent_type: AgentType,
        harness: Optional[Harness] = None,
        auto_learn: bool = True,
    ):
        """
        Initialize hooks for an agent.

        Args:
            alma: ALMA instance for memory operations
            agent_type: Type of Claude Code agent
            harness: Optional pre-configured harness
            auto_learn: Whether to automatically learn from outcomes
        """
        self.alma = alma
        self.agent_type = agent_type
        self.agent_name = agent_type.value
        self.harness = harness
        self.auto_learn = auto_learn
        self._task_start_times: Dict[str, datetime] = {}

    def pre_task(
        self,
        context: TaskContext,
        top_k: int = 5,
    ) -> MemorySlice:
        """
        Pre-task hook: Retrieve relevant memories.

        Called before the agent executes a task to get relevant
        heuristics, patterns, and domain knowledge.

        Args:
            context: Task context
            top_k: Maximum items per memory type

        Returns:
            MemorySlice with relevant memories
        """
        # Track start time for duration calculation
        task_id = f"{context.project_id}:{context.timestamp.isoformat()}"
        self._task_start_times[task_id] = datetime.now(timezone.utc)

        logger.debug(
            f"[{self.agent_name}] Pre-task: Retrieving memories for '{context.task_description[:50]}...'"
        )

        if self.harness:
            # Use harness pre_run method
            harness_context = context.to_harness_context()
            return self.harness.pre_run(harness_context)
        else:
            # Direct ALMA retrieval
            return self.alma.retrieve(
                task=context.task_description,
                agent=self.agent_name,
                user_id=context.user_id,
                top_k=top_k,
            )

    def post_task(
        self,
        context: TaskContext,
        outcome: TaskOutcome,
    ) -> bool:
        """
        Post-task hook: Learn from the outcome.

        Called after the agent completes a task to record the outcome
        and potentially update heuristics.

        Args:
            context: Original task context
            outcome: Task outcome

        Returns:
            True if learning was recorded, False otherwise
        """
        if not self.auto_learn:
            logger.debug(f"[{self.agent_name}] Auto-learn disabled, skipping")
            return False

        # Calculate duration if not provided
        task_id = f"{context.project_id}:{context.timestamp.isoformat()}"
        if outcome.duration_ms is None and task_id in self._task_start_times:
            start = self._task_start_times.pop(task_id)
            outcome.duration_ms = int(
                (datetime.now(timezone.utc) - start).total_seconds() * 1000
            )

        logger.debug(
            f"[{self.agent_name}] Post-task: Recording {'success' if outcome.success else 'failure'} "
            f"for '{context.task_description[:50]}...'"
        )

        if self.harness:
            # Use harness post_run method
            harness_context = context.to_harness_context()
            run_result = outcome.to_run_result()
            self.harness.post_run(harness_context, run_result)
            return True
        else:
            # Direct ALMA learning
            return self.alma.learn(
                agent=self.agent_name,
                task=context.task_description,
                outcome="success" if outcome.success else "failure",
                strategy_used=outcome.strategy_used,
                task_type=context.task_type,
                duration_ms=outcome.duration_ms,
                error_message=outcome.error_message,
                feedback=outcome.feedback,
            )

    def format_memories_for_prompt(
        self,
        memories: MemorySlice,
        include_section_headers: bool = True,
    ) -> str:
        """
        Format memories for injection into agent prompt.

        Converts MemorySlice into a formatted string suitable for
        inclusion in the agent's system prompt.

        Args:
            memories: Retrieved memory slice
            include_section_headers: Whether to include markdown headers

        Returns:
            Formatted string for prompt injection
        """
        if memories.total_items == 0:
            return ""

        sections = []

        if include_section_headers:
            sections.append("## Relevant Memory (from past runs)")

        # Heuristics
        if memories.heuristics:
            if include_section_headers:
                sections.append("\n### Proven Strategies:")
            for h in memories.heuristics:
                confidence_pct = int(h.confidence * 100)
                sections.append(
                    f"- **{h.condition}**: {h.strategy} (confidence: {confidence_pct}%)"
                )

        # Anti-patterns
        if memories.anti_patterns:
            if include_section_headers:
                sections.append("\n### Avoid These:")
            for ap in memories.anti_patterns:
                sections.append(f"- ⚠️ {ap.pattern}: {ap.consequence}")

        # Domain knowledge
        if memories.domain_knowledge:
            if include_section_headers:
                sections.append("\n### Domain Knowledge:")
            for dk in memories.domain_knowledge:
                sections.append(f"- [{dk.domain}] {dk.fact}")

        # User preferences
        if memories.preferences:
            if include_section_headers:
                sections.append("\n### User Preferences:")
            for up in memories.preferences:
                sections.append(f"- [{up.category}] {up.preference}")

        # Recent outcomes
        if memories.outcomes:
            if include_section_headers:
                sections.append("\n### Recent Outcomes:")
            for o in memories.outcomes[:3]:  # Limit to 3 most recent
                status = "✓" if o.success else "✗"
                sections.append(
                    f"- {status} {o.task_type}: {o.task_description[:50]}..."
                )

        return "\n".join(sections)

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get memory statistics for this agent."""
        return self.alma.get_stats(agent=self.agent_name)

    def add_knowledge(
        self,
        domain: str,
        fact: str,
        source: str = "agent_discovered",
    ) -> bool:
        """
        Add domain knowledge discovered by the agent.

        Args:
            domain: Knowledge domain (must be within agent's scope)
            fact: The fact to remember
            source: How this was discovered

        Returns:
            True if knowledge was added, False if scope violation
        """
        result = self.alma.add_domain_knowledge(
            agent=self.agent_name,
            domain=domain,
            fact=fact,
            source=source,
        )
        return result is not None


class AgentIntegration:
    """
    High-level integration manager for multiple Claude Code agents.

    Manages hooks for all registered agents and provides a unified
    interface for the Claude Code runtime.

    Usage:
        integration = AgentIntegration(alma)
        integration.register_agent(AgentType.HELENA, helena_harness)
        integration.register_agent(AgentType.VICTOR, victor_harness)

        # Get hooks for a specific agent
        helena_hooks = integration.get_hooks("helena")
    """

    def __init__(self, alma: ALMA):
        """
        Initialize the integration manager.

        Args:
            alma: ALMA instance for memory operations
        """
        self.alma = alma
        self._agents: Dict[str, ClaudeAgentHooks] = {}

    def register_agent(
        self,
        agent_type: AgentType,
        harness: Optional[Harness] = None,
        auto_learn: bool = True,
    ) -> ClaudeAgentHooks:
        """
        Register an agent for integration.

        Args:
            agent_type: Type of agent
            harness: Optional pre-configured harness
            auto_learn: Whether to automatically learn from outcomes

        Returns:
            ClaudeAgentHooks for the agent
        """
        hooks = ClaudeAgentHooks(
            alma=self.alma,
            agent_type=agent_type,
            harness=harness,
            auto_learn=auto_learn,
        )
        self._agents[agent_type.value] = hooks
        logger.info(f"Registered agent: {agent_type.value}")
        return hooks

    def get_hooks(self, agent_name: str) -> Optional[ClaudeAgentHooks]:
        """
        Get hooks for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ClaudeAgentHooks or None if not registered
        """
        return self._agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self._agents.keys())

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get memory statistics for all registered agents."""
        return {name: hooks.get_agent_stats() for name, hooks in self._agents.items()}


def create_integration(
    alma: ALMA,
    agents: Optional[List[AgentType]] = None,
) -> AgentIntegration:
    """
    Convenience function to create an integration with default agents.

    Args:
        alma: ALMA instance
        agents: List of agents to register, or None for defaults (Helena, Victor)

    Returns:
        Configured AgentIntegration
    """
    from alma.harness.domains import CodingDomain

    integration = AgentIntegration(alma)

    if agents is None:
        agents = [AgentType.HELENA, AgentType.VICTOR]

    for agent_type in agents:
        if agent_type == AgentType.HELENA:
            harness = CodingDomain.create_helena(alma)
            integration.register_agent(agent_type, harness)
        elif agent_type == AgentType.VICTOR:
            harness = CodingDomain.create_victor(alma)
            integration.register_agent(agent_type, harness)
        else:
            # Register without harness for custom agents
            integration.register_agent(agent_type)

    return integration
