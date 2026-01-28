"""
ALMA Harness Pattern - Base Classes.

The harness pattern decouples agent capabilities from domain, making any
tool-using workflow able to "learn" over time through memory injections.

Pattern Components:
1. Setting - Fixed environment (tools, constraints)
2. Context - Ephemeral per-run inputs
3. Agent - The executor that acts within setting+context
4. Memory Schema - Domain-specific structure for logging/retrieving learnings

Flow:
    Pre-run  -> Inject relevant memory slices
    Run      -> Agent acts, uses tools, logs reflections
    Post-run -> Update memory schema
    Repeat   -> Agent appears to "learn" without weight changes
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from alma.types import MemoryScope, MemorySlice


class ToolType(Enum):
    """Categories of tools available to agents."""

    SEARCH = "search"  # Web search, semantic search
    DATA_ACCESS = "data_access"  # APIs, databases
    EXECUTION = "execution"  # Code execution, automation
    COMMUNICATION = "communication"  # Email, messaging
    ANALYSIS = "analysis"  # Data processing, synthesis
    CREATION = "creation"  # Content generation, design


@dataclass
class Tool:
    """
    A tool available in the agent's setting.

    Tools are the building blocks agents use to accomplish tasks.
    """

    name: str
    description: str
    tool_type: ToolType
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format tool for inclusion in agent prompt."""
        prompt = f"**{self.name}** ({self.tool_type.value}): {self.description}"
        if self.constraints:
            prompt += f"\n  Constraints: {', '.join(self.constraints)}"
        return prompt


@dataclass
class Setting:
    """
    The fixed environment in which an agent operates.

    Includes available tools and immutable constraints that don't change
    between runs. The setting defines WHAT the agent CAN do.
    """

    name: str
    description: str
    tools: List[Tool] = field(default_factory=list)
    global_constraints: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Format setting for inclusion in agent prompt."""
        lines = [
            f"## Setting: {self.name}",
            self.description,
            "",
            "### Available Tools:",
        ]
        for tool in self.tools:
            lines.append(f"- {tool.to_prompt()}")

        if self.global_constraints:
            lines.append("")
            lines.append("### Constraints:")
            for c in self.global_constraints:
                lines.append(f"- {c}")

        return "\n".join(lines)


@dataclass
class Context:
    """
    Ephemeral inputs for a single run.

    This is injected fresh each time and contains task-specific information.
    The context defines WHAT the agent should do THIS run.
    """

    task: str
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_prompt(self) -> str:
        """Format context for inclusion in agent prompt."""
        lines = [
            "## Current Task",
            self.task,
        ]
        if self.inputs:
            lines.append("")
            lines.append("### Inputs:")
            for k, v in self.inputs.items():
                lines.append(f"- {k}: {v}")

        if self.constraints:
            lines.append("")
            lines.append("### Task Constraints:")
            for c in self.constraints:
                lines.append(f"- {c}")

        return "\n".join(lines)


@dataclass
class MemorySchema:
    """
    Domain-specific structure for logging and retrieving learnings.

    This defines WHAT gets remembered and HOW, ensuring relevance
    and preventing scope creep. Each domain has its own schema.
    """

    domain: str
    description: str

    # What this domain can learn
    learnable_categories: List[str] = field(default_factory=list)

    # What this domain should NOT learn (prevents over-scoping)
    forbidden_categories: List[str] = field(default_factory=list)

    # Heuristic templates for this domain
    heuristic_templates: List[str] = field(default_factory=list)

    # Outcome fields to track
    outcome_fields: List[str] = field(default_factory=list)

    # Minimum occurrences before creating a heuristic
    min_occurrences: int = 3

    # Custom metadata schema
    metadata_schema: Dict[str, str] = field(default_factory=dict)

    def to_scope(self, agent_name: str) -> MemoryScope:
        """Convert to MemoryScope for ALMA integration."""
        return MemoryScope(
            agent_name=agent_name,
            can_learn=self.learnable_categories,
            cannot_learn=self.forbidden_categories,
            min_occurrences_for_heuristic=self.min_occurrences,
        )

    def to_prompt(self) -> str:
        """Format schema for documentation."""
        lines = [
            f"## Memory Schema: {self.domain}",
            self.description,
            "",
            "### Learnable Categories:",
        ]
        for cat in self.learnable_categories:
            lines.append(f"- {cat}")

        if self.heuristic_templates:
            lines.append("")
            lines.append("### Heuristic Templates:")
            for t in self.heuristic_templates:
                lines.append(f"- {t}")

        return "\n".join(lines)


@dataclass
class Agent:
    """
    The executor that operates within a setting, given a context.

    Agents start "dumb" but get smarter via memory injections.
    They use tools to accomplish tasks and log reflections post-run.
    """

    name: str
    role: str
    description: str
    memory_schema: MemorySchema

    # Personality/style traits
    traits: List[str] = field(default_factory=list)

    # Default behaviors
    default_actions: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format agent identity for system prompt."""
        lines = [
            f"## You are {self.name}",
            f"**Role**: {self.role}",
            "",
            self.description,
        ]

        if self.traits:
            lines.append("")
            lines.append("### Traits:")
            for t in self.traits:
                lines.append(f"- {t}")

        return "\n".join(lines)


@dataclass
class RunResult:
    """Result of a harness run."""

    success: bool
    output: Any
    reflections: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class Harness:
    """
    The complete harness that orchestrates Setting + Context + Agent + Memory.

    This is the main interface for running learning agents across any domain.

    Usage:
        harness = Harness(setting, agent, alma)
        result = harness.run(context)
    """

    def __init__(
        self,
        setting: Setting,
        agent: Agent,
        alma: Any,  # ALMA instance
    ):
        """
        Initialize harness.

        Args:
            setting: The fixed environment
            agent: The executor
            alma: ALMA instance for memory management
        """
        self.setting = setting
        self.agent = agent
        self.alma = alma

    def build_prompt(
        self,
        context: Context,
        memory_slice: Optional[MemorySlice] = None,
    ) -> str:
        """
        Build the complete prompt for a run.

        Combines: Agent identity + Setting + Memory + Context
        """
        sections = [
            self.agent.to_prompt(),
            "",
            self.setting.to_prompt(),
        ]

        # Inject memory if available
        if memory_slice and memory_slice.total_items > 0:
            sections.append("")
            sections.append("## Relevant Memory (from past runs)")
            sections.append(memory_slice.to_prompt())

        sections.append("")
        sections.append(context.to_prompt())

        return "\n".join(sections)

    def pre_run(self, context: Context) -> MemorySlice:
        """
        Pre-run: Retrieve relevant memory for this task.

        Returns memory slice to inject into prompt.
        """
        return self.alma.retrieve(
            task=context.task,
            agent=self.agent.name,
            user_id=context.user_id,
            top_k=5,
        )

    def post_run(
        self,
        context: Context,
        result: RunResult,
    ):
        """
        Post-run: Update memory based on outcome.

        Logs the outcome and potentially creates new heuristics.
        """
        # Learn from the outcome
        self.alma.learn(
            agent=self.agent.name,
            task=context.task,
            outcome="success" if result.success else "failure",
            strategy_used=(
                ", ".join(result.tools_used) if result.tools_used else "direct"
            ),
            duration_ms=result.duration_ms,
            error_message=result.error,
            feedback="; ".join(result.reflections) if result.reflections else None,
        )

    def run(
        self,
        context: Context,
        executor: Optional[Callable[[str], RunResult]] = None,
    ) -> RunResult:
        """
        Execute the full harness flow.

        1. Pre-run: Retrieve relevant memories
        2. Build prompt with injected memory
        3. Execute (via provided executor or return prompt for external use)
        4. Post-run: Update memory with outcome

        Args:
            context: The task context for this run
            executor: Optional function that takes prompt and returns RunResult.
                      If not provided, returns a RunResult with the built prompt.

        Returns:
            RunResult with output or prompt
        """
        import time

        start_time = time.time()

        # 1. Pre-run: Get relevant memories
        memory_slice = self.pre_run(context)

        # 2. Build prompt
        prompt = self.build_prompt(context, memory_slice)

        # 3. Execute
        if executor:
            result = executor(prompt)
            result.duration_ms = int((time.time() - start_time) * 1000)

            # 4. Post-run: Update memory
            self.post_run(context, result)

            return result
        else:
            # Return prompt for external execution
            return RunResult(
                success=True,
                output=prompt,
                duration_ms=int((time.time() - start_time) * 1000),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics for this agent."""
        return self.alma.get_stats(agent=self.agent.name)
