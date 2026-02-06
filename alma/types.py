"""
ALMA Memory Types

Defines the core data structures for all memory types.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from alma.workflow.outcomes import WorkflowOutcome


class MemoryType(Enum):
    """Categories of memory that agents can store and retrieve."""

    HEURISTIC = "heuristic"
    OUTCOME = "outcome"
    USER_PREFERENCE = "user_preference"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class MemoryScope:
    """
    Defines what an agent is allowed to learn and share.

    Prevents scope creep by explicitly listing allowed and forbidden domains.
    Supports multi-agent memory sharing through share_with and inherit_from.
    """

    agent_name: str
    can_learn: List[str]
    cannot_learn: List[str]
    share_with: List[str] = field(
        default_factory=list
    )  # Agents that can read this agent's memories
    inherit_from: List[str] = field(
        default_factory=list
    )  # Agents whose memories this agent can read
    min_occurrences_for_heuristic: int = 3

    def is_allowed(self, domain: str) -> bool:
        """Check if learning in this domain is permitted."""
        if domain in self.cannot_learn:
            return False
        if not self.can_learn:  # Empty means all allowed (except cannot_learn)
            return True
        return domain in self.can_learn

    def get_readable_agents(self) -> List[str]:
        """
        Get list of agents whose memories this agent can read.

        Returns:
            List containing this agent's name plus all inherited agents.
        """
        return [self.agent_name] + list(self.inherit_from)

    def can_read_from(self, other_agent: str) -> bool:
        """
        Check if this agent can read memories from another agent.

        Args:
            other_agent: Name of the agent to check

        Returns:
            True if this agent can read from other_agent
        """
        return other_agent == self.agent_name or other_agent in self.inherit_from

    def shares_with(self, other_agent: str) -> bool:
        """
        Check if this agent shares memories with another agent.

        Args:
            other_agent: Name of the agent to check

        Returns:
            True if this agent shares with other_agent
        """
        return other_agent in self.share_with


@dataclass
class Heuristic:
    """
    A learned rule: "When condition X, strategy Y works N% of the time."

    Heuristics are only created after min_occurrences validations.
    """

    id: str
    agent: str
    project_id: str
    condition: str  # "form with multiple required fields"
    strategy: str  # "test happy path first, then individual validation"
    confidence: float  # 0.0 to 1.0
    occurrence_count: int
    success_count: int
    last_validated: datetime
    created_at: datetime
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate from occurrences."""
        if self.occurrence_count == 0:
            return 0.0
        return self.success_count / self.occurrence_count


@dataclass
class Outcome:
    """
    Record of a task execution - success or failure with context.

    Outcomes are raw data that can be consolidated into heuristics.
    """

    id: str
    agent: str
    project_id: str
    task_type: str  # "api_validation", "form_testing", etc.
    task_description: str
    success: bool
    strategy_used: str
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    user_feedback: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """
    A remembered user constraint or communication preference.

    Persists across sessions so users don't repeat themselves.
    """

    id: str
    user_id: str
    category: str  # "communication", "code_style", "workflow"
    preference: str  # "No emojis in documentation"
    source: str  # "explicit_instruction", "inferred_from_correction"
    confidence: float = 1.0  # Lower for inferred preferences
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainKnowledge:
    """
    Accumulated domain-specific facts within agent's scope.

    Different from heuristics - these are facts, not strategies.
    """

    id: str
    agent: str
    project_id: str
    domain: str  # "authentication", "database_schema", etc.
    fact: str  # "Login endpoint uses JWT with 24h expiry"
    source: str  # "code_analysis", "documentation", "user_stated"
    confidence: float = 1.0
    last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AntiPattern:
    """
    What NOT to do - learned from validated failures.

    Helps agents avoid repeating mistakes.
    """

    id: str
    agent: str
    project_id: str
    pattern: str  # "Using fixed sleep() for async waits"
    why_bad: str  # "Causes flaky tests, doesn't adapt to load"
    better_alternative: str  # "Use explicit waits with conditions"
    occurrence_count: int
    last_seen: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySlice:
    """
    A compact, relevant subset of memories for injection into context.

    This is what gets injected per-call - must stay under token budget.
    """

    heuristics: List[Heuristic] = field(default_factory=list)
    outcomes: List[Outcome] = field(default_factory=list)
    preferences: List[UserPreference] = field(default_factory=list)
    domain_knowledge: List[DomainKnowledge] = field(default_factory=list)
    anti_patterns: List[AntiPattern] = field(default_factory=list)

    # Workflow outcomes (v0.6.0+)
    workflow_outcomes: List["WorkflowOutcome"] = field(default_factory=list)

    # Retrieval metadata
    query: Optional[str] = None
    agent: Optional[str] = None
    retrieval_time_ms: Optional[int] = None

    def to_prompt(
        self,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> str:
        """
        Format memories for injection into agent context.

        Respects token budget by prioritizing high-confidence items.
        Uses accurate token counting via tiktoken when available.

        Args:
            max_tokens: Maximum tokens allowed for the output
            model: Optional model name for accurate tokenization
                   (e.g., "gpt-4", "claude-3-sonnet"). If not provided,
                   uses a general-purpose estimation.

        Returns:
            Formatted prompt string, truncated if necessary
        """
        from alma.utils.tokenizer import TokenEstimator

        # Initialize token estimator
        estimator = TokenEstimator(model=model) if model else TokenEstimator()

        sections = []

        if self.heuristics:
            h_text = "## Relevant Strategies\n"
            for h in sorted(self.heuristics, key=lambda x: -x.confidence)[:5]:
                h_text += f"- When: {h.condition}\n  Do: {h.strategy} (confidence: {h.confidence:.0%})\n"
            sections.append(h_text)

        if self.anti_patterns:
            ap_text = "## Avoid These Patterns\n"
            for ap in self.anti_patterns[:3]:
                ap_text += f"- Don't: {ap.pattern}\n  Why: {ap.why_bad}\n  Instead: {ap.better_alternative}\n"
            sections.append(ap_text)

        if self.preferences:
            p_text = "## User Preferences\n"
            for p in self.preferences[:5]:
                p_text += f"- {p.preference}\n"
            sections.append(p_text)

        if self.domain_knowledge:
            dk_text = "## Domain Context\n"
            for dk in self.domain_knowledge[:5]:
                dk_text += f"- {dk.fact}\n"
            sections.append(dk_text)

        result = "\n".join(sections)

        # Use accurate token estimation and truncation
        result = estimator.truncate_to_token_limit(
            text=result,
            max_tokens=max_tokens,
            suffix="\n[truncated]",
        )

        return result

    @property
    def total_items(self) -> int:
        """Total number of memory items in this slice."""
        return (
            len(self.heuristics)
            + len(self.outcomes)
            + len(self.preferences)
            + len(self.domain_knowledge)
            + len(self.anti_patterns)
        )
