"""
ALMA Test Factories.

Provides factory functions for creating test data with sensible defaults.
All factory functions accept keyword arguments to override any field.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

__all__ = [
    "create_test_heuristic",
    "create_test_outcome",
    "create_test_preference",
    "create_test_knowledge",
    "create_test_anti_pattern",
]


def create_test_heuristic(
    id: Optional[str] = None,
    agent: str = "test-agent",
    project_id: str = "test-project",
    condition: str = "test condition",
    strategy: str = "test strategy",
    confidence: float = 0.85,
    occurrence_count: int = 10,
    success_count: int = 8,
    last_validated: Optional[datetime] = None,
    created_at: Optional[datetime] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Heuristic:
    """
    Create a test Heuristic with sensible defaults.

    All parameters can be overridden to customize the test data.

    Example:
        >>> heuristic = create_test_heuristic(agent="helena", confidence=0.95)
        >>> assert heuristic.agent == "helena"
        >>> assert heuristic.confidence == 0.95

    Args:
        id: Unique identifier (auto-generated if not provided)
        agent: Agent name that owns this heuristic
        project_id: Project identifier
        condition: When this heuristic applies
        strategy: What strategy to use
        confidence: Confidence score (0.0-1.0)
        occurrence_count: Number of times this heuristic has been observed
        success_count: Number of successful applications
        last_validated: Last validation timestamp
        created_at: Creation timestamp
        embedding: Optional embedding vector
        metadata: Additional metadata

    Returns:
        A fully populated Heuristic instance
    """
    now = datetime.now(timezone.utc)
    return Heuristic(
        id=id or str(uuid.uuid4()),
        agent=agent,
        project_id=project_id,
        condition=condition,
        strategy=strategy,
        confidence=confidence,
        occurrence_count=occurrence_count,
        success_count=success_count,
        last_validated=last_validated or now,
        created_at=created_at or now - timedelta(days=7),
        embedding=embedding,
        metadata=metadata or {},
    )


def create_test_outcome(
    id: Optional[str] = None,
    agent: str = "test-agent",
    project_id: str = "test-project",
    task_type: str = "test_task",
    task_description: str = "Test task description",
    success: bool = True,
    strategy_used: str = "test strategy",
    duration_ms: Optional[int] = 500,
    error_message: Optional[str] = None,
    user_feedback: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Outcome:
    """
    Create a test Outcome with sensible defaults.

    All parameters can be overridden to customize the test data.

    Example:
        >>> outcome = create_test_outcome(success=False, error_message="Failed")
        >>> assert outcome.success is False
        >>> assert outcome.error_message == "Failed"

    Args:
        id: Unique identifier (auto-generated if not provided)
        agent: Agent name that produced this outcome
        project_id: Project identifier
        task_type: Type of task (e.g., "api_validation")
        task_description: Description of the task
        success: Whether the task succeeded
        strategy_used: Strategy that was applied
        duration_ms: Task duration in milliseconds
        error_message: Error message if failed
        user_feedback: Optional user feedback
        timestamp: When the outcome occurred
        embedding: Optional embedding vector
        metadata: Additional metadata

    Returns:
        A fully populated Outcome instance
    """
    return Outcome(
        id=id or str(uuid.uuid4()),
        agent=agent,
        project_id=project_id,
        task_type=task_type,
        task_description=task_description,
        success=success,
        strategy_used=strategy_used,
        duration_ms=duration_ms,
        error_message=error_message,
        user_feedback=user_feedback,
        timestamp=timestamp or datetime.now(timezone.utc),
        embedding=embedding,
        metadata=metadata or {},
    )


def create_test_preference(
    id: Optional[str] = None,
    user_id: str = "test-user",
    category: str = "code_style",
    preference: str = "Test preference value",
    source: str = "explicit_instruction",
    confidence: float = 1.0,
    timestamp: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> UserPreference:
    """
    Create a test UserPreference with sensible defaults.

    All parameters can be overridden to customize the test data.

    Example:
        >>> pref = create_test_preference(
        ...     category="communication",
        ...     preference="No emojis"
        ... )
        >>> assert pref.category == "communication"

    Args:
        id: Unique identifier (auto-generated if not provided)
        user_id: User identifier
        category: Preference category (e.g., "code_style", "communication")
        preference: The actual preference text
        source: How this preference was learned
        confidence: Confidence in this preference (0.0-1.0)
        timestamp: When the preference was recorded
        metadata: Additional metadata

    Returns:
        A fully populated UserPreference instance
    """
    return UserPreference(
        id=id or str(uuid.uuid4()),
        user_id=user_id,
        category=category,
        preference=preference,
        source=source,
        confidence=confidence,
        timestamp=timestamp or datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def create_test_knowledge(
    id: Optional[str] = None,
    agent: str = "test-agent",
    project_id: str = "test-project",
    domain: str = "test_domain",
    fact: str = "Test domain fact",
    source: str = "test_source",
    confidence: float = 1.0,
    last_verified: Optional[datetime] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DomainKnowledge:
    """
    Create a test DomainKnowledge with sensible defaults.

    All parameters can be overridden to customize the test data.

    Example:
        >>> knowledge = create_test_knowledge(
        ...     domain="authentication",
        ...     fact="JWT tokens expire in 24h"
        ... )
        >>> assert knowledge.domain == "authentication"

    Args:
        id: Unique identifier (auto-generated if not provided)
        agent: Agent name that owns this knowledge
        project_id: Project identifier
        domain: Knowledge domain (e.g., "authentication", "database")
        fact: The factual information
        source: How this knowledge was acquired
        confidence: Confidence in this knowledge (0.0-1.0)
        last_verified: Last verification timestamp
        embedding: Optional embedding vector
        metadata: Additional metadata

    Returns:
        A fully populated DomainKnowledge instance
    """
    return DomainKnowledge(
        id=id or str(uuid.uuid4()),
        agent=agent,
        project_id=project_id,
        domain=domain,
        fact=fact,
        source=source,
        confidence=confidence,
        last_verified=last_verified or datetime.now(timezone.utc),
        embedding=embedding,
        metadata=metadata or {},
    )


def create_test_anti_pattern(
    id: Optional[str] = None,
    agent: str = "test-agent",
    project_id: str = "test-project",
    pattern: str = "Test anti-pattern",
    why_bad: str = "This is why it's bad",
    better_alternative: str = "Do this instead",
    occurrence_count: int = 3,
    last_seen: Optional[datetime] = None,
    created_at: Optional[datetime] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AntiPattern:
    """
    Create a test AntiPattern with sensible defaults.

    All parameters can be overridden to customize the test data.

    Example:
        >>> anti_pattern = create_test_anti_pattern(
        ...     pattern="Using sleep() for waits",
        ...     better_alternative="Use explicit waits"
        ... )
        >>> assert "sleep" in anti_pattern.pattern

    Args:
        id: Unique identifier (auto-generated if not provided)
        agent: Agent name that identified this anti-pattern
        project_id: Project identifier
        pattern: Description of the anti-pattern
        why_bad: Explanation of why this pattern is problematic
        better_alternative: Recommended alternative approach
        occurrence_count: How often this pattern has been observed
        last_seen: Last time this pattern was observed
        created_at: When this anti-pattern was first identified
        embedding: Optional embedding vector
        metadata: Additional metadata

    Returns:
        A fully populated AntiPattern instance
    """
    now = datetime.now(timezone.utc)
    return AntiPattern(
        id=id or str(uuid.uuid4()),
        agent=agent,
        project_id=project_id,
        pattern=pattern,
        why_bad=why_bad,
        better_alternative=better_alternative,
        occurrence_count=occurrence_count,
        last_seen=last_seen or now,
        created_at=created_at or now - timedelta(days=3),
        embedding=embedding,
        metadata=metadata or {},
    )
