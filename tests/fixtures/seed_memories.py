"""
Memory Seeding Utilities for ALMA Tests.

Provides functions to populate ALMA with test data for
integration and E2E testing scenarios.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)


def seed_helena_memories(
    storage: StorageBackend,
    project_id: str = "test-project",
    count: int = 10,
) -> Dict[str, int]:
    """
    Seed storage with Helena-specific memories.

    Args:
        storage: Storage backend to populate
        project_id: Project ID for memories
        count: Base count for each memory type

    Returns:
        Dict with counts of seeded items
    """
    now = datetime.now(timezone.utc)
    counts = {
        "heuristics": 0,
        "outcomes": 0,
        "domain_knowledge": 0,
        "anti_patterns": 0,
    }

    # Helena heuristics - UI testing strategies
    helena_heuristics = [
        (
            "form with multiple required fields",
            "validate each field individually before full form submit",
        ),
        ("modal dialog testing", "wait for animation to complete before interacting"),
        ("dropdown selection", "use keyboard navigation for more reliable selection"),
        ("table with pagination", "verify total count before and after page change"),
        ("file upload component", "test with various file types and sizes"),
        ("responsive layout testing", "test at exact breakpoint boundaries"),
        ("accessibility testing", "use axe-core for automated WCAG checks"),
        ("visual regression", "capture baseline before any style changes"),
        ("infinite scroll", "use intersection observer detection for loading"),
        ("toast notifications", "verify aria-live announcements"),
    ]

    for i, (condition, strategy) in enumerate(helena_heuristics[:count]):
        h = Heuristic(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id=project_id,
            condition=condition,
            strategy=strategy,
            confidence=0.75 + (i % 25) / 100,
            occurrence_count=5 + i,
            success_count=4 + i,
            last_validated=now - timedelta(days=i),
            created_at=now - timedelta(days=30 + i),
        )
        storage.save_heuristic(h)
        counts["heuristics"] += 1

    # Helena domain knowledge
    helena_knowledge = [
        ("selector_patterns", "data-testid selectors are most stable for testing"),
        ("selector_patterns", "avoid using CSS classes for test selectors"),
        ("accessibility_testing", "all form inputs must have associated labels"),
        ("accessibility_testing", "color alone should not convey information"),
        ("form_testing", "client-side validation should match server validation"),
        ("ui_component_patterns", "modals should trap focus when open"),
        ("testing_strategies", "wait for network idle before taking screenshots"),
    ]

    for domain, fact in helena_knowledge[:count]:
        dk = DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id=project_id,
            domain=domain,
            fact=fact,
            source="test_generation",
            confidence=0.9,
            last_verified=now,
        )
        storage.save_domain_knowledge(dk)
        counts["domain_knowledge"] += 1

    # Helena anti-patterns
    helena_anti_patterns = [
        (
            "Using fixed sleep() for async waits",
            "Causes flaky tests, doesn't adapt to load",
            "Use explicit waits with conditions",
        ),
        (
            "Selecting by element position",
            "Breaks when layout changes",
            "Use stable selectors like data-testid",
        ),
        (
            "Not waiting for animations",
            "Clicks may miss moving elements",
            "Wait for animation-complete or reduced-motion",
        ),
    ]

    for pattern, why_bad, alternative in helena_anti_patterns:
        ap = AntiPattern(
            id=str(uuid.uuid4()),
            agent="helena",
            project_id=project_id,
            pattern=pattern,
            why_bad=why_bad,
            better_alternative=alternative,
            occurrence_count=3,
            last_seen=now,
            created_at=now - timedelta(days=10),
        )
        storage.save_anti_pattern(ap)
        counts["anti_patterns"] += 1

    return counts


def seed_victor_memories(
    storage: StorageBackend,
    project_id: str = "test-project",
    count: int = 10,
) -> Dict[str, int]:
    """
    Seed storage with Victor-specific memories.

    Args:
        storage: Storage backend to populate
        project_id: Project ID for memories
        count: Base count for each memory type

    Returns:
        Dict with counts of seeded items
    """
    now = datetime.now(timezone.utc)
    counts = {
        "heuristics": 0,
        "outcomes": 0,
        "domain_knowledge": 0,
        "anti_patterns": 0,
    }

    # Victor heuristics - API testing strategies
    victor_heuristics = [
        ("API endpoint testing", "check authentication before payload validation"),
        ("POST request with JSON", "validate Content-Type header is set"),
        ("paginated endpoint", "test edge cases: page 0, negative, beyond total"),
        ("file upload API", "test with multipart/form-data and verify streaming"),
        ("rate-limited endpoint", "include retry-after header in response"),
        ("webhook endpoint", "verify signature before processing payload"),
        ("batch operation", "return detailed status for each item"),
        ("search endpoint", "use query string for GET, body for complex queries"),
        ("async operation", "return 202 Accepted with job ID for tracking"),
        ("error response", "include request_id for debugging"),
    ]

    for i, (condition, strategy) in enumerate(victor_heuristics[:count]):
        h = Heuristic(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id=project_id,
            condition=condition,
            strategy=strategy,
            confidence=0.80 + (i % 20) / 100,
            occurrence_count=8 + i,
            success_count=7 + i,
            last_validated=now - timedelta(days=i),
            created_at=now - timedelta(days=25 + i),
        )
        storage.save_heuristic(h)
        counts["heuristics"] += 1

    # Victor domain knowledge
    victor_knowledge = [
        ("error_handling", "Always return structured error responses with error codes"),
        ("error_handling", "Include request_id in all error responses"),
        ("authentication_patterns", "JWT tokens should expire within 1 hour"),
        ("authentication_patterns", "Refresh tokens should be single-use"),
        ("api_design_patterns", "Use 201 Created with Location header for POST"),
        ("api_design_patterns", "Return 204 No Content for successful DELETE"),
        ("database_query_patterns", "Use cursor-based pagination for large datasets"),
        ("performance_optimization", "Enable gzip compression for JSON responses"),
    ]

    for domain, fact in victor_knowledge[:count]:
        dk = DomainKnowledge(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id=project_id,
            domain=domain,
            fact=fact,
            source="api_design_review",
            confidence=0.95,
            last_verified=now,
        )
        storage.save_domain_knowledge(dk)
        counts["domain_knowledge"] += 1

    # Victor anti-patterns
    victor_anti_patterns = [
        (
            "Testing with hardcoded auth tokens",
            "Tokens expire, tests become brittle",
            "Generate fresh tokens in test setup",
        ),
        (
            "Not validating response schema",
            "API changes go undetected",
            "Use JSON Schema or Pydantic for validation",
        ),
        (
            "Ignoring rate limit headers",
            "Tests may fail unpredictably",
            "Parse and respect Retry-After header",
        ),
    ]

    for pattern, why_bad, alternative in victor_anti_patterns:
        ap = AntiPattern(
            id=str(uuid.uuid4()),
            agent="victor",
            project_id=project_id,
            pattern=pattern,
            why_bad=why_bad,
            better_alternative=alternative,
            occurrence_count=4,
            last_seen=now,
            created_at=now - timedelta(days=8),
        )
        storage.save_anti_pattern(ap)
        counts["anti_patterns"] += 1

    return counts


def seed_all_memories(
    storage: StorageBackend,
    project_id: str = "test-project",
    count: int = 10,
    include_preferences: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Seed storage with memories for all agents.

    Args:
        storage: Storage backend to populate
        project_id: Project ID for memories
        count: Base count for each memory type per agent
        include_preferences: Whether to include user preferences

    Returns:
        Dict with counts per agent
    """
    results = {
        "helena": seed_helena_memories(storage, project_id, count),
        "victor": seed_victor_memories(storage, project_id, count),
    }

    if include_preferences:
        now = datetime.now(timezone.utc)
        preferences = [
            ("code_style", "No emojis in documentation"),
            ("code_style", "Use 4 spaces for indentation"),
            ("testing", "Prefer data-testid over CSS selectors"),
            ("testing", "Always include negative test cases"),
            ("communication", "Keep test names descriptive"),
        ]

        for category, preference in preferences:
            pref = UserPreference(
                id=str(uuid.uuid4()),
                user_id="test-user",
                category=category,
                preference=preference,
                source="explicit_instruction",
                confidence=1.0,
                timestamp=now,
            )
            storage.save_user_preference(pref)

        results["preferences"] = {"count": len(preferences)}

    return results


def create_learning_progression(
    storage: StorageBackend,
    agent: str,
    project_id: str,
    task_type: str,
    iterations: int = 15,
) -> List[str]:
    """
    Create a sequence of outcomes that demonstrate learning.

    Simulates an agent getting better at a task over time.

    Args:
        storage: Storage backend
        agent: Agent name
        project_id: Project ID
        task_type: Type of task
        iterations: Number of outcomes to create

    Returns:
        List of outcome IDs created
    """
    now = datetime.now(timezone.utc)
    outcome_ids = []

    # Learning curve: starts with failures, improves over time
    strategies = [
        "naive approach - no prior knowledge",
        "basic approach - some structure",
        "improved approach - learned from failures",
        "refined approach - applying heuristics",
        "optimized approach - consistent success",
    ]

    for i in range(iterations):
        # Success rate improves over iterations
        success_probability = 0.3 + (i / iterations) * 0.6
        success = (i / iterations) > (1 - success_probability) or i > iterations * 0.7

        strategy_idx = min(i // 3, len(strategies) - 1)

        outcome = Outcome(
            id=str(uuid.uuid4()),
            agent=agent,
            project_id=project_id,
            task_type=task_type,
            task_description=f"Task iteration {i + 1} for {task_type}",
            success=success,
            strategy_used=strategies[strategy_idx],
            duration_ms=1000 - (i * 30) if success else 2000,  # Faster with experience
            error_message=None if success else f"Attempt {i + 1} failed",
            timestamp=now - timedelta(hours=iterations - i),
        )

        storage.save_outcome(outcome)
        outcome_ids.append(outcome.id)

    return outcome_ids


def create_failure_pattern(
    storage: StorageBackend,
    agent: str,
    project_id: str,
    pattern: str,
    occurrences: int = 5,
) -> str:
    """
    Create a sequence of failures that should trigger anti-pattern creation.

    Args:
        storage: Storage backend
        agent: Agent name
        project_id: Project ID
        pattern: The problematic pattern being used
        occurrences: Number of failures to create

    Returns:
        Pattern identifier for verification
    """
    now = datetime.now(timezone.utc)
    pattern_id = str(uuid.uuid4())[:8]

    for i in range(occurrences):
        outcome = Outcome(
            id=str(uuid.uuid4()),
            agent=agent,
            project_id=project_id,
            task_type="pattern_test",
            task_description=f"Test with problematic pattern: {pattern}",
            success=False,
            strategy_used=pattern,
            error_message=f"Failed due to: {pattern}",
            timestamp=now - timedelta(hours=occurrences - i),
            metadata={"pattern_id": pattern_id},
        )

        storage.save_outcome(outcome)

    return pattern_id


def create_scope_violation_scenario(
    storage: StorageBackend,
    agent: str,
    project_id: str,
    forbidden_domain: str,
) -> Dict[str, Any]:
    """
    Create test data for scope violation testing.

    Creates knowledge in a domain the agent shouldn't have access to.

    Args:
        storage: Storage backend
        agent: Agent name (will be different from knowledge owner)
        project_id: Project ID
        forbidden_domain: Domain that should be forbidden for agent

    Returns:
        Dict with test scenario details
    """
    now = datetime.now(timezone.utc)

    # Create knowledge owned by a different agent
    other_agent = "system" if agent != "system" else "admin"

    dk = DomainKnowledge(
        id=str(uuid.uuid4()),
        agent=other_agent,
        project_id=project_id,
        domain=forbidden_domain,
        fact=f"This is {forbidden_domain} knowledge that {agent} should not access",
        source="scope_test",
        confidence=1.0,
        last_verified=now,
    )
    storage.save_domain_knowledge(dk)

    return {
        "knowledge_id": dk.id,
        "forbidden_domain": forbidden_domain,
        "owner_agent": other_agent,
        "requesting_agent": agent,
    }
