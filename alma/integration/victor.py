"""
ALMA Victor Integration.

Victor-specific integration for backend/API QA testing with ALMA memory.

Victor specializes in:
- API endpoint testing
- Database validation
- Performance testing
- Health check verification
- Authentication/authorization testing

This module provides Victor-specific memory categories, prompts, and utilities.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from alma.core import ALMA
from alma.types import MemorySlice
from alma.harness.domains import CodingDomain
from alma.integration.claude_agents import (
    ClaudeAgentHooks,
    TaskContext,
    TaskOutcome,
    AgentType,
)

logger = logging.getLogger(__name__)


# Victor's learning categories
VICTOR_CATEGORIES = [
    "api_design_patterns",
    "authentication_patterns",
    "error_handling",
    "performance_optimization",
    "database_query_patterns",
    "caching_strategies",
]

# Categories Victor should NOT learn
VICTOR_FORBIDDEN = [
    "frontend_styling",
    "ui_testing",
    "marketing_content",
]


@dataclass
class APITestContext(TaskContext):
    """
    Victor-specific test context.

    Extends TaskContext with API/backend testing-specific fields.
    """
    endpoint: Optional[str] = None
    method: str = "GET"
    expected_status: Optional[int] = None
    request_body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    is_auth_test: bool = False
    is_performance_test: bool = False
    is_database_test: bool = False

    def __post_init__(self):
        # Ensure agent name is victor
        self.agent_name = "victor"
        # Set default task type if not specified
        if not self.task_type:
            self.task_type = self._infer_task_type()

    def _infer_task_type(self) -> str:
        """Infer task type from context."""
        if self.is_auth_test:
            return "authentication_patterns"
        if self.is_performance_test:
            return "performance_optimization"
        if self.is_database_test:
            return "database_query_patterns"
        if self.endpoint:
            return "api_design_patterns"
        return "api_testing"


@dataclass
class APITestOutcome(TaskOutcome):
    """
    Victor-specific test outcome.

    Extends TaskOutcome with API/backend testing-specific results.
    """
    response_status: Optional[int] = None
    response_time_ms: Optional[int] = None
    response_body: Optional[Dict[str, Any]] = None
    database_queries_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Use response_time_ms as duration if not set
        if self.duration_ms is None and self.response_time_ms is not None:
            self.duration_ms = self.response_time_ms


class VictorHooks(ClaudeAgentHooks):
    """
    Victor-specific integration hooks.

    Extends ClaudeAgentHooks with backend/API testing-specific functionality.
    """

    def __init__(self, alma: ALMA, auto_learn: bool = True):
        """
        Initialize Victor hooks.

        Args:
            alma: ALMA instance
            auto_learn: Whether to automatically learn from outcomes
        """
        harness = CodingDomain.create_victor(alma)
        super().__init__(
            alma=alma,
            agent_type=AgentType.VICTOR,
            harness=harness,
            auto_learn=auto_learn,
        )

    def get_api_patterns(
        self,
        endpoint_type: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get proven API patterns for an endpoint type.

        Args:
            endpoint_type: Type of endpoint (CRUD, auth, search, etc.)
            top_k: Maximum patterns to return

        Returns:
            List of API patterns with success rates
        """
        memories = self.alma.retrieve(
            task=f"API patterns for {endpoint_type} endpoints",
            agent=self.agent_name,
            top_k=top_k,
        )

        patterns = []
        for h in memories.heuristics:
            if "api" in h.condition.lower() or "endpoint" in h.condition.lower():
                patterns.append({
                    "pattern": h.strategy,
                    "condition": h.condition,
                    "confidence": h.confidence,
                    "occurrences": h.occurrence_count,
                })

        return patterns

    def get_error_handling_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get proven error handling patterns.

        Returns strategies for handling API errors, validation, and responses.
        """
        memories = self.alma.retrieve(
            task="error handling patterns validation responses",
            agent=self.agent_name,
            top_k=top_k,
        )

        patterns = []
        for h in memories.heuristics:
            if any(kw in h.condition.lower() for kw in ["error", "exception", "validation"]):
                patterns.append({
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                })

        return patterns

    def get_performance_strategies(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get performance optimization strategies.

        Returns strategies for caching, query optimization, etc.
        """
        memories = self.alma.retrieve(
            task="performance optimization caching query database",
            agent=self.agent_name,
            top_k=top_k,
        )

        strategies = []
        for h in memories.heuristics:
            if any(kw in h.condition.lower() for kw in ["performance", "cache", "query", "slow"]):
                strategies.append({
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                })

        for dk in memories.domain_knowledge:
            if dk.domain in ["performance_optimization", "caching_strategies"]:
                strategies.append({
                    "fact": dk.fact,
                    "source": dk.source,
                })

        return strategies

    def get_auth_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get authentication/authorization testing patterns.

        Returns patterns for testing auth flows, tokens, permissions.
        """
        memories = self.alma.retrieve(
            task="authentication authorization JWT token permission testing",
            agent=self.agent_name,
            top_k=top_k,
        )

        patterns = []
        for h in memories.heuristics:
            if any(kw in h.condition.lower() for kw in ["auth", "token", "permission", "jwt"]):
                patterns.append({
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                })

        return patterns

    def record_api_pattern(
        self,
        endpoint: str,
        method: str,
        pattern_type: str,
        description: str,
        success: bool,
    ) -> bool:
        """
        Record an API pattern for learning.

        Args:
            endpoint: API endpoint tested
            method: HTTP method
            pattern_type: Type of pattern (error_handling, validation, etc.)
            description: Pattern description
            success: Whether the pattern worked

        Returns:
            True if recorded successfully
        """
        fact = f"[{method}] {endpoint}: {description}"
        return self.add_knowledge(
            domain=pattern_type,
            fact=fact,
            source=f"api_test:success={success}",
        )

    def record_performance_metric(
        self,
        endpoint: str,
        response_time_ms: int,
        query_count: int,
        threshold_ms: int = 500,
    ) -> bool:
        """
        Record a performance metric for learning.

        Args:
            endpoint: API endpoint tested
            response_time_ms: Response time in milliseconds
            query_count: Number of database queries
            threshold_ms: Performance threshold

        Returns:
            True if recorded successfully
        """
        is_slow = response_time_ms > threshold_ms
        status = "SLOW" if is_slow else "OK"
        fact = f"{endpoint}: {response_time_ms}ms, {query_count} queries [{status}]"

        return self.add_knowledge(
            domain="performance_optimization",
            fact=fact,
            source=f"performance_test:threshold={threshold_ms}ms",
        )

    def format_api_test_prompt(
        self,
        memories: MemorySlice,
        test_context: APITestContext,
    ) -> str:
        """
        Format memories for Victor's API testing prompt.

        Provides Victor-specific formatting with test context.
        """
        sections = []

        # Base memory formatting
        base_format = self.format_memories_for_prompt(memories)
        if base_format:
            sections.append(base_format)

        # Add test context
        sections.append("\n## Current Test Context")
        sections.append(f"- **Task**: {test_context.task_description}")
        sections.append(f"- **Task Type**: {test_context.task_type}")

        if test_context.endpoint:
            sections.append(f"- **Endpoint**: {test_context.method} {test_context.endpoint}")
        if test_context.expected_status:
            sections.append(f"- **Expected Status**: {test_context.expected_status}")
        if test_context.request_body:
            sections.append(f"- **Request Body**: {len(test_context.request_body)} fields")

        if test_context.is_auth_test:
            sections.append("- **Focus**: Authentication/Authorization")
        if test_context.is_performance_test:
            sections.append("- **Focus**: Performance testing")
        if test_context.is_database_test:
            sections.append("- **Focus**: Database validation")

        return "\n".join(sections)


def create_victor_hooks(alma: ALMA, auto_learn: bool = True) -> VictorHooks:
    """
    Convenience function to create Victor hooks.

    Args:
        alma: ALMA instance
        auto_learn: Whether to automatically learn

    Returns:
        Configured VictorHooks
    """
    return VictorHooks(alma=alma, auto_learn=auto_learn)


def victor_pre_task(
    alma: ALMA,
    task: str,
    endpoint: Optional[str] = None,
    method: str = "GET",
    project_id: str = "default",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function for Victor pre-task hook.

    Quick integration without creating full hooks object.

    Args:
        alma: ALMA instance
        task: Task description
        endpoint: Optional API endpoint
        method: HTTP method
        project_id: Project ID
        top_k: Max memories per type

    Returns:
        Dict with memories and formatted prompt
    """
    hooks = VictorHooks(alma=alma, auto_learn=False)

    context = APITestContext(
        task_description=task,
        task_type="",  # Will be inferred
        agent_name="victor",
        project_id=project_id,
        endpoint=endpoint,
        method=method,
    )

    memories = hooks.pre_task(context, top_k=top_k)
    prompt = hooks.format_api_test_prompt(memories, context)

    return {
        "memories": memories,
        "prompt": prompt,
        "context": context,
    }


def victor_post_task(
    alma: ALMA,
    task: str,
    success: bool,
    strategy_used: str,
    endpoint: Optional[str] = None,
    method: str = "GET",
    response_status: Optional[int] = None,
    response_time_ms: Optional[int] = None,
    project_id: str = "default",
    error_message: Optional[str] = None,
) -> bool:
    """
    Convenience function for Victor post-task hook.

    Quick integration without creating full hooks object.

    Args:
        alma: ALMA instance
        task: Task description
        success: Whether task succeeded
        strategy_used: Strategy used
        endpoint: API endpoint tested
        method: HTTP method
        response_status: HTTP response status
        response_time_ms: Response time
        project_id: Project ID
        error_message: Error if failed

    Returns:
        True if learning was recorded
    """
    hooks = VictorHooks(alma=alma, auto_learn=True)

    context = APITestContext(
        task_description=task,
        task_type="",  # Will be inferred
        agent_name="victor",
        project_id=project_id,
        endpoint=endpoint,
        method=method,
    )

    outcome = APITestOutcome(
        success=success,
        strategy_used=strategy_used,
        response_status=response_status,
        response_time_ms=response_time_ms,
        error_message=error_message,
    )

    return hooks.post_task(context, outcome)
