"""
ALMA Agent Integration.

Provides integration hooks for Claude Code agents (Helena, Victor, etc).
"""

from alma.integration.claude_agents import (
    AgentType,
    TaskContext,
    TaskOutcome,
    ClaudeAgentHooks,
    AgentIntegration,
    create_integration,
)
from alma.integration.helena import (
    UITestContext,
    UITestOutcome,
    HelenaHooks,
    create_helena_hooks,
    helena_pre_task,
    helena_post_task,
    HELENA_CATEGORIES,
    HELENA_FORBIDDEN,
)
from alma.integration.victor import (
    APITestContext,
    APITestOutcome,
    VictorHooks,
    create_victor_hooks,
    victor_pre_task,
    victor_post_task,
    VICTOR_CATEGORIES,
    VICTOR_FORBIDDEN,
)

__all__ = [
    # Core Integration
    "AgentType",
    "TaskContext",
    "TaskOutcome",
    "ClaudeAgentHooks",
    "AgentIntegration",
    "create_integration",
    # Helena Integration
    "UITestContext",
    "UITestOutcome",
    "HelenaHooks",
    "create_helena_hooks",
    "helena_pre_task",
    "helena_post_task",
    "HELENA_CATEGORIES",
    "HELENA_FORBIDDEN",
    # Victor Integration
    "APITestContext",
    "APITestOutcome",
    "VictorHooks",
    "create_victor_hooks",
    "victor_pre_task",
    "victor_post_task",
    "VICTOR_CATEGORIES",
    "VICTOR_FORBIDDEN",
]
