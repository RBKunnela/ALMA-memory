"""
ALMA Harness Pattern.

A structured framework for creating learning agents across any domain:
- Setting: Environment, tools, constraints
- Context: Task-specific inputs per run
- Agent: The executor with scoped intelligence
- Memory Schema: Domain-specific persistent learning
"""

from alma.harness.base import (
    Agent,
    Context,
    Harness,
    MemorySchema,
    Setting,
)
from alma.harness.domains import (
    CodingDomain,
    ContentDomain,
    OperationsDomain,
    ResearchDomain,
)

__all__ = [
    "Setting",
    "Context",
    "Agent",
    "MemorySchema",
    "Harness",
    "CodingDomain",
    "ResearchDomain",
    "ContentDomain",
    "OperationsDomain",
]
