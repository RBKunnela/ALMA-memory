"""
ALMA Harness Pattern.

A structured framework for creating learning agents across any domain:
- Setting: Environment, tools, constraints
- Context: Task-specific inputs per run
- Agent: The executor with scoped intelligence
- Memory Schema: Domain-specific persistent learning
"""

from alma.harness.base import (
    Setting,
    Context,
    Agent,
    MemorySchema,
    Harness,
)
from alma.harness.domains import (
    CodingDomain,
    ResearchDomain,
    ContentDomain,
    OperationsDomain,
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
