"""
ALMA - Agent Learning Memory Architecture

Persistent memory system for AI agents that learn and improve over time
through structured memory layers - without model weight updates.

The Harness Pattern:
    1. Setting - Fixed environment (tools, constraints)
    2. Context - Ephemeral per-run inputs
    3. Agent - The executor with scoped intelligence
    4. Memory Schema - Domain-specific learning structure

This makes any tool-using agent appear to "learn" by injecting relevant
memory slices before each run and updating memory after.
"""

__version__ = "0.2.0"

# Core
from alma.core import ALMA
from alma.types import (
    Heuristic,
    Outcome,
    UserPreference,
    DomainKnowledge,
    AntiPattern,
    MemorySlice,
    MemoryScope,
)

# Harness Pattern
from alma.harness.base import (
    Setting,
    Context,
    Agent,
    MemorySchema,
    Harness,
    Tool,
    ToolType,
    RunResult,
)
from alma.harness.domains import (
    CodingDomain,
    ResearchDomain,
    ContentDomain,
    OperationsDomain,
    create_harness,
)

__all__ = [
    # Core
    "ALMA",
    "Heuristic",
    "Outcome",
    "UserPreference",
    "DomainKnowledge",
    "AntiPattern",
    "MemorySlice",
    "MemoryScope",
    # Harness Pattern
    "Setting",
    "Context",
    "Agent",
    "MemorySchema",
    "Harness",
    "Tool",
    "ToolType",
    "RunResult",
    # Domain Configurations
    "CodingDomain",
    "ResearchDomain",
    "ContentDomain",
    "OperationsDomain",
    "create_harness",
]
