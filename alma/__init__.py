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

__version__ = "0.4.0"

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

# Progress Tracking (Phase 10)
from alma.progress import (
    WorkItem,
    WorkItemStatus,
    ProgressLog,
    ProgressSummary,
    ProgressTracker,
)

# Session Management (Phase 10)
from alma.session import (
    SessionHandoff,
    SessionContext,
    SessionOutcome,
    SessionManager,
)

# Domain Memory Factory (Phase 10)
from alma.domains import (
    DomainSchema,
    EntityType,
    RelationshipType,
    DomainMemoryFactory,
    get_coding_schema,
    get_research_schema,
    get_sales_schema,
    get_general_schema,
)

# Session Initializer (Phase 11)
from alma.initializer import (
    CodebaseOrientation,
    InitializationResult,
    RulesOfEngagement,
    SessionInitializer,
)

# Confidence Engine (Phase 12)
from alma.confidence import (
    ConfidenceEngine,
    ConfidenceSignal,
    OpportunitySignal,
    RiskSignal,
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
    # Progress Tracking
    "WorkItem",
    "WorkItemStatus",
    "ProgressLog",
    "ProgressSummary",
    "ProgressTracker",
    # Session Management
    "SessionHandoff",
    "SessionContext",
    "SessionOutcome",
    "SessionManager",
    # Domain Memory Factory
    "DomainSchema",
    "EntityType",
    "RelationshipType",
    "DomainMemoryFactory",
    "get_coding_schema",
    "get_research_schema",
    "get_sales_schema",
    "get_general_schema",
    # Session Initializer
    "CodebaseOrientation",
    "InitializationResult",
    "RulesOfEngagement",
    "SessionInitializer",
    # Confidence Engine
    "ConfidenceEngine",
    "ConfidenceSignal",
    "OpportunitySignal",
    "RiskSignal",
]
