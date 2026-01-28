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

Testing Support:
    For testing ALMA integrations, use the `alma.testing` module:

        from alma.testing import MockStorage, create_test_heuristic

        def test_my_integration():
            storage = MockStorage()
            heuristic = create_test_heuristic(agent="test-agent")
            storage.save_heuristic(heuristic)

    Available utilities:
    - MockStorage: In-memory storage backend
    - MockEmbedder: Deterministic fake embeddings
    - create_test_heuristic(), create_test_outcome(), etc.
"""

__version__ = "0.5.1"

# Core
# Confidence Engine (Phase 12)
from alma.confidence import (
    ConfidenceEngine,
    ConfidenceSignal,
    OpportunitySignal,
    RiskSignal,
)

# Consolidation Engine (Phase 13)
from alma.consolidation import (
    ConsolidationEngine,
    ConsolidationResult,
)
from alma.core import ALMA

# Domain Memory Factory (Phase 10)
from alma.domains import (
    DomainMemoryFactory,
    DomainSchema,
    EntityType,
    RelationshipType,
    get_coding_schema,
    get_general_schema,
    get_research_schema,
    get_sales_schema,
)

# Event System (Phase 19)
from alma.events import (
    EventEmitter,
    MemoryEvent,
    MemoryEventType,
    WebhookConfig,
    WebhookManager,
    get_emitter,
)

# Exceptions
from alma.exceptions import (
    ALMAError,
    ConfigurationError,
    EmbeddingError,
    ExtractionError,
    RetrievalError,
    ScopeViolationError,
    StorageError,
)

# Harness Pattern
from alma.harness.base import (
    Agent,
    Context,
    Harness,
    MemorySchema,
    RunResult,
    Setting,
    Tool,
    ToolType,
)
from alma.harness.domains import (
    CodingDomain,
    ContentDomain,
    OperationsDomain,
    ResearchDomain,
    create_harness,
)

# Session Initializer (Phase 11)
from alma.initializer import (
    CodebaseOrientation,
    InitializationResult,
    RulesOfEngagement,
    SessionInitializer,
)

# Observability (Phase 20)
from alma.observability import (
    ALMAMetrics,
    configure_observability,
    get_logger,
    get_metrics,
    get_tracer,
    trace_method,
)

# Progress Tracking (Phase 10)
from alma.progress import (
    ProgressLog,
    ProgressSummary,
    ProgressTracker,
    WorkItem,
    WorkItemStatus,
)

# Session Management (Phase 10)
from alma.session import (
    SessionContext,
    SessionHandoff,
    SessionManager,
    SessionOutcome,
)
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemoryScope,
    MemorySlice,
    Outcome,
    UserPreference,
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
    # Consolidation Engine
    "ConsolidationEngine",
    "ConsolidationResult",
    # Event System
    "MemoryEvent",
    "MemoryEventType",
    "EventEmitter",
    "get_emitter",
    "WebhookConfig",
    "WebhookManager",
    # Exceptions
    "ALMAError",
    "ConfigurationError",
    "ScopeViolationError",
    "StorageError",
    "EmbeddingError",
    "RetrievalError",
    "ExtractionError",
    # Observability
    "configure_observability",
    "get_tracer",
    "get_logger",
    "get_metrics",
    "ALMAMetrics",
    "trace_method",
]
