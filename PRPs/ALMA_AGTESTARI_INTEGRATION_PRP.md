# PRP: ALMA Enhancement for AGtestari Workflow Studio Integration

**Version**: 1.1
**Date**: 2026-01-30
**Author**: Claude Opus 4.5 + RBKunnela
**Status**: Draft - Pending Approval
**Project**: ALMA-memory
**Target Integration**: AGtestari AI Workflow Studio

### Collaborators
| Role | Agent | Responsibility |
|------|-------|----------------|
| Architecture | @architect (Aria) | System design, API patterns, cross-backend compatibility |
| Database | @data-analyst (Dana) | Schema design, query optimization, database provisioning |
| Implementation | @dev | Code implementation following this PRP |

### Infrastructure Preferences
| Component | Provider | Notes |
|-----------|----------|-------|
| PostgreSQL Hosting | **Cloudflare** (D1/Hyperdrive) | NOT Supabase |
| Vector Search | pgvector extension | Available on Cloudflare Hyperdrive |
| Blob Storage | Cloudflare R2 | For workflow artifacts |

---

## Goal

Enhance ALMA (Agent Learning Memory Architecture) with a **Workflow Context Layer** that enables:

1. **Workflow-scoped memory** - Memory tied to specific workflow executions, not just agents
2. **Automatic execution checkpointing** - Save state at every node for crash recovery
3. **State reducers for parallel merging** - Deterministic state combination when parallel branches rejoin
4. **Multi-tenant memory hierarchy** - Tenant > Workflow > Run > Node scoping
5. **Artifact-linked memory** - Connect learnings to workflow artifacts (screenshots, reports)

This transforms ALMA from an "agent learning system" to an "enterprise workflow intelligence layer" suitable for AGtestari's production requirements.

---

## Why

### Business Value

| Value | Impact |
|-------|--------|
| **Workflow Intelligence** | Agents learn from workflow context, not just individual tasks |
| **Crash Recovery** | Workflows resume from exact checkpoint after failures |
| **Multi-tenant SaaS** | Enterprise customers get isolated memory per tenant |
| **Audit Compliance** | Full traceability of what agents learned and when |
| **Parallel Execution** | Safe state merging enables parallel node execution |

### Problems Solved

| Problem | Current State | After Enhancement |
|---------|---------------|-------------------|
| Memory scope too broad | Agent sees all project memories | Agent sees only relevant workflow/run memories |
| No crash recovery | Manual session handoff | Automatic checkpoint at every node |
| Parallel state conflicts | Not supported | Reducer functions handle merges |
| No tenant isolation | `project_id` only | Full tenant hierarchy |
| Disconnected artifacts | Artifacts separate from memory | Artifacts linked to learnings |

### Integration with AGtestari

```
AGtestari Orchestrator (Temporal)
         │
         │ on_node_start()  → alma.checkpoint_start(run_id, node_id)
         │ on_node_complete() → alma.checkpoint_complete(run_id, node_id, state)
         │ on_parallel_join() → alma.merge_states(run_id, branch_states)
         │ on_workflow_complete() → alma.learn_from_workflow(run_id)
         │
         ▼
    ALMA Workflow Context Layer
```

---

## What

### User-Visible Behavior

**For AGtestari Workflow Developers:**

```python
from alma import ALMA
from alma.workflow import WorkflowContext, RetrievalScope

# Initialize with workflow context
alma = ALMA.from_config(".alma/config.yaml")
ctx = WorkflowContext(
    tenant_id="acme-corp",
    workflow_id="approval-flow-v2",
    run_id="run-abc123",
)

# Retrieve with workflow scope
memories = alma.retrieve(
    task="Review document for compliance",
    agent="reviewer",
    context=ctx,
    scope=RetrievalScope.WORKFLOW,  # NEW: Only memories from this workflow
)

# Checkpoint at node completion
alma.checkpoint(
    context=ctx,
    node_id="review-node",
    state={"approved": True, "comments": ["Minor edits needed"]},
)

# Learn from workflow outcome (after workflow completes)
alma.learn_from_workflow(
    context=ctx,
    outcome="success",
    artifacts=["report-abc123.pdf"],
)
```

**For AGtestari Orchestrator:**

```python
# Automatic checkpoint integration
@workflow.on_node_complete
async def checkpoint_handler(run_id: str, node_id: str, state: dict):
    await alma.checkpoint(
        context=WorkflowContext(run_id=run_id),
        node_id=node_id,
        state=state,
    )

# Parallel branch merge
@workflow.on_parallel_join
async def merge_handler(run_id: str, branch_states: list[dict]):
    merged = await alma.merge_states(
        context=WorkflowContext(run_id=run_id),
        states=branch_states,
        reducers=workflow.get_reducers(),
    )
    return merged

# Crash recovery
@workflow.on_resume
async def resume_handler(run_id: str):
    checkpoint = await alma.get_latest_checkpoint(run_id)
    return checkpoint.node_id, checkpoint.state
```

### Success Criteria

- [ ] Workflow-scoped retrieval returns only memories from specified workflow
- [ ] Checkpoints survive worker crashes and enable exact resumption
- [ ] Parallel branch states merge correctly using configured reducers
- [ ] Tenant isolation prevents cross-tenant memory access
- [ ] Artifacts are linked to memories and retrievable
- [ ] All existing ALMA tests still pass (backward compatible)
- [ ] Performance: Checkpoint write < 50ms, retrieval < 200ms p95
- [ ] Integration tests pass with mock AGtestari orchestrator

---

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- file: /mnt/d/1.GITHUB/ALMA-memory/alma/types.py
  why: Core memory types (Heuristic, Outcome, etc.) - must extend, not replace

- file: /mnt/d/1.GITHUB/ALMA-memory/alma/storage/base.py
  why: Storage interface - must add new methods while preserving existing

- file: /mnt/d/1.GITHUB/ALMA-memory/alma/retrieval/engine.py
  why: Retrieval pattern - add workflow filtering to existing semantic search

- file: /mnt/d/1.GITHUB/ALMA-memory/alma/session/
  why: Session handoff pattern - checkpoint builds on this concept

- url: https://docs.temporal.io/workflows#state
  why: Temporal's state model - ALMA checkpoints must be compatible

- url: https://docs.langchain.com/oss/python/langgraph/state
  why: LangGraph's reducer pattern - adopt for parallel merge
```

### Current Codebase Tree

```
alma/
├── __init__.py              # Public API exports
├── core.py                  # ALMA main class
├── types.py                 # Memory types (Heuristic, Outcome, etc.)
├── config/
│   ├── __init__.py
│   └── loader.py            # Config loading with Key Vault support
├── retrieval/
│   ├── __init__.py
│   ├── engine.py            # Semantic search + scoring
│   ├── embeddings.py        # Local/Azure embedders
│   ├── scoring.py           # Recency/success weighting
│   └── cache.py             # Query result caching
├── learning/
│   ├── __init__.py
│   ├── protocols.py         # Learning rules
│   ├── validation.py        # Scope checking
│   ├── forgetting.py        # Decay + pruning
│   └── heuristic_extractor.py
├── storage/
│   ├── __init__.py
│   ├── base.py              # Abstract storage interface
│   ├── sqlite_local.py      # SQLite + FAISS
│   └── azure_cosmos.py      # Azure Cosmos DB
├── session/                 # Session handoff (existing)
├── progress/                # Progress tracking (existing)
├── domains/                 # Domain schemas (existing)
├── confidence/              # Strategy confidence (existing)
├── initializer/             # Session initializer (existing)
├── extraction/              # LLM fact extraction (existing)
├── graph/                   # Neo4j graph memory (existing)
├── integration/             # Agent integrations (existing)
├── harness/                 # Harness pattern (existing)
└── mcp/                     # MCP server (existing)
```

### Desired Codebase Tree (New/Modified Files)

```
alma/
├── types.py                 # MODIFY: Add WorkflowContext, Checkpoint, ArtifactRef
├── core.py                  # MODIFY: Add workflow-aware methods
├── workflow/                # NEW: Workflow context layer
│   ├── __init__.py
│   ├── context.py           # WorkflowContext, RetrievalScope enum
│   ├── checkpoint.py        # Checkpoint management
│   ├── reducers.py          # State merge reducers
│   ├── hierarchy.py         # Tenant > Workflow > Run > Node scoping
│   └── artifacts.py         # Artifact linking
├── storage/
│   ├── base.py              # MODIFY: Add checkpoint/artifact methods
│   ├── sqlite_local.py      # MODIFY: Add checkpoint tables
│   └── azure_cosmos.py      # MODIFY: Add checkpoint containers
├── retrieval/
│   └── engine.py            # MODIFY: Add workflow-scoped filtering
└── mcp/
    └── tools.py             # MODIFY: Add workflow MCP tools
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: ALMA uses dataclasses, not Pydantic
# All new types MUST use @dataclass, not BaseModel
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# CRITICAL: Storage backend is abstract - changes must go in ALL backends
# - sqlite_local.py (SQLite + FAISS)
# - azure_cosmos.py (Cosmos DB)
# - postgresql.py (PostgreSQL + pgvector) ← DON'T FORGET THIS ONE!
# - file_based.py (JSON files - low priority, testing only)
# All four must implement new methods

# CRITICAL: Embeddings are optional on memory types
# embedding: Optional[List[float]] = None
# Don't assume embeddings exist

# CRITICAL: ALMA uses UTC timestamps everywhere
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)  # NOT datetime.utcnow()

# CRITICAL: Config uses environment variables with Key Vault fallback
# ${KEYVAULT:secret-name} syntax for secrets

# CRITICAL: MemoryScope dataclass already exists in alma/types.py:22-40
# Use RetrievalScope for the new enum to avoid naming collision!

# GOTCHA: Cosmos DB partition keys must be defined upfront
# New containers need partition key strategy:
#   - checkpoints: /run_id (checkpoints scoped to single run)
#   - workflow_outcomes: /tenant_id (outcomes partitioned by tenant)
#   - artifact_links: /memory_id (artifacts co-located with their memory)

# GOTCHA: FAISS index must be rebuilt if embedding dimensions change
# Don't change embedding dimensions in existing deployments

# GOTCHA: Checkpoint state size can cause storage bloat
# Enforce max state size (default 1MB) - use artifacts for large data

# INFRASTRUCTURE: Use Cloudflare for all database hosting (NOT Supabase)
# - PostgreSQL: Cloudflare Hyperdrive (with pgvector support)
# - SQLite: Cloudflare D1 for edge deployments
# - Blob Storage: Cloudflare R2 for artifacts
# - Database planning: Coordinate with @data-analyst (Dana)
```

---

## Implementation Blueprint

### Phase 1: Core Data Models

#### New Types (alma/types.py additions)

```python
# ADD to alma/types.py
# NOTE: Do NOT name this MemoryScope - that dataclass already exists at line 22-40!

class RetrievalScope(Enum):
    """
    Scope level for memory retrieval filtering.

    Named RetrievalScope (not MemoryScope) to avoid collision with
    existing MemoryScope dataclass which defines agent learning boundaries.
    """
    NODE = "node"           # Only this node's memories
    RUN = "run"             # Only this workflow run
    WORKFLOW = "workflow"   # All runs of this workflow
    AGENT = "agent"         # All workflows for this agent (current behavior)
    TENANT = "tenant"       # All workflows for this tenant
    GLOBAL = "global"       # Platform-wide (admin only)


@dataclass
class WorkflowContext:
    """
    Context for workflow-scoped memory operations.

    Hierarchy: tenant > workflow > run > node
    """
    tenant_id: Optional[str] = None
    workflow_id: Optional[str] = None
    workflow_version: Optional[str] = None
    run_id: Optional[str] = None
    node_id: Optional[str] = None

    def validate(self, require_tenant: bool = False) -> None:
        """
        Validate hierarchy consistency.

        Args:
            require_tenant: If True, raises error when tenant_id is missing.
                          Use for multi-tenant deployments. Can be configured
                          via ALMA_REQUIRE_TENANT environment variable.
        """
        if self.run_id and not self.workflow_id:
            raise ValueError("run_id requires workflow_id")
        if self.node_id and not self.run_id:
            raise ValueError("node_id requires run_id")

        # Enforce tenant in multi-tenant deployments
        if require_tenant and not self.tenant_id:
            raise ValueError(
                "tenant_id required for multi-tenant deployment. "
                "Set ALMA_REQUIRE_TENANT=false to disable this check."
            )

    def to_partition_key(self, scope: 'RetrievalScope') -> str:
        """Generate partition key for given scope."""
        # Implementation details in hierarchy.py


@dataclass
class Checkpoint:
    """
    State checkpoint at a specific point in workflow execution.

    Enables crash recovery by storing state after each node.
    """
    id: str
    run_id: str
    node_id: str
    state: Dict[str, Any]
    state_hash: str  # For detecting changes
    sequence_number: int  # Order within run
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For parallel branches
    branch_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None


@dataclass
class ArtifactRef:
    """
    Reference to a workflow artifact linked to a memory.

    Artifacts themselves stored in blob storage, this is metadata.
    """
    id: str
    artifact_type: str  # "screenshot", "report", "log", etc.
    storage_path: str   # Blob storage path
    content_hash: str   # For integrity verification
    size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowOutcome:
    """
    Aggregated outcome of a complete workflow run.

    Used for learning from entire workflow, not just individual nodes.
    """
    id: str
    tenant_id: str
    workflow_id: str
    workflow_version: str
    run_id: str
    success: bool
    duration_ms: int
    node_count: int
    nodes_succeeded: int
    nodes_failed: int
    error_message: Optional[str] = None
    artifacts: List[ArtifactRef] = field(default_factory=list)
    learnings_extracted: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# UPDATE existing MemorySlice dataclass (alma/types.py:149-216)
# Add workflow_outcomes field to support workflow-level learnings

@dataclass
class MemorySlice:
    """
    A compact, relevant subset of memories for injection into context.
    """
    heuristics: List[Heuristic] = field(default_factory=list)
    outcomes: List[Outcome] = field(default_factory=list)
    preferences: List[UserPreference] = field(default_factory=list)
    domain_knowledge: List[DomainKnowledge] = field(default_factory=list)
    anti_patterns: List[AntiPattern] = field(default_factory=list)
    workflow_outcomes: List[WorkflowOutcome] = field(default_factory=list)  # NEW

    # Retrieval metadata
    query: Optional[str] = None
    agent: Optional[str] = None
    retrieval_time_ms: Optional[int] = None

    def to_prompt(self, max_tokens: int = 2000) -> str:
        """Format memories for injection into agent context."""
        # ... existing implementation ...
        # ADD workflow_outcomes section:
        if self.workflow_outcomes:
            wo_text = "## Past Workflow Patterns\n"
            for wo in self.workflow_outcomes[:3]:
                status = "succeeded" if wo.success else "failed"
                wo_text += f"- {wo.workflow_id}: {status} ({wo.nodes_succeeded}/{wo.node_count} nodes)\n"
            sections.append(wo_text)
        # ...
```

#### State Reducers (alma/workflow/reducers.py)

```python
# NEW FILE: alma/workflow/reducers.py
"""
State reducers for merging parallel branch states.

Inspired by LangGraph's Annotated[type, reducer] pattern.
"""

from typing import Any, Callable, Dict, List, TypeVar
from abc import ABC, abstractmethod

T = TypeVar('T')


class StateReducer(ABC):
    """Abstract base for state reducers."""

    @abstractmethod
    def reduce(self, values: List[Any]) -> Any:
        """Merge multiple values into one."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return reducer name for serialization."""
        pass


class AppendReducer(StateReducer):
    """Append lists together."""

    def reduce(self, values: List[List[T]]) -> List[T]:
        result = []
        for v in values:
            if isinstance(v, list):
                result.extend(v)
            else:
                result.append(v)
        return result

    def get_name(self) -> str:
        return "append"


class MergeDictReducer(StateReducer):
    """Merge dicts with later values winning conflicts."""

    def reduce(self, values: List[Dict]) -> Dict:
        result = {}
        for v in values:
            if isinstance(v, dict):
                result.update(v)
        return result

    def get_name(self) -> str:
        return "merge_dict"


class LastValueReducer(StateReducer):
    """Take the last non-None value."""

    def reduce(self, values: List[Any]) -> Any:
        for v in reversed(values):
            if v is not None:
                return v
        return None

    def get_name(self) -> str:
        return "last_value"


class FirstValueReducer(StateReducer):
    """Take the first non-None value."""

    def reduce(self, values: List[Any]) -> Any:
        for v in values:
            if v is not None:
                return v
        return None

    def get_name(self) -> str:
        return "first_value"


class SumReducer(StateReducer):
    """Sum numeric values."""

    def reduce(self, values: List[float]) -> float:
        return sum(v for v in values if v is not None)

    def get_name(self) -> str:
        return "sum"


class MaxReducer(StateReducer):
    """Take maximum value."""

    def reduce(self, values: List[float]) -> float:
        valid = [v for v in values if v is not None]
        return max(valid) if valid else None

    def get_name(self) -> str:
        return "max"


# Registry of built-in reducers
REDUCER_REGISTRY: Dict[str, StateReducer] = {
    "append": AppendReducer(),
    "merge_dict": MergeDictReducer(),
    "last_value": LastValueReducer(),
    "first_value": FirstValueReducer(),
    "sum": SumReducer(),
    "max": MaxReducer(),
}


@dataclass
class ReducerConfig:
    """Configuration for state field reducers."""
    field_reducers: Dict[str, str] = field(default_factory=dict)
    default_reducer: str = "last_value"

    def get_reducer(self, field_name: str) -> StateReducer:
        """Get reducer for a field."""
        reducer_name = self.field_reducers.get(field_name, self.default_reducer)
        return REDUCER_REGISTRY.get(reducer_name, LastValueReducer())


class StateMerger:
    """
    Merges multiple states using configured reducers.

    Usage:
        merger = StateMerger(ReducerConfig(
            field_reducers={
                "messages": "append",
                "context": "merge_dict",
                "score": "max",
            }
        ))
        merged = merger.merge([state1, state2, state3])
    """

    def __init__(self, config: ReducerConfig):
        self.config = config

    def merge(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple states into one."""
        if not states:
            return {}
        if len(states) == 1:
            return states[0].copy()

        # Collect all keys
        all_keys = set()
        for state in states:
            all_keys.update(state.keys())

        # Merge each key
        result = {}
        for key in all_keys:
            values = [s.get(key) for s in states]
            reducer = self.config.get_reducer(key)
            result[key] = reducer.reduce(values)

        return result
```

### Phase 2: Storage Layer Extensions

#### Storage Interface Updates (alma/storage/base.py additions)

```python
# MODIFY existing read methods to support scope_filter
# This enables workflow-scoped retrieval without breaking existing code

@abstractmethod
def get_heuristics(
    self,
    project_id: str,
    agent: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    top_k: int = 5,
    min_confidence: float = 0.0,
    scope_filter: Optional[Dict[str, Any]] = None,  # NEW - optional for backward compat
) -> List[Heuristic]:
    """Get heuristics with optional scope filtering."""
    pass

@abstractmethod
def get_outcomes(
    self,
    project_id: str,
    agent: Optional[str] = None,
    task_type: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    top_k: int = 5,
    success_only: bool = False,
    scope_filter: Optional[Dict[str, Any]] = None,  # NEW
) -> List[Outcome]:
    """Get outcomes with optional scope filtering."""
    pass

@abstractmethod
def get_domain_knowledge(
    self,
    project_id: str,
    agent: Optional[str] = None,
    domain: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    top_k: int = 5,
    scope_filter: Optional[Dict[str, Any]] = None,  # NEW
) -> List[DomainKnowledge]:
    """Get domain knowledge with optional scope filtering."""
    pass

@abstractmethod
def get_anti_patterns(
    self,
    project_id: str,
    agent: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    top_k: int = 5,
    scope_filter: Optional[Dict[str, Any]] = None,  # NEW
) -> List[AntiPattern]:
    """Get anti-patterns with optional scope filtering."""
    pass
```

```python
# ADD new methods to StorageBackend abstract class

@abstractmethod
def save_checkpoint(self, checkpoint: Checkpoint) -> str:
    """Save a checkpoint, return checkpoint ID."""
    pass

@abstractmethod
def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
    """Get a specific checkpoint."""
    pass

@abstractmethod
def get_latest_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
    """Get most recent checkpoint for a run."""
    pass

@abstractmethod
def get_checkpoints_for_run(
    self,
    run_id: str,
    limit: int = 100,
) -> List[Checkpoint]:
    """Get all checkpoints for a run, ordered by sequence."""
    pass

@abstractmethod
def delete_checkpoints_for_run(self, run_id: str) -> int:
    """Delete all checkpoints for a run (cleanup after completion)."""
    pass

@abstractmethod
def save_workflow_outcome(self, outcome: WorkflowOutcome) -> str:
    """Save a workflow outcome."""
    pass

@abstractmethod
def get_workflow_outcomes(
    self,
    tenant_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    success_only: bool = False,
    top_k: int = 100,
    embedding: Optional[List[float]] = None,
) -> List[WorkflowOutcome]:
    """Get workflow outcomes with optional filtering."""
    pass

@abstractmethod
def link_artifact_to_memory(
    self,
    memory_id: str,
    memory_type: str,
    artifact: ArtifactRef,
) -> None:
    """Link an artifact to a memory item."""
    pass

@abstractmethod
def get_artifacts_for_memory(
    self,
    memory_id: str,
    memory_type: str,
) -> List[ArtifactRef]:
    """Get artifacts linked to a memory."""
    pass
```

#### SQLite Implementation (alma/storage/sqlite_local.py additions)

```python
# ADD new tables to schema

CHECKPOINT_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    branch_id TEXT,
    parent_checkpoint_id TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (parent_checkpoint_id) REFERENCES checkpoints(id)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_checkpoints_branch ON checkpoints(run_id, branch_id);  -- For parallel branch queries
"""

WORKFLOW_OUTCOME_TABLE = """
CREATE TABLE IF NOT EXISTS workflow_outcomes (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    workflow_version TEXT,
    run_id TEXT NOT NULL UNIQUE,
    success INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    nodes_succeeded INTEGER NOT NULL,
    nodes_failed INTEGER NOT NULL,
    error_message TEXT,
    artifacts_json TEXT,
    learnings_extracted INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    embedding BLOB,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_wo_tenant ON workflow_outcomes(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wo_workflow ON workflow_outcomes(workflow_id);
"""

ARTIFACT_LINKS_TABLE = """
CREATE TABLE IF NOT EXISTS artifact_links (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_artifact_memory ON artifact_links(memory_id, memory_type);
"""
```

### Phase 3: Retrieval Engine Updates

#### Workflow-Scoped Retrieval (alma/retrieval/engine.py modifications)

```python
# MODIFY retrieve() method signature and implementation

def retrieve(
    self,
    query: str,
    agent: str,
    project_id: str,
    user_id: Optional[str] = None,
    top_k: int = 5,
    scope: Optional[RetrievalScope] = None,
    bypass_cache: bool = False,
    # NEW parameters
    context: Optional[WorkflowContext] = None,
    include_workflow_outcomes: bool = False,
) -> MemorySlice:
    """
    Retrieve relevant memories for a task.

    NEW: context and scope parameters enable workflow-scoped retrieval.

    Args:
        context: WorkflowContext for scoped retrieval
        scope: RetrievalScope to filter by (default: AGENT for backward compat)
        include_workflow_outcomes: Include learnings from past workflow runs
    """
    # ... existing cache check ...

    # NEW: Build scope filter
    scope_filter = self._build_scope_filter(context, scope or RetrievalScope.AGENT)

    # MODIFY: Pass scope_filter to storage queries
    raw_heuristics = self.storage.get_heuristics(
        project_id=project_id,
        agent=agent,
        embedding=query_embedding,
        top_k=top_k * 2,
        min_confidence=0.0,
        scope_filter=scope_filter,  # NEW
    )

    # ... rest of retrieval logic ...

    # NEW: Include workflow outcomes if requested
    if include_workflow_outcomes and context:
        workflow_outcomes = self.storage.get_workflow_outcomes(
            tenant_id=context.tenant_id,
            workflow_id=context.workflow_id,
            embedding=query_embedding,
            top_k=top_k,
        )
        result.workflow_outcomes = self._extract_top_k(
            self.scorer.score_workflow_outcomes(workflow_outcomes),
            top_k
        )

    return result


def _build_scope_filter(
    self,
    context: Optional[WorkflowContext],
    scope: RetrievalScope,
) -> Dict[str, Any]:
    """Build filter dict for scoped queries."""
    if not context:
        return {}

    filter_dict = {}

    if scope == RetrievalScope.GLOBAL:
        pass  # No filtering
    elif scope == RetrievalScope.TENANT and context.tenant_id:
        filter_dict["tenant_id"] = context.tenant_id
    elif scope == RetrievalScope.AGENT:
        if context.tenant_id:
            filter_dict["tenant_id"] = context.tenant_id
        # agent filtering handled by existing parameter
    elif scope == RetrievalScope.WORKFLOW and context.workflow_id:
        filter_dict["tenant_id"] = context.tenant_id
        filter_dict["workflow_id"] = context.workflow_id
    elif scope == RetrievalScope.RUN and context.run_id:
        filter_dict["run_id"] = context.run_id
    elif scope == RetrievalScope.NODE and context.node_id:
        filter_dict["run_id"] = context.run_id
        filter_dict["node_id"] = context.node_id

    return filter_dict
```

### Phase 4: Checkpoint Manager

#### New File: alma/workflow/checkpoint.py

```python
"""
Checkpoint management for workflow state persistence.

Enables crash recovery by saving state after each node execution.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from alma.types import Checkpoint, WorkflowContext
from alma.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation."""
    checkpoint_id: str
    sequence_number: int
    state_hash: str
    created_at: datetime
    is_new: bool  # False if state unchanged from previous


class CheckpointManager:
    """
    Manages workflow state checkpoints.

    Features:
    - Automatic sequence numbering
    - State change detection (skip if unchanged)
    - Branch tracking for parallel execution
    - Efficient cleanup after workflow completion

    Usage:
        manager = CheckpointManager(storage)

        # Save checkpoint after node
        result = manager.checkpoint(
            context=WorkflowContext(run_id="run-123"),
            node_id="process-node",
            state={"data": "processed"},
        )

        # Resume from crash
        checkpoint = manager.get_resume_point("run-123")
        # Resume from checkpoint.node_id with checkpoint.state
    """

    def __init__(
        self,
        storage: StorageBackend,
        skip_unchanged: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            storage: Storage backend
            skip_unchanged: Skip saving if state unchanged from previous
        """
        self.storage = storage
        self.skip_unchanged = skip_unchanged
        self._sequence_cache: Dict[str, int] = {}  # run_id -> last sequence

    def checkpoint(
        self,
        context: WorkflowContext,
        node_id: str,
        state: Dict[str, Any],
        branch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointResult:
        """
        Save a state checkpoint.

        Args:
            context: Workflow context (must have run_id)
            node_id: ID of completed node
            state: State to checkpoint
            branch_id: Branch ID for parallel execution
            metadata: Additional metadata

        Returns:
            CheckpointResult with details
        """
        if not context.run_id:
            raise ValueError("WorkflowContext must have run_id for checkpointing")

        run_id = context.run_id
        state_json = json.dumps(state, sort_keys=True, default=str)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()[:16]

        # Check if state changed
        if self.skip_unchanged:
            latest = self.storage.get_latest_checkpoint(run_id)
            if latest and latest.state_hash == state_hash:
                logger.debug(f"Skipping checkpoint for {node_id} - state unchanged")
                return CheckpointResult(
                    checkpoint_id=latest.id,
                    sequence_number=latest.sequence_number,
                    state_hash=state_hash,
                    created_at=latest.created_at,
                    is_new=False,
                )

        # Get next sequence number
        sequence = self._get_next_sequence(run_id)

        # Create checkpoint
        checkpoint = Checkpoint(
            id=f"cp_{run_id}_{sequence}",
            run_id=run_id,
            node_id=node_id,
            state=state,
            state_hash=state_hash,
            sequence_number=sequence,
            branch_id=branch_id,
            parent_checkpoint_id=None,  # Set if needed for branch tracking
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        checkpoint_id = self.storage.save_checkpoint(checkpoint)

        logger.info(
            f"Checkpoint saved: run={run_id}, node={node_id}, "
            f"seq={sequence}, hash={state_hash}"
        )

        return CheckpointResult(
            checkpoint_id=checkpoint_id,
            sequence_number=sequence,
            state_hash=state_hash,
            created_at=checkpoint.created_at,
            is_new=True,
        )

    def get_resume_point(self, run_id: str) -> Optional[Checkpoint]:
        """
        Get the checkpoint to resume from after a crash.

        Args:
            run_id: Workflow run ID

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        return self.storage.get_latest_checkpoint(run_id)

    def get_run_history(
        self,
        run_id: str,
        limit: int = 100,
    ) -> List[Checkpoint]:
        """
        Get checkpoint history for a run.

        Args:
            run_id: Workflow run ID
            limit: Max checkpoints to return

        Returns:
            List of checkpoints, ordered by sequence
        """
        return self.storage.get_checkpoints_for_run(run_id, limit)

    def get_branch_checkpoints(
        self,
        run_id: str,
        branch_ids: List[str],
    ) -> Dict[str, Checkpoint]:
        """
        Get latest checkpoint for each parallel branch.

        Args:
            run_id: Workflow run ID
            branch_ids: List of branch IDs

        Returns:
            Dict mapping branch_id to latest checkpoint
        """
        all_checkpoints = self.storage.get_checkpoints_for_run(run_id, 1000)

        result = {}
        for branch_id in branch_ids:
            branch_checkpoints = [
                cp for cp in all_checkpoints
                if cp.branch_id == branch_id
            ]
            if branch_checkpoints:
                # Get latest by sequence
                result[branch_id] = max(
                    branch_checkpoints,
                    key=lambda x: x.sequence_number
                )

        return result

    def cleanup(self, run_id: str) -> int:
        """
        Delete all checkpoints for a completed run.

        Call this after workflow completes successfully to free storage.

        Args:
            run_id: Workflow run ID

        Returns:
            Number of checkpoints deleted
        """
        count = self.storage.delete_checkpoints_for_run(run_id)
        self._sequence_cache.pop(run_id, None)
        logger.info(f"Cleaned up {count} checkpoints for run {run_id}")
        return count

    def _get_next_sequence(self, run_id: str) -> int:
        """Get next sequence number for a run."""
        if run_id in self._sequence_cache:
            self._sequence_cache[run_id] += 1
            return self._sequence_cache[run_id]

        latest = self.storage.get_latest_checkpoint(run_id)
        next_seq = (latest.sequence_number + 1) if latest else 1
        self._sequence_cache[run_id] = next_seq
        return next_seq
```

### Phase 5: Core API Extensions

#### ALMA Core Updates (alma/core.py additions)

```python
# ADD to ALMA class

from alma.workflow.context import WorkflowContext, RetrievalScope
from alma.workflow.checkpoint import CheckpointManager, CheckpointResult
from alma.workflow.reducers import StateMerger, ReducerConfig
from alma.types import Checkpoint, WorkflowOutcome, ArtifactRef


class ALMA:
    """Extended with workflow context methods."""

    def __init__(self, ...):
        # ... existing init ...
        self._checkpoint_manager: Optional[CheckpointManager] = None

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Lazy-initialize checkpoint manager."""
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(self.storage)
        return self._checkpoint_manager

    # ==================== WORKFLOW CONTEXT METHODS ====================

    def checkpoint(
        self,
        context: WorkflowContext,
        node_id: str,
        state: Dict[str, Any],
        branch_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointResult:
        """
        Save a state checkpoint after node completion.

        Call this after each node completes to enable crash recovery.

        Args:
            context: Workflow context (must have run_id)
            node_id: ID of the completed node
            state: Current workflow state to checkpoint
            branch_id: Branch ID for parallel execution
            metadata: Additional checkpoint metadata

        Returns:
            CheckpointResult with checkpoint details
        """
        return self.checkpoint_manager.checkpoint(
            context=context,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            metadata=metadata,
        )

    def get_resume_point(self, run_id: str) -> Optional[Checkpoint]:
        """
        Get checkpoint to resume from after a crash.

        Args:
            run_id: Workflow run ID

        Returns:
            Latest checkpoint or None
        """
        return self.checkpoint_manager.get_resume_point(run_id)

    def merge_states(
        self,
        context: WorkflowContext,
        states: List[Dict[str, Any]],
        reducer_config: Optional[ReducerConfig] = None,
    ) -> Dict[str, Any]:
        """
        Merge parallel branch states using configured reducers.

        Call this when parallel branches rejoin to combine their states.

        Args:
            context: Workflow context
            states: List of states from parallel branches
            reducer_config: Optional reducer configuration

        Returns:
            Merged state
        """
        config = reducer_config or ReducerConfig()
        merger = StateMerger(config)
        merged = merger.merge(states)

        logger.info(
            f"Merged {len(states)} branch states for run {context.run_id}"
        )

        return merged

    def learn_from_workflow(
        self,
        context: WorkflowContext,
        outcome: str,  # "success" or "failure"
        duration_ms: int,
        node_count: int,
        nodes_succeeded: int,
        nodes_failed: int,
        error_message: Optional[str] = None,
        artifacts: Optional[List[ArtifactRef]] = None,
        auto_extract: bool = True,
        cleanup_checkpoints: bool = True,  # NEW: Make cleanup optional
    ) -> WorkflowOutcome:
        """
        Learn from a completed workflow run.

        Call this after workflow completes to:
        1. Store workflow outcome for future retrieval
        2. Extract learnings from successful patterns
        3. Record anti-patterns from failures
        4. Optionally cleanup checkpoints (default: True)

        Args:
            context: Workflow context (must have tenant_id, workflow_id, run_id)
            outcome: "success" or "failure"
            duration_ms: Total workflow duration
            node_count: Total nodes in workflow
            nodes_succeeded: Nodes that succeeded
            nodes_failed: Nodes that failed
            error_message: Error message if failed
            artifacts: Artifacts produced by workflow
            auto_extract: Automatically extract learnings
            cleanup_checkpoints: Delete checkpoints after learning (default True).
                               Set False if you need to preserve checkpoint history
                               for audit or if this is called mid-workflow.

        Returns:
            WorkflowOutcome record
        """
        context.validate()

        workflow_outcome = WorkflowOutcome(
            id=f"wo_{context.run_id}",
            tenant_id=context.tenant_id or "default",
            workflow_id=context.workflow_id,
            workflow_version=context.workflow_version or "1.0",
            run_id=context.run_id,
            success=(outcome == "success"),
            duration_ms=duration_ms,
            node_count=node_count,
            nodes_succeeded=nodes_succeeded,
            nodes_failed=nodes_failed,
            error_message=error_message,
            artifacts=artifacts or [],
        )

        # Generate embedding for semantic search
        outcome_text = (
            f"Workflow {context.workflow_id} "
            f"{'succeeded' if workflow_outcome.success else 'failed'} "
            f"with {nodes_succeeded}/{node_count} nodes completing"
        )
        workflow_outcome.embedding = self.retrieval_engine._get_embedding(outcome_text)

        # Save outcome
        self.storage.save_workflow_outcome(workflow_outcome)

        # Auto-extract learnings if enabled
        if auto_extract and workflow_outcome.success:
            learnings = self._extract_workflow_learnings(context, workflow_outcome)
            workflow_outcome.learnings_extracted = len(learnings)

        # Cleanup checkpoints for completed workflow (optional)
        if cleanup_checkpoints:
            self.checkpoint_manager.cleanup(context.run_id)

        logger.info(
            f"Workflow outcome saved: {context.workflow_id}, "
            f"success={workflow_outcome.success}, "
            f"learnings={workflow_outcome.learnings_extracted}"
        )

        return workflow_outcome

    def link_artifact(
        self,
        memory_id: str,
        memory_type: str,
        artifact: ArtifactRef,
    ) -> None:
        """
        Link an artifact to a memory item.

        Args:
            memory_id: ID of the memory (heuristic, outcome, etc.)
            memory_type: Type of memory ("heuristic", "outcome", etc.)
            artifact: Artifact reference to link
        """
        self.storage.link_artifact_to_memory(
            memory_id=memory_id,
            memory_type=memory_type,
            artifact=artifact,
        )

    def _extract_workflow_learnings(
        self,
        context: WorkflowContext,
        outcome: WorkflowOutcome,
    ) -> List[str]:
        """Extract heuristics from successful workflow patterns."""
        # Implementation: analyze checkpoint history for patterns
        # This could use the LLM extraction module
        checkpoints = self.checkpoint_manager.get_run_history(context.run_id)

        learnings = []
        # ... pattern extraction logic ...

        return learnings
```

---

## Tasks (Ordered by Dependency)

### Task 1: Core Types Extension
**Estimated Complexity: Medium**

```yaml
MODIFY alma/types.py:
  - ADD RetrievalScope enum after MemoryType enum (NOT MemoryScope - that name is taken!)
  - ADD WorkflowContext dataclass
  - ADD Checkpoint dataclass
  - ADD ArtifactRef dataclass
  - ADD WorkflowOutcome dataclass
  - UPDATE MemorySlice to include workflow_outcomes field
  - PRESERVE all existing types unchanged (especially MemoryScope dataclass at line 22-40)

VALIDATION:
  - Run: python -c "from alma.types import RetrievalScope, WorkflowContext, Checkpoint"
  - Run: pytest tests/unit/test_types.py -v
```

### Task 2: State Reducers Module
**Estimated Complexity: Medium**

```yaml
CREATE alma/workflow/__init__.py:
  - Export all public classes

CREATE alma/workflow/reducers.py:
  - Implement StateReducer abstract class
  - Implement AppendReducer, MergeDictReducer, LastValueReducer, etc.
  - Implement ReducerConfig dataclass
  - Implement StateMerger class
  - Include REDUCER_REGISTRY

CREATE tests/unit/test_reducers.py:
  - Test each reducer type
  - Test StateMerger with multiple configs
  - Test edge cases (empty lists, None values)

VALIDATION:
  - Run: pytest tests/unit/test_reducers.py -v
```

### Task 3: Storage Interface Extension
**Estimated Complexity: High**
**Requires: Schema review by @data-analyst (Dana) before implementation**

```yaml
PREREQUISITE:
  - @data-analyst (Dana) reviews and approves table schemas
  - Dana validates index strategies for query patterns
  - Dana confirms partition key choices for Cosmos DB

MODIFY alma/storage/base.py:
  - UPDATE get_heuristics() - add scope_filter: Optional[Dict[str, Any]] = None
  - UPDATE get_outcomes() - add scope_filter: Optional[Dict[str, Any]] = None
  - UPDATE get_domain_knowledge() - add scope_filter: Optional[Dict[str, Any]] = None
  - UPDATE get_anti_patterns() - add scope_filter: Optional[Dict[str, Any]] = None
  - ADD abstract methods for checkpoints
  - ADD abstract methods for workflow outcomes
  - ADD abstract methods for artifact links
  - PRESERVE all existing methods and backward compatibility

MODIFY alma/storage/sqlite_local.py:
  - ADD checkpoint table schema (with branch index!)
  - ADD workflow_outcome table schema
  - ADD artifact_links table schema
  - IMPLEMENT all new abstract methods
  - UPDATE existing read methods to handle scope_filter
  - ADD migration for existing databases (see migrations section below)

MODIFY alma/storage/azure_cosmos.py:
  - ADD checkpoint container (partition key: /run_id)
  - ADD workflow_outcome container (partition key: /tenant_id)
  - ADD artifact_links handling (partition key: /memory_id)
  - IMPLEMENT all new abstract methods
  - UPDATE existing read methods to handle scope_filter

CREATE tests/unit/test_storage_checkpoints.py:
  - Test checkpoint CRUD
  - Test workflow outcome CRUD
  - Test artifact linking
  - Test scope_filter on read methods

VALIDATION:
  - Run: pytest tests/unit/test_storage*.py -v
```

### Task 3.5: PostgreSQL Storage Extension
**Estimated Complexity: Medium**
**IMPORTANT: PostgreSQL is a production backend (1,078 lines) - DO NOT SKIP!**
**Database Planning: @data-analyst (Dana)**
**Hosting: Cloudflare Hyperdrive (NOT Supabase)**

```yaml
PREREQUISITE:
  - @data-analyst (Dana) designs and reviews PostgreSQL schemas
  - Dana provisions database on Cloudflare Hyperdrive
  - Dana validates pgvector extension availability
  - Dana optimizes indexes for expected query patterns

DEPLOYMENT NOTE:
  - Use Cloudflare Hyperdrive for PostgreSQL hosting
  - Cloudflare D1 for SQLite-compatible workloads
  - Cloudflare R2 for artifact blob storage
  - DO NOT use Supabase

MODIFY alma/storage/postgresql.py:
  - UPDATE get_heuristics() - add scope_filter parameter
  - UPDATE get_outcomes() - add scope_filter parameter
  - UPDATE get_domain_knowledge() - add scope_filter parameter
  - UPDATE get_anti_patterns() - add scope_filter parameter
  - ADD checkpoint table schema (same pattern as SQLite)
  - ADD workflow_outcome table schema
  - ADD artifact_links table schema
  - ADD pgvector index for workflow_outcomes.embedding
  - IMPLEMENT all new abstract methods
  - ADD migration for existing databases

Tables to add (follow postgresql.py:171-291 pattern):
  CREATE TABLE IF NOT EXISTS {schema}.alma_checkpoints (
      id TEXT PRIMARY KEY,
      run_id TEXT NOT NULL,
      node_id TEXT NOT NULL,
      state_json JSONB NOT NULL,
      state_hash TEXT NOT NULL,
      sequence_number INTEGER NOT NULL,
      branch_id TEXT,
      parent_checkpoint_id TEXT,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      metadata JSONB,
      FOREIGN KEY (parent_checkpoint_id) REFERENCES {schema}.alma_checkpoints(id)
  );
  CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON {schema}.alma_checkpoints(run_id, sequence_number);
  CREATE INDEX IF NOT EXISTS idx_checkpoints_branch ON {schema}.alma_checkpoints(run_id, branch_id);

  CREATE TABLE IF NOT EXISTS {schema}.alma_workflow_outcomes (
      id TEXT PRIMARY KEY,
      tenant_id TEXT NOT NULL,
      workflow_id TEXT NOT NULL,
      workflow_version TEXT,
      run_id TEXT NOT NULL UNIQUE,
      success BOOLEAN NOT NULL,
      duration_ms INTEGER NOT NULL,
      node_count INTEGER NOT NULL,
      nodes_succeeded INTEGER NOT NULL,
      nodes_failed INTEGER NOT NULL,
      error_message TEXT,
      artifacts_json JSONB,
      learnings_extracted INTEGER DEFAULT 0,
      timestamp TIMESTAMPTZ DEFAULT NOW(),
      embedding VECTOR({embedding_dim}),  -- pgvector
      metadata JSONB
  );
  CREATE INDEX IF NOT EXISTS idx_wo_tenant ON {schema}.alma_workflow_outcomes(tenant_id);
  CREATE INDEX IF NOT EXISTS idx_wo_workflow ON {schema}.alma_workflow_outcomes(workflow_id);

  CREATE TABLE IF NOT EXISTS {schema}.alma_artifact_links (
      id TEXT PRIMARY KEY,
      memory_id TEXT NOT NULL,
      memory_type TEXT NOT NULL,
      artifact_id TEXT NOT NULL,
      artifact_type TEXT NOT NULL,
      storage_path TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      size_bytes INTEGER NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      metadata JSONB
  );
  CREATE INDEX IF NOT EXISTS idx_artifact_memory ON {schema}.alma_artifact_links(memory_id, memory_type);

CREATE tests/unit/test_storage_postgresql.py:
  - Test checkpoint CRUD with PostgreSQL
  - Test workflow outcome CRUD
  - Test pgvector similarity search on workflow_outcomes
  - Test scope_filter on read methods

VALIDATION:
  - Run: pytest tests/unit/test_storage_postgresql.py -v
```

### Task 4: Checkpoint Manager
**Estimated Complexity: Medium**

```yaml
CREATE alma/workflow/checkpoint.py:
  - Implement CheckpointResult dataclass
  - Implement CheckpointManager class
  - Include sequence numbering logic
  - Include state change detection
  - Include branch tracking
  - Include cleanup functionality
  - ADD state size validation (default max 1MB):
      MAX_STATE_SIZE_BYTES = 1_000_000  # Configurable
      if len(state_json) > self.max_state_size:
          raise ValueError("Checkpoint state exceeds limit. Use artifacts for large data.")

CREATE tests/unit/test_checkpoint_manager.py:
  - Test checkpoint creation
  - Test sequence numbering
  - Test skip unchanged
  - Test resume point retrieval
  - Test branch checkpoints
  - Test cleanup
  - Test state size limit enforcement

CREATE tests/unit/test_checkpoint_concurrent.py:
  - Test concurrent checkpoint writes to same run_id
  - Test checkpoint read during write
  - Test sequence number atomicity (no duplicates under load)
  - Test cleanup during active checkpointing
  - Use threading/asyncio to simulate real workflow concurrency

VALIDATION:
  - Run: pytest tests/unit/test_checkpoint_manager.py -v
  - Run: pytest tests/unit/test_checkpoint_concurrent.py -v
```

### Task 5: Retrieval Engine Updates
**Estimated Complexity: Medium**

```yaml
MODIFY alma/retrieval/engine.py:
  - ADD context parameter to retrieve()
  - ADD scope parameter to retrieve()
  - ADD include_workflow_outcomes parameter
  - IMPLEMENT _build_scope_filter() method
  - MODIFY storage calls to use scope_filter
  - PRESERVE backward compatibility (default scope=AGENT)

MODIFY alma/retrieval/scoring.py:
  - ADD score_workflow_outcomes() method

UPDATE tests/integration/test_retrieval.py:
  - Test workflow-scoped retrieval
  - Test different RetrievalScope values
  - Test backward compatibility

VALIDATION:
  - Run: pytest tests/integration/test_retrieval.py -v
```

### Task 6: Core API Extension
**Estimated Complexity: High**

```yaml
MODIFY alma/core.py:
  - ADD checkpoint_manager property (lazy init)
  - ADD checkpoint() method
  - ADD get_resume_point() method
  - ADD merge_states() method
  - ADD learn_from_workflow() method
  - ADD link_artifact() method
  - ADD _extract_workflow_learnings() helper
  - UPDATE retrieve() to accept context parameter

UPDATE alma/__init__.py:
  - Export WorkflowContext, RetrievalScope, Checkpoint, etc.
  - Export CheckpointManager, StateMerger, ReducerConfig

CREATE tests/integration/test_workflow_integration.py:
  - Test full workflow lifecycle
  - Test checkpoint -> crash -> resume
  - Test parallel merge
  - Test learn_from_workflow

VALIDATION:
  - Run: pytest tests/integration/test_workflow_integration.py -v
```

### Task 7: MCP Tools Extension
**Estimated Complexity: Low**

```yaml
MODIFY alma/mcp/tools.py:
  - ADD alma_checkpoint tool
  - ADD alma_get_resume_point tool
  - ADD alma_merge_states tool
  - ADD alma_learn_workflow tool
  - UPDATE alma_retrieve with context/scope parameters

UPDATE tests/unit/test_mcp_tools.py:
  - Test new MCP tools

VALIDATION:
  - Run: pytest tests/unit/test_mcp_tools.py -v
```

### Task 8: Documentation & Examples
**Estimated Complexity: Low**

```yaml
UPDATE README.md:
  - ADD Workflow Context section
  - ADD Checkpoint usage examples
  - ADD State reducers documentation
  - ADD AGtestari integration guide
  - ADD Migration guide for existing deployments

CREATE examples/agtestari_integration.py:
  - Complete example showing all new features
  - Mock AGtestari orchestrator integration

UPDATE docs/architecture/PRD.md:
  - ADD workflow context layer section
  - UPDATE architecture diagram

VALIDATION:
  - Run: python examples/agtestari_integration.py
```

### Task 9: FileBasedStorage Update (Low Priority)
**Estimated Complexity: Low**
**NOTE: FileBasedStorage is for testing/fallback only - NO vector search support**

```yaml
MODIFY alma/storage/file_based.py:
  - ADD checkpoints.json file handling
  - ADD workflow_outcomes.json file handling
  - ADD artifact_links.json file handling
  - IMPLEMENT all new abstract methods (basic, non-optimized)
  - UPDATE read methods to handle scope_filter (simple dict matching)

Files to add to .alma/ directory:
  - checkpoints.json
  - workflow_outcomes.json
  - artifact_links.json

NOTE: This backend does NOT support:
  - Vector similarity search
  - Efficient concurrent access
  - Production workloads

USE CASES:
  - Unit tests that don't need real database
  - Local development without database setup
  - Fallback when other backends unavailable

VALIDATION:
  - Run: pytest tests/unit/test_storage_file_based.py -v
```

---

## Migration Strategy

### Overview

Existing ALMA deployments need schema updates. This section defines the migration approach.

### Migration Principles

1. **Non-destructive**: Never delete existing data automatically
2. **Backward compatible**: Old code continues to work during migration
3. **Rollback-safe**: Migrations can be reversed if issues arise
4. **Versioned**: Track which migrations have been applied

### Migration Files

```
alma/storage/migrations/
├── __init__.py
├── runner.py           # Migration executor
├── 001_add_workflow_columns.py
├── 002_add_checkpoint_tables.py
├── 003_add_workflow_outcome_tables.py
└── 004_add_artifact_links.py
```

### Migration 001: Add Workflow Columns to Existing Tables

```python
# alma/storage/migrations/001_add_workflow_columns.py
"""
Add tenant_id, workflow_id, run_id, node_id columns to existing memory tables.
All columns nullable for backward compatibility.
"""

SQLITE_UP = [
    "ALTER TABLE heuristics ADD COLUMN tenant_id TEXT DEFAULT 'default'",
    "ALTER TABLE heuristics ADD COLUMN workflow_id TEXT",
    "ALTER TABLE heuristics ADD COLUMN run_id TEXT",
    "ALTER TABLE heuristics ADD COLUMN node_id TEXT",
    "ALTER TABLE outcomes ADD COLUMN tenant_id TEXT DEFAULT 'default'",
    "ALTER TABLE outcomes ADD COLUMN workflow_id TEXT",
    "ALTER TABLE outcomes ADD COLUMN run_id TEXT",
    "ALTER TABLE outcomes ADD COLUMN node_id TEXT",
    "ALTER TABLE domain_knowledge ADD COLUMN tenant_id TEXT DEFAULT 'default'",
    "ALTER TABLE domain_knowledge ADD COLUMN workflow_id TEXT",
    "ALTER TABLE anti_patterns ADD COLUMN tenant_id TEXT DEFAULT 'default'",
    "ALTER TABLE anti_patterns ADD COLUMN workflow_id TEXT",
    # Add indexes for scope filtering
    "CREATE INDEX IF NOT EXISTS idx_heuristics_tenant ON heuristics(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_heuristics_workflow ON heuristics(workflow_id)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_tenant ON outcomes(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_workflow ON outcomes(workflow_id)",
]

SQLITE_DOWN = [
    # SQLite doesn't support DROP COLUMN easily, but we document rollback
    "-- Manual: Remove columns tenant_id, workflow_id, run_id, node_id"
]

POSTGRES_UP = [
    "ALTER TABLE alma_heuristics ADD COLUMN IF NOT EXISTS tenant_id TEXT DEFAULT 'default'",
    "ALTER TABLE alma_heuristics ADD COLUMN IF NOT EXISTS workflow_id TEXT",
    "ALTER TABLE alma_heuristics ADD COLUMN IF NOT EXISTS run_id TEXT",
    "ALTER TABLE alma_heuristics ADD COLUMN IF NOT EXISTS node_id TEXT",
    # ... similar for other tables
]

COSMOS_UP = [
    # Cosmos doesn't need schema changes - document-based
    # But we need to update partition key strategy for new containers
]
```

### Running Migrations

```bash
# Check current migration status
python -m alma.storage.migrations.runner status

# Run pending migrations
python -m alma.storage.migrations.runner up

# Rollback last migration (if needed)
python -m alma.storage.migrations.runner down

# Run specific migration
python -m alma.storage.migrations.runner up --to 002
```

### Migration Table

Each backend tracks applied migrations:

```sql
-- SQLite/PostgreSQL
CREATE TABLE IF NOT EXISTS alma_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_name TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Pre-Migration Checklist

- [ ] Backup database before migration
- [ ] Test migration on staging/dev first
- [ ] Verify application can connect during migration
- [ ] Plan rollback procedure
- [ ] Schedule maintenance window for production

---

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
cd /mnt/d/1.GITHUB/ALMA-memory

# Lint all modified/new files
ruff check alma/types.py alma/workflow/ alma/core.py alma/retrieval/engine.py --fix

# Type checking
mypy alma/types.py alma/workflow/ alma/core.py alma/retrieval/engine.py

# Expected: No errors
```

### Level 2: Unit Tests

```bash
# Run unit tests for each component
pytest tests/unit/test_types.py -v
pytest tests/unit/test_reducers.py -v
pytest tests/unit/test_storage_checkpoints.py -v
pytest tests/unit/test_storage_postgresql.py -v  # NEW: PostgreSQL backend
pytest tests/unit/test_checkpoint_manager.py -v
pytest tests/unit/test_checkpoint_concurrent.py -v  # NEW: Concurrency tests
pytest tests/unit/test_mcp_tools.py -v

# All existing tests must still pass
pytest tests/unit/ -v

# Expected: All tests pass
```

### Level 3: Integration Tests

```bash
# Test retrieval with workflow scope
pytest tests/integration/test_retrieval.py -v

# Test full workflow lifecycle
pytest tests/integration/test_workflow_integration.py -v

# Run full test suite
pytest tests/ -v --cov=alma --cov-report=term-missing

# Expected: >80% coverage, all tests pass
```

### Level 4: End-to-End Test

```python
# Run the integration example
python examples/agtestari_integration.py

# Expected output:
# - Checkpoint saved successfully
# - Workflow outcome recorded
# - Learnings extracted
# - Memory scoped correctly
```

---

## Final Validation Checklist

### Types & Core
- [ ] RetrievalScope enum defined (NOT MemoryScope - that name is taken!)
- [ ] WorkflowContext dataclass with validate() supporting require_tenant
- [ ] Checkpoint, ArtifactRef, WorkflowOutcome dataclasses defined
- [ ] MemorySlice updated with workflow_outcomes field
- [ ] State reducers working for parallel merge

### Storage Layer
- [ ] StorageBackend.get_* methods updated with scope_filter parameter
- [ ] SQLite storage supports checkpoints (with branch index)
- [ ] PostgreSQL storage supports checkpoints (with pgvector)
- [ ] Azure Cosmos storage supports checkpoints (with partition keys)
- [ ] FileBasedStorage updated (low priority, testing only)
- [ ] Migration scripts created and tested

### Features
- [ ] Checkpoint manager handles sequences correctly
- [ ] Checkpoint manager enforces state size limits
- [ ] Concurrent checkpoint tests pass
- [ ] Retrieval respects workflow scope
- [ ] learn_from_workflow has cleanup_checkpoints parameter
- [ ] Core API exposes all workflow methods
- [ ] MCP tools updated for workflow operations

### Quality
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check alma/`
- [ ] No type errors: `mypy alma/`
- [ ] Backward compatibility maintained
- [ ] Documentation updated with migration guide
- [ ] Example integration script works

---

## Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Checkpoint write | < 50ms p95 | Time from call to storage confirmation |
| Checkpoint read | < 20ms p95 | Time to retrieve latest checkpoint |
| Scoped retrieval | < 200ms p95 | Time for workflow-scoped semantic search |
| State merge | < 10ms | Time to merge 5 branch states |
| Workflow outcome save | < 100ms | Including embedding generation |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Storage schema migration breaks existing data | Medium | High | Version migration scripts, backup before upgrade |
| Checkpoint storage grows unbounded | Medium | Medium | Auto-cleanup after workflow completion, TTL policies |
| Scope filtering degrades query performance | Low | Medium | Index on scope fields, query optimization |
| Reducer conflicts in parallel merge | Low | High | Comprehensive test suite, clear documentation |
| Backward compatibility broken | Low | High | All new parameters optional with defaults |

---

## Approval

- [ ] Architecture approved
- [ ] Security review passed (no new secrets exposure)
- [ ] Performance targets accepted
- [ ] Ready for Phase 1 implementation

**Approver**: ________________
**Date**: ________________

---

## Appendix: AGtestari Integration Example

```python
"""
Example: ALMA integration with AGtestari Workflow Studio
"""

from alma import ALMA
from alma.workflow import WorkflowContext, RetrievalScope, ReducerConfig

# Initialize ALMA
alma = ALMA.from_config(".alma/config.yaml")

# Create workflow context
ctx = WorkflowContext(
    tenant_id="acme-corp",
    workflow_id="document-review-v2",
    workflow_version="2.1.0",
    run_id="run-20260130-abc123",
)

# Node 1: Retrieve relevant memories for review task
memories = alma.retrieve(
    task="Review legal document for compliance issues",
    agent="reviewer",
    project_id="acme-legal",
    context=ctx,
    scope=RetrievalScope.WORKFLOW,  # Only memories from this workflow type
    include_workflow_outcomes=True,  # Include past workflow learnings
)

# Inject memories into agent prompt
prompt = f"""
## Your Task
Review the attached legal document for compliance issues.

## Knowledge from Past Reviews
{memories.to_prompt()}

## Past Workflow Patterns
{memories.workflow_outcomes_to_prompt() if hasattr(memories, 'workflow_outcomes') else ''}
"""

# ... agent executes ...

# Checkpoint after node completion
alma.checkpoint(
    context=ctx,
    node_id="review-node",
    state={
        "document_id": "doc-456",
        "issues_found": ["missing_date", "unclear_terms"],
        "confidence": 0.85,
    },
)

# Parallel branches: approval + legal check
branch_states = [
    {"approved": True, "approver": "manager@acme.com"},
    {"legal_ok": True, "legal_notes": ["Compliant with SOX"]},
]

# Merge parallel results
merged = alma.merge_states(
    context=ctx,
    states=branch_states,
    reducer_config=ReducerConfig(
        field_reducers={
            "legal_notes": "append",  # Combine notes
        },
        default_reducer="last_value",
    ),
)
# merged = {"approved": True, "approver": "...", "legal_ok": True, "legal_notes": [...]}

# After workflow completes
outcome = alma.learn_from_workflow(
    context=ctx,
    outcome="success",
    duration_ms=45000,
    node_count=5,
    nodes_succeeded=5,
    nodes_failed=0,
    auto_extract=True,  # Learn from this successful pattern
)

print(f"Workflow complete. Learnings extracted: {outcome.learnings_extracted}")
```
