# ALMA Enhancement Summary for AGtestari Integration

**Updated**: 2026-01-30 (Architectural Review Applied)

## Quick Reference Card

### What's Being Added

```
ALMA v0.5.0 (Current)              ALMA v0.6.0 (Enhanced)
─────────────────────              ────────────────────────
Memory Types:                      Memory Types:
├── Heuristics                     ├── Heuristics
├── Outcomes                       ├── Outcomes
├── Preferences                    ├── Preferences
├── Domain Knowledge               ├── Domain Knowledge
├── Anti-patterns                  ├── Anti-patterns
└── Graph Memory (Neo4j)           ├── Graph Memory (Neo4j)
                                   ├── Checkpoints (NEW)
                                   ├── WorkflowOutcomes (NEW)
                                   └── ArtifactRefs (NEW)

Scoping:                           Scoping:
└── project_id + agent             ├── tenant_id (NEW)
                                   ├── workflow_id (NEW)
                                   ├── run_id (NEW)
                                   ├── node_id (NEW)
                                   └── RetrievalScope enum (NEW)

NOTE: Named RetrievalScope (not MemoryScope) to avoid
collision with existing MemoryScope dataclass

Features:                          Features:
├── Semantic retrieval             ├── Semantic retrieval
├── Learning protocols             ├── Learning protocols
├── Forgetting/decay               ├── Forgetting/decay
├── Session handoff                ├── Session handoff
├── Progress tracking              ├── Progress tracking
├── LLM extraction                 ├── LLM extraction
├── Confidence engine              ├── Confidence engine
└── MCP server                     ├── MCP server
                                   ├── Checkpoint manager (NEW)
                                   ├── State reducers (NEW)
                                   ├── Workflow learning (NEW)
                                   └── Artifact linking (NEW)
```

---

## New API Methods

### ALMA Core

| Method | Purpose |
|--------|---------|
| `alma.checkpoint(ctx, node_id, state)` | Save state after node completion |
| `alma.get_resume_point(run_id)` | Get checkpoint for crash recovery |
| `alma.merge_states(ctx, states, config)` | Merge parallel branch states |
| `alma.learn_from_workflow(ctx, outcome, ...)` | Learn from completed workflow |
| `alma.link_artifact(memory_id, type, artifact)` | Link artifact to memory |

### Retrieval (Updated)

| Parameter | Purpose |
|-----------|---------|
| `context: WorkflowContext` | Workflow context for scoping |
| `scope: MemoryScope` | Filter level (NODE/RUN/WORKFLOW/AGENT/TENANT/GLOBAL) |
| `include_workflow_outcomes: bool` | Include past workflow learnings |

---

## Retrieval Scope Hierarchy

```
RetrievalScope.GLOBAL     ─── All tenants (admin only)
    │
    └── RetrievalScope.TENANT     ─── All workflows for tenant
            │
            └── RetrievalScope.AGENT      ─── All runs for agent (default)
                    │
                    └── RetrievalScope.WORKFLOW   ─── All runs of this workflow
                            │
                            └── RetrievalScope.RUN        ─── Only this run
                                    │
                                    └── RetrievalScope.NODE       ─── Only this node
```

**NOTE**: Named `RetrievalScope` to avoid collision with existing `MemoryScope` dataclass
(which defines what an agent is *allowed* to learn, not *where* to search).

---

## State Reducers for Parallel Merge

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `append` | Concatenate lists | Messages, logs, notes |
| `merge_dict` | Merge dicts (later wins) | Context, metadata |
| `last_value` | Take last non-None | Single values |
| `first_value` | Take first non-None | Priority values |
| `sum` | Sum numbers | Counters, scores |
| `max` | Take maximum | High scores, limits |

```python
# Example configuration
ReducerConfig(
    field_reducers={
        "messages": "append",      # Combine all messages
        "context": "merge_dict",   # Merge all context
        "total_score": "sum",      # Sum all scores
    },
    default_reducer="last_value",  # Default for other fields
)
```

---

## Implementation Phases

```
Phase 1: Core Types (2 days)
├── RetrievalScope enum (NOT MemoryScope!)
├── WorkflowContext dataclass (with require_tenant validation)
├── Checkpoint dataclass
├── ArtifactRef dataclass
├── WorkflowOutcome dataclass
└── MemorySlice update (add workflow_outcomes field)

Phase 2: State Reducers (1 day)
├── StateReducer abstract class
├── Built-in reducers (6 types)
├── ReducerConfig
└── StateMerger

Phase 3: Storage Extension (3 days)
├── Abstract interface updates (add scope_filter param)
├── SQLite implementation (with branch index)
├── Cosmos DB implementation (with partition keys)
└── Migration scripts

Phase 3.5: PostgreSQL Extension (1.5 days) ← NEW!
├── PostgreSQL checkpoint tables
├── pgvector index for workflow_outcomes
└── scope_filter implementation

Phase 4: Checkpoint Manager (2 days)
├── CheckpointManager class
├── Sequence numbering
├── State change detection
├── State size limits (1MB default)
├── Cleanup automation
└── Concurrent access tests

Phase 5: Retrieval Updates (2 days)
├── Scope filtering
├── Workflow outcome retrieval
└── Backward compatibility

Phase 6: Core API (2 days)
├── New ALMA methods
├── cleanup_checkpoints parameter
├── MCP tools
└── Integration hooks

Phase 7: Testing & Docs (2 days)
├── Unit tests (>80% coverage)
├── Integration tests
├── Concurrent checkpoint tests
├── Migration guide
├── Documentation
└── Example scripts

Phase 8: FileBasedStorage (0.5 days) - Low Priority
└── Basic implementation for testing

Total: ~16 days (increased from 14)
```

---

## File Changes

### New Files
```
alma/workflow/
├── __init__.py
├── context.py           # WorkflowContext, RetrievalScope (NOT MemoryScope!)
├── checkpoint.py        # CheckpointManager
├── reducers.py          # State merge reducers
├── hierarchy.py         # Tenant hierarchy logic
└── artifacts.py         # Artifact linking

alma/storage/migrations/
├── __init__.py
├── runner.py            # Migration executor
├── 001_add_workflow_columns.py
├── 002_add_checkpoint_tables.py
├── 003_add_workflow_outcome_tables.py
└── 004_add_artifact_links.py

tests/unit/
├── test_reducers.py
├── test_checkpoint_manager.py
├── test_checkpoint_concurrent.py    # NEW: Concurrency tests
├── test_storage_checkpoints.py
└── test_storage_postgresql.py       # NEW: PostgreSQL tests

tests/integration/
└── test_workflow_integration.py

examples/
└── agtestari_integration.py
```

### Modified Files
```
alma/types.py                 # Add RetrievalScope, new types, update MemorySlice
alma/core.py                  # Add workflow methods, cleanup_checkpoints param
alma/storage/base.py          # Add scope_filter to read methods, new abstract methods
alma/storage/sqlite_local.py  # Add tables + implementations + branch index
alma/storage/azure_cosmos.py  # Add containers + partition keys + implementations
alma/storage/postgresql.py    # Add tables + pgvector index + implementations
alma/storage/file_based.py    # Add basic implementations (low priority)
alma/retrieval/engine.py      # Add scope filtering with RetrievalScope
alma/retrieval/scoring.py     # Add workflow outcome scoring
alma/mcp/tools.py             # Add workflow tools
alma/__init__.py              # Export new classes
README.md                     # Documentation + migration guide
```

---

## AGtestari Integration Points

```
AGtestari Orchestrator              ALMA Workflow Layer
═══════════════════════             ═══════════════════════

┌─────────────────────┐
│   Workflow Start    │────────────► WorkflowContext created
└─────────────────────┘

┌─────────────────────┐
│   Before Node       │────────────► alma.retrieve(ctx, scope=WORKFLOW)
│                     │              ↳ Returns relevant memories
└─────────────────────┘

┌─────────────────────┐
│   Node Executes     │              (Agent uses memories)
└─────────────────────┘

┌─────────────────────┐
│   After Node        │────────────► alma.checkpoint(ctx, node_id, state)
│                     │              ↳ State saved for recovery
└─────────────────────┘

┌─────────────────────┐
│   Parallel Join     │────────────► alma.merge_states(ctx, branch_states)
│                     │              ↳ Reducers merge state
└─────────────────────┘

┌─────────────────────┐
│   Workflow Complete │────────────► alma.learn_from_workflow(ctx, outcome)
│                     │              ↳ Learnings extracted
│                     │              ↳ Checkpoints cleaned up
└─────────────────────┘

┌─────────────────────┐
│   Worker Crash      │────────────► alma.get_resume_point(run_id)
│   (Recovery)        │              ↳ Resume from last checkpoint
└─────────────────────┘
```

---

## Performance Targets

| Operation | Target | Current |
|-----------|--------|---------|
| Checkpoint write | < 50ms | N/A (new) |
| Checkpoint read | < 20ms | N/A (new) |
| Scoped retrieval | < 200ms | ~150ms (unscoped) |
| State merge (5 branches) | < 10ms | N/A (new) |
| Workflow outcome save | < 100ms | N/A (new) |

---

## Backward Compatibility

All existing code continues to work unchanged:

```python
# OLD CODE - Still works (100% backward compatible)
alma = ALMA.from_config(".alma/config.yaml")
memories = alma.retrieve(
    task="Test login form",
    agent="helena",
    project_id="my-app",
)

# NEW CODE - Optional enhancements
from alma.workflow import WorkflowContext, RetrievalScope

alma = ALMA.from_config(".alma/config.yaml")
memories = alma.retrieve(
    task="Test login form",
    agent="helena",
    project_id="my-app",
    context=WorkflowContext(run_id="run-123"),   # Optional
    scope=RetrievalScope.WORKFLOW,               # Optional (note: RetrievalScope, not MemoryScope!)
)
```

### Migration Required for Existing Databases

Existing deployments need schema migrations:

```bash
# Check migration status
python -m alma.storage.migrations.runner status

# Apply migrations (backup first!)
python -m alma.storage.migrations.runner up
```

---

## Critical Issues Resolved (Architectural Review)

The following issues were identified and addressed in this update:

| Issue | Resolution |
|-------|------------|
| **Naming Collision** | Renamed `MemoryScope` enum to `RetrievalScope` (existing `MemoryScope` dataclass at types.py:22-40) |
| **Missing PostgreSQL** | Added Task 3.5 for PostgreSQL storage extension (1,078-line production backend) |
| **Incomplete Storage Interface** | Added `scope_filter` parameter to all read methods in `StorageBackend` |
| **WorkflowContext Validation** | Added `require_tenant` parameter to `validate()` for multi-tenant enforcement |
| **Missing Branch Index** | Added `idx_checkpoints_branch` for parallel branch queries |
| **Cosmos Partition Keys** | Documented: checkpoints=/run_id, workflow_outcomes=/tenant_id, artifact_links=/memory_id |
| **Checkpoint Cleanup Risk** | Added `cleanup_checkpoints` parameter to `learn_from_workflow()` |
| **Missing Concurrency Tests** | Added `test_checkpoint_concurrent.py` for concurrent access testing |
| **No Migration Strategy** | Added complete migration section with versioned scripts |
| **Missing MemorySlice Update** | Added `workflow_outcomes` field to MemorySlice dataclass |
| **State Size Bloat** | Added configurable state size limits (default 1MB) |
| **FileBasedStorage Omitted** | Added Task 9 for FileBasedStorage (low priority, testing only) |

---

## Next Steps

1. **Review PRP** - `PRPs/ALMA_AGTESTARI_INTEGRATION_PRP.md`
2. **Approve architecture** - Sign off on approach
3. **Start Phase 1** - Core types implementation
4. **Iterate** - Follow validation loop per task

---

**Full PRP Location**: `PRPs/ALMA_AGTESTARI_INTEGRATION_PRP.md`
**Estimated Total Effort**: ~16 days (updated from 14)
**Risk Level**: Medium (backward compatible, well-defined scope)
**Last Updated**: 2026-01-30 (Architectural Review by @architect)
