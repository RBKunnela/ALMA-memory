# ALMA Workflow Context Schema Design

**Version**: 1.1.0
**Date**: 2026-01-30
**Author**: @data-analyst (Dara)
**Reviewed by**: @architect (Aria)
**Sprint**: 0 (Tasks 0.2, 0.3, 0.4)

---

## Overview

This document defines the database schema for ALMA's Workflow Context Layer, enabling:
- Crash recovery via checkpoints
- Learning from workflow outcomes
- Artifact linking to memories
- Multi-tenant, workflow-scoped memory retrieval

---

## Schema Summary

| Table | Purpose | Backend Support |
|-------|---------|-----------------|
| `alma_checkpoints` | Crash recovery state | PostgreSQL, SQLite |
| `alma_workflow_outcomes` | Workflow learning | PostgreSQL (pgvector), SQLite |
| `alma_artifact_links` | Artifact references | PostgreSQL, SQLite |

---

## Table 1: Checkpoints (`alma_checkpoints`)

### Purpose
Store workflow state after each node completion for crash recovery.

### PostgreSQL Schema
```sql
CREATE TABLE alma_checkpoints (
    -- Primary key
    id TEXT PRIMARY KEY,

    -- Workflow context (required)
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,

    -- State data
    state_json JSONB NOT NULL,
    state_hash TEXT NOT NULL,  -- SHA256 for change detection

    -- Sequencing
    sequence_number INTEGER NOT NULL,

    -- Parallel execution support
    branch_id TEXT,  -- NULL for main branch
    parent_checkpoint_id TEXT REFERENCES alma_checkpoints(id),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Extensibility
    metadata JSONB,

    -- Constraints
    CONSTRAINT uk_checkpoint_run_seq UNIQUE (run_id, sequence_number)
);

-- Indexes
CREATE INDEX idx_checkpoints_run_seq ON alma_checkpoints(run_id, sequence_number DESC);
CREATE INDEX idx_checkpoints_run_branch ON alma_checkpoints(run_id, branch_id) WHERE branch_id IS NOT NULL;
CREATE INDEX idx_checkpoints_created ON alma_checkpoints(created_at);
```

### SQLite Schema
```sql
CREATE TABLE checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    branch_id TEXT,
    parent_checkpoint_id TEXT REFERENCES checkpoints(id),
    created_at TEXT NOT NULL,
    metadata_json TEXT,
    UNIQUE(run_id, sequence_number)
);

CREATE INDEX idx_checkpoints_run_seq ON checkpoints(run_id, sequence_number DESC);
CREATE INDEX idx_checkpoints_run_branch ON checkpoints(run_id, branch_id);
```

### Key Design Decisions

1. **JSONB for state**: Allows flexible state storage without schema changes
2. **state_hash**: Enables skip-if-unchanged optimization (SHA256 truncated to 16 chars)
3. **sequence_number**: Enables ordered retrieval and "latest" queries
4. **branch_id**: Supports parallel execution tracking
5. **Unique constraint**: Prevents duplicate sequence numbers per run

### Access Patterns

| Query | Index Used |
|-------|------------|
| Get latest checkpoint for run | `idx_checkpoints_run_seq` |
| Get all checkpoints for run | `idx_checkpoints_run_seq` |
| Get branch checkpoints | `idx_checkpoints_run_branch` |
| Cleanup old checkpoints | `idx_checkpoints_created` |

---

## Table 2: Workflow Outcomes (`alma_workflow_outcomes`)

### Purpose
Store aggregated outcomes from completed workflow runs for cross-workflow learning.

### PostgreSQL Schema (with pgvector)
```sql
CREATE TABLE alma_workflow_outcomes (
    -- Primary key
    id TEXT PRIMARY KEY,

    -- Multi-tenant hierarchy
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workflow_id TEXT NOT NULL,
    workflow_version TEXT DEFAULT '1.0',
    run_id TEXT NOT NULL UNIQUE,

    -- Outcome data
    success BOOLEAN NOT NULL,
    duration_ms INTEGER NOT NULL,

    -- Node statistics
    node_count INTEGER NOT NULL,
    nodes_succeeded INTEGER NOT NULL DEFAULT 0,
    nodes_failed INTEGER NOT NULL DEFAULT 0,

    -- Error tracking
    error_message TEXT,

    -- Artifacts (JSON array)
    artifacts_json JSONB,

    -- Learning metrics
    learnings_extracted INTEGER DEFAULT 0,

    -- Timestamps
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Semantic search (pgvector)
    embedding VECTOR(384),

    -- Extensibility
    metadata JSONB,

    -- Constraints
    CONSTRAINT chk_nodes_count CHECK (nodes_succeeded + nodes_failed <= node_count)
);

-- Indexes
CREATE INDEX idx_wo_tenant ON alma_workflow_outcomes(tenant_id);
CREATE INDEX idx_wo_workflow ON alma_workflow_outcomes(tenant_id, workflow_id);
CREATE INDEX idx_wo_success ON alma_workflow_outcomes(tenant_id, success);
CREATE INDEX idx_wo_timestamp ON alma_workflow_outcomes(timestamp DESC);

-- pgvector index (IVFFlat for approximate nearest neighbor)
CREATE INDEX idx_wo_embedding ON alma_workflow_outcomes
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Key Design Decisions

1. **tenant_id DEFAULT 'default'**: Backward compatible for single-tenant deployments
2. **run_id UNIQUE**: One outcome per workflow run (no duplicates)
3. **Check constraint**: Ensures node counts are consistent
4. **pgvector VECTOR(384)**: Matches all-MiniLM-L6-v2 embedding dimension
5. **IVFFlat index**: Efficient approximate nearest neighbor search

### Access Patterns

| Query | Index Used |
|-------|------------|
| Filter by tenant | `idx_wo_tenant` |
| Filter by workflow | `idx_wo_workflow` |
| Filter successful runs | `idx_wo_success` |
| Recent outcomes | `idx_wo_timestamp` |
| Semantic search | `idx_wo_embedding` (pgvector) |

---

## Table 3: Artifact Links (`alma_artifact_links`)

### Purpose
Link external artifacts (screenshots, reports, logs) to memory items.

### PostgreSQL Schema
```sql
CREATE TABLE alma_artifact_links (
    -- Primary key
    id TEXT PRIMARY KEY,

    -- Link to memory item
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- 'heuristic', 'outcome', etc.

    -- Artifact reference
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,  -- 'screenshot', 'report', 'log'

    -- Storage location (Cloudflare R2)
    storage_path TEXT NOT NULL,

    -- Integrity
    content_hash TEXT NOT NULL,  -- SHA256
    size_bytes INTEGER NOT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Extensibility
    metadata JSONB,

    -- Constraints
    CONSTRAINT chk_size_positive CHECK (size_bytes > 0)
);

-- Indexes
CREATE INDEX idx_artifact_memory ON alma_artifact_links(memory_id, memory_type);
CREATE INDEX idx_artifact_type ON alma_artifact_links(artifact_type);
CREATE INDEX idx_artifact_created ON alma_artifact_links(created_at);
```

### Key Design Decisions

1. **memory_type column**: Supports linking to any memory type (heuristics, outcomes, etc.)
2. **storage_path**: Full path to Cloudflare R2 (e.g., `r2://alma-artifacts/tenant/run/artifact.png`)
3. **content_hash**: SHA256 for integrity verification and deduplication
4. **size_bytes CHECK**: Prevents invalid zero/negative sizes

### Access Patterns

| Query | Index Used |
|-------|------------|
| Get artifacts for memory | `idx_artifact_memory` |
| List by artifact type | `idx_artifact_type` |
| Cleanup old artifacts | `idx_artifact_created` |

---

## Alterations to Existing Tables

### New Columns Added

| Table | New Column | Type | Purpose |
|-------|------------|------|---------|
| `heuristics` | `tenant_id` | TEXT DEFAULT 'default' | Multi-tenant isolation |
| `heuristics` | `workflow_id` | TEXT | Workflow scoping |
| `heuristics` | `run_id` | TEXT | Run scoping |
| `heuristics` | `node_id` | TEXT | Node scoping |
| `outcomes` | `tenant_id` | TEXT DEFAULT 'default' | Multi-tenant isolation |
| `outcomes` | `workflow_id` | TEXT | Workflow scoping |
| `outcomes` | `run_id` | TEXT | Run scoping |
| `outcomes` | `node_id` | TEXT | Node scoping |
| `domain_knowledge` | `tenant_id` | TEXT DEFAULT 'default' | Multi-tenant isolation |
| `domain_knowledge` | `workflow_id` | TEXT | Workflow scoping |
| `anti_patterns` | `tenant_id` | TEXT DEFAULT 'default' | Multi-tenant isolation |
| `anti_patterns` | `workflow_id` | TEXT | Workflow scoping |

### New Indexes

```sql
-- Scope filtering indexes (partial to save space)
CREATE INDEX idx_heuristics_tenant ON alma_heuristics(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_heuristics_workflow ON alma_heuristics(workflow_id) WHERE workflow_id IS NOT NULL;
CREATE INDEX idx_outcomes_tenant ON alma_outcomes(tenant_id) WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_outcomes_workflow ON alma_outcomes(workflow_id) WHERE workflow_id IS NOT NULL;
```

---

## Differences from PRP Proposal

| PRP Proposal | Final Schema | Rationale |
|--------------|--------------|-----------|
| `state TEXT` | `state_json JSONB` (PG) / `state_json TEXT` (SQLite) | JSONB enables query and indexing |
| No check constraint | `CHECK (nodes_succeeded + nodes_failed <= node_count)` | Data integrity |
| No default for tenant_id | `DEFAULT 'default'` | Backward compatibility |
| Basic indexes | Partial indexes + pgvector IVFFlat | Performance optimization |
| No COMMENT ON | Added table comments | Self-documenting schema |

---

## Migration Notes

### Running Migrations

```bash
# Check current version
python -m alma.storage.migrations.runner status

# Apply migration
python -m alma.storage.migrations.runner up

# Rollback (if needed)
python -m alma.storage.migrations.runner down --to 1.0.0
```

### Backward Compatibility

- All new columns are nullable or have defaults
- Existing queries continue to work unchanged
- New tables don't affect existing functionality
- Rollback preserves data in existing tables

---

## Cloudflare Infrastructure

### Hyperdrive (PostgreSQL)

```yaml
# Connection settings for Cloudflare Hyperdrive
postgres:
  host: ${CLOUDFLARE_HYPERDRIVE_HOST}
  port: 5432
  database: alma_memory
  user: ${CLOUDFLARE_HYPERDRIVE_USER}
  password: ${CLOUDFLARE_HYPERDRIVE_PASSWORD}
  ssl_mode: require
  pool_size: 10
```

### R2 (Artifact Storage)

```yaml
# Cloudflare R2 bucket for artifacts
r2:
  bucket: alma-artifacts
  access_key_id: ${CLOUDFLARE_R2_ACCESS_KEY}
  secret_access_key: ${CLOUDFLARE_R2_SECRET_KEY}
  endpoint: https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com
```

---

## Performance Considerations

### Checkpoint Operations

| Operation | Target | Strategy |
|-----------|--------|----------|
| Write checkpoint | < 50ms | Single INSERT, minimal indexes |
| Read latest | < 20ms | Composite index on (run_id, sequence_number DESC) |
| Cleanup run | < 100ms | Batch DELETE by run_id |

### Workflow Outcome Search

| Operation | Target | Strategy |
|-----------|--------|----------|
| Filter by tenant/workflow | < 50ms | B-tree indexes |
| Semantic search | < 200ms | pgvector IVFFlat (lists=100) |

---

*— Dara, arquitetando dados 🗄️*
