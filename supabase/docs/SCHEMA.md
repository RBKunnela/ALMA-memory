# ALMA-Memory Database Schema Documentation

**Generated:** 2026-01-28
**Version:** 1.0.0

## Table of Contents

1. [Overview](#overview)
2. [Data Models](#data-models)
3. [PostgreSQL Schema](#postgresql-schema)
4. [SQLite Schema](#sqlite-schema)
5. [Azure Cosmos DB Schema](#azure-cosmos-db-schema)
6. [File-Based Storage Schema](#file-based-storage-schema)
7. [Index Definitions](#index-definitions)
8. [Schema Diagrams](#schema-diagrams)

---

## Overview

ALMA-Memory supports four storage backends, each implementing the same abstract interface (`StorageBackend`) but with different underlying schemas:

| Backend | Use Case | Vector Search | Schema Type |
|---------|----------|---------------|-------------|
| PostgreSQL | Production | pgvector (native) | Relational (SQL) |
| SQLite | Development/Testing | FAISS (external) | Relational (SQL) |
| Azure Cosmos DB | Cloud Production | DiskANN (native) | Document (NoSQL) |
| File-Based | Testing/Fallback | None | JSON Files |

**Source:** `alma/storage/base.py:21-29`

---

## Data Models

All backends store the same five memory types defined in `alma/types.py`:

### 1. Heuristic

A learned rule representing successful strategies.

**Source:** `alma/types.py:44-68`

| Field | Type | Description | Nullable |
|-------|------|-------------|----------|
| `id` | TEXT | Primary key (UUID) | No |
| `agent` | TEXT | Agent that created the heuristic | No |
| `project_id` | TEXT | Project scope identifier | No |
| `condition` | TEXT | When condition that triggers this strategy | No |
| `strategy` | TEXT | What strategy to apply | No |
| `confidence` | REAL | Success confidence (0.0-1.0) | No (default: 0.0) |
| `occurrence_count` | INTEGER | Total times pattern observed | No (default: 0) |
| `success_count` | INTEGER | Successful occurrences | No (default: 0) |
| `last_validated` | TIMESTAMP | Last time heuristic was validated | Yes |
| `created_at` | TIMESTAMP | Creation timestamp | Yes |
| `metadata` | JSON/JSONB | Additional key-value data | Yes |
| `embedding` | VECTOR/BLOB | Semantic embedding vector | Yes |

### 2. Outcome

Raw task execution record (success or failure with context).

**Source:** `alma/types.py:71-91`

| Field | Type | Description | Nullable |
|-------|------|-------------|----------|
| `id` | TEXT | Primary key (UUID) | No |
| `agent` | TEXT | Agent that executed the task | No |
| `project_id` | TEXT | Project scope identifier | No |
| `task_type` | TEXT | Category of task (e.g., "api_validation") | Yes |
| `task_description` | TEXT | Full task description | No |
| `success` | BOOLEAN | Whether task succeeded | No (default: false) |
| `strategy_used` | TEXT | Strategy applied during execution | Yes |
| `duration_ms` | INTEGER | Execution time in milliseconds | Yes |
| `error_message` | TEXT | Error details if failed | Yes |
| `user_feedback` | TEXT | Optional user feedback | Yes |
| `timestamp` | TIMESTAMP | Execution timestamp | Yes |
| `metadata` | JSON/JSONB | Additional context | Yes |
| `embedding` | VECTOR/BLOB | Semantic embedding vector | Yes |

### 3. UserPreference

Remembered user constraints and communication preferences.

**Source:** `alma/types.py:94-108`

| Field | Type | Description | Nullable |
|-------|------|-------------|----------|
| `id` | TEXT | Primary key (UUID) | No |
| `user_id` | TEXT | User identifier (partition key) | No |
| `category` | TEXT | Preference category | Yes |
| `preference` | TEXT | The actual preference text | No |
| `source` | TEXT | How preference was learned | Yes |
| `confidence` | REAL | Confidence level (0.0-1.0) | No (default: 1.0) |
| `timestamp` | TIMESTAMP | When preference was recorded | Yes |
| `metadata` | JSON/JSONB | Additional context | Yes |

**Note:** UserPreference does NOT have an embedding field or vector search capability.

### 4. DomainKnowledge

Accumulated domain-specific facts (facts, not strategies).

**Source:** `alma/types.py:111-127`

| Field | Type | Description | Nullable |
|-------|------|-------------|----------|
| `id` | TEXT | Primary key (UUID) | No |
| `agent` | TEXT | Agent that learned the fact | No |
| `project_id` | TEXT | Project scope identifier | No |
| `domain` | TEXT | Knowledge domain (e.g., "authentication") | Yes |
| `fact` | TEXT | The learned fact | No |
| `source` | TEXT | Where the fact came from | Yes |
| `confidence` | REAL | Confidence level (0.0-1.0) | No (default: 1.0) |
| `last_verified` | TIMESTAMP | Last verification timestamp | Yes |
| `metadata` | JSON/JSONB | Additional context | Yes |
| `embedding` | VECTOR/BLOB | Semantic embedding vector | Yes |

### 5. AntiPattern

Learned patterns to avoid (from validated failures).

**Source:** `alma/types.py:130-147`

| Field | Type | Description | Nullable |
|-------|------|-------------|----------|
| `id` | TEXT | Primary key (UUID) | No |
| `agent` | TEXT | Agent that identified the pattern | No |
| `project_id` | TEXT | Project scope identifier | No |
| `pattern` | TEXT | The anti-pattern description | No |
| `why_bad` | TEXT | Explanation of why it's bad | Yes |
| `better_alternative` | TEXT | Recommended alternative | Yes |
| `occurrence_count` | INTEGER | Times this was observed | No (default: 1) |
| `last_seen` | TIMESTAMP | Last observation timestamp | Yes |
| `created_at` | TIMESTAMP | Creation timestamp | Yes |
| `metadata` | JSON/JSONB | Additional context | Yes |
| `embedding` | VECTOR/BLOB | Semantic embedding vector | Yes |

---

## PostgreSQL Schema

**Source:** `alma/storage/postgresql.py:52-293`

PostgreSQL backend uses native tables with pgvector extension for vector similarity search.

### Table: `alma_heuristics`

```sql
CREATE TABLE IF NOT EXISTS {schema}.alma_heuristics (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    occurrence_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_validated TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR({embedding_dim})  -- or BYTEA if pgvector unavailable
);
```

**Source:** `alma/storage/postgresql.py:172-187`

### Table: `alma_outcomes`

```sql
CREATE TABLE IF NOT EXISTS {schema}.alma_outcomes (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_type TEXT,
    task_description TEXT NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    strategy_used TEXT,
    duration_ms INTEGER,
    error_message TEXT,
    user_feedback TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR({embedding_dim})
);
```

**Source:** `alma/storage/postgresql.py:194-209`

### Table: `alma_preferences`

```sql
CREATE TABLE IF NOT EXISTS {schema}.alma_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT,
    preference TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);
```

**Source:** `alma/storage/postgresql.py:221-232`

**Note:** No embedding column - preferences use exact matching only.

### Table: `alma_domain_knowledge`

```sql
CREATE TABLE IF NOT EXISTS {schema}.alma_domain_knowledge (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    domain TEXT,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    last_verified TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR({embedding_dim})
);
```

**Source:** `alma/storage/postgresql.py:238-252`

### Table: `alma_anti_patterns`

```sql
CREATE TABLE IF NOT EXISTS {schema}.alma_anti_patterns (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    pattern TEXT NOT NULL,
    why_bad TEXT,
    better_alternative TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR({embedding_dim})
);
```

**Source:** `alma/storage/postgresql.py:259-273`

---

## SQLite Schema

**Source:** `alma/storage/sqlite_local.py:37-224`

SQLite backend uses relational tables with a separate embeddings table and FAISS for vector search.

### Table: `heuristics`

```sql
CREATE TABLE IF NOT EXISTS heuristics (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    occurrence_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_validated TEXT,
    created_at TEXT,
    metadata TEXT
);
```

**Source:** `alma/storage/sqlite_local.py:107-121`

### Table: `outcomes`

```sql
CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_type TEXT,
    task_description TEXT NOT NULL,
    success INTEGER DEFAULT 0,
    strategy_used TEXT,
    duration_ms INTEGER,
    error_message TEXT,
    user_feedback TEXT,
    timestamp TEXT,
    metadata TEXT
);
```

**Source:** `alma/storage/sqlite_local.py:128-143`

**Note:** `success` stored as INTEGER (0/1) instead of BOOLEAN.

### Table: `preferences`

```sql
CREATE TABLE IF NOT EXISTS preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT,
    preference TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    timestamp TEXT,
    metadata TEXT
);
```

**Source:** `alma/storage/sqlite_local.py:154-165`

### Table: `domain_knowledge`

```sql
CREATE TABLE IF NOT EXISTS domain_knowledge (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    domain TEXT,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    last_verified TEXT,
    metadata TEXT
);
```

**Source:** `alma/storage/sqlite_local.py:173-184`

### Table: `anti_patterns`

```sql
CREATE TABLE IF NOT EXISTS anti_patterns (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    pattern TEXT NOT NULL,
    why_bad TEXT,
    better_alternative TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_seen TEXT,
    created_at TEXT,
    metadata TEXT
);
```

**Source:** `alma/storage/sqlite_local.py:191-203`

### Table: `embeddings`

SQLite separates embeddings into their own table for FAISS indexing:

```sql
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_type TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE(memory_type, memory_id)
);
```

**Source:** `alma/storage/sqlite_local.py:211-219`

---

## Azure Cosmos DB Schema

**Source:** `alma/storage/azure_cosmos.py:57-178`

Azure Cosmos DB uses document collections (containers) with DiskANN vector indexing.

### Container: `alma-heuristics`

**Partition Key:** `/project_id`

```json
{
  "id": "string",
  "agent": "string",
  "project_id": "string",
  "condition": "string",
  "strategy": "string",
  "confidence": 0.0,
  "occurrence_count": 0,
  "success_count": 0,
  "last_validated": "ISO8601 datetime",
  "created_at": "ISO8601 datetime",
  "metadata": {},
  "embedding": [float...],
  "type": "heuristic"
}
```

**Source:** `alma/storage/azure_cosmos.py:240-266`

### Container: `alma-outcomes`

**Partition Key:** `/project_id`

```json
{
  "id": "string",
  "agent": "string",
  "project_id": "string",
  "task_type": "string",
  "task_description": "string",
  "success": true,
  "strategy_used": "string",
  "duration_ms": 0,
  "error_message": "string",
  "user_feedback": "string",
  "timestamp": "ISO8601 datetime",
  "metadata": {},
  "embedding": [float...],
  "type": "outcome"
}
```

**Source:** `alma/storage/azure_cosmos.py:268-291`

### Container: `alma-preferences`

**Partition Key:** `/user_id`

```json
{
  "id": "string",
  "user_id": "string",
  "category": "string",
  "preference": "string",
  "source": "string",
  "confidence": 1.0,
  "timestamp": "ISO8601 datetime",
  "metadata": {},
  "type": "preference"
}
```

**Source:** `alma/storage/azure_cosmos.py:293-313`

**Note:** No embedding field for preferences.

### Container: `alma-knowledge`

**Partition Key:** `/project_id`

```json
{
  "id": "string",
  "agent": "string",
  "project_id": "string",
  "domain": "string",
  "fact": "string",
  "source": "string",
  "confidence": 1.0,
  "last_verified": "ISO8601 datetime",
  "metadata": {},
  "embedding": [float...],
  "type": "domain_knowledge"
}
```

**Source:** `alma/storage/azure_cosmos.py:315-337`

### Container: `alma-antipatterns`

**Partition Key:** `/project_id`

```json
{
  "id": "string",
  "agent": "string",
  "project_id": "string",
  "pattern": "string",
  "why_bad": "string",
  "better_alternative": "string",
  "occurrence_count": 1,
  "last_seen": "ISO8601 datetime",
  "created_at": "ISO8601 datetime",
  "metadata": {},
  "embedding": [float...],
  "type": "anti_pattern"
}
```

**Source:** `alma/storage/azure_cosmos.py:339-364`

---

## File-Based Storage Schema

**Source:** `alma/storage/file_based.py:28-73`

File-based storage uses JSON files in a directory structure.

### Directory Structure

```
{storage_dir}/
├── heuristics.json       # Array of heuristic objects
├── outcomes.json         # Array of outcome objects
├── preferences.json      # Array of preference objects
├── domain_knowledge.json # Array of knowledge objects
└── anti_patterns.json    # Array of anti-pattern objects
```

### JSON Format

Each file contains an array of objects matching the data model fields.

---

## Index Definitions

### PostgreSQL Indexes

**Source:** `alma/storage/postgresql.py:188-291`

| Table | Index Name | Columns | Type |
|-------|------------|---------|------|
| `alma_heuristics` | `idx_heuristics_project_agent` | `(project_id, agent)` | B-tree |
| `alma_heuristics` | `idx_alma_heuristics_embedding` | `(embedding)` | IVFFlat (pgvector) |
| `alma_outcomes` | `idx_outcomes_project_agent` | `(project_id, agent)` | B-tree |
| `alma_outcomes` | `idx_outcomes_task_type` | `(project_id, agent, task_type)` | B-tree |
| `alma_outcomes` | `idx_alma_outcomes_embedding` | `(embedding)` | IVFFlat (pgvector) |
| `alma_preferences` | `idx_preferences_user` | `(user_id)` | B-tree |
| `alma_domain_knowledge` | `idx_domain_knowledge_project_agent` | `(project_id, agent)` | B-tree |
| `alma_domain_knowledge` | `idx_alma_domain_knowledge_embedding` | `(embedding)` | IVFFlat (pgvector) |
| `alma_anti_patterns` | `idx_anti_patterns_project_agent` | `(project_id, agent)` | B-tree |
| `alma_anti_patterns` | `idx_alma_anti_patterns_embedding` | `(embedding)` | IVFFlat (pgvector) |

**Vector Index Configuration:**
```sql
CREATE INDEX IF NOT EXISTS idx_{table}_embedding
ON {schema}.{table}
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Source:** `alma/storage/postgresql.py:280-291`

### SQLite Indexes

**Source:** `alma/storage/sqlite_local.py:122-223`

| Table | Index Name | Columns |
|-------|------------|---------|
| `heuristics` | `idx_heuristics_project_agent` | `(project_id, agent)` |
| `outcomes` | `idx_outcomes_project_agent` | `(project_id, agent)` |
| `outcomes` | `idx_outcomes_task_type` | `(project_id, agent, task_type)` |
| `preferences` | `idx_preferences_user` | `(user_id)` |
| `domain_knowledge` | `idx_domain_knowledge_project_agent` | `(project_id, agent)` |
| `anti_patterns` | `idx_anti_patterns_project_agent` | `(project_id, agent)` |
| `embeddings` | `idx_embeddings_type` | `(memory_type)` |

### Azure Cosmos DB Indexing

**Source:** `alma/storage/azure_cosmos.py:182-232`

Cosmos DB uses automatic indexing with specific configurations:

**Standard Indexing Policy:**
```json
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [{"path": "/*"}],
  "excludedPaths": [
    {"path": "/\"_etag\"/?"},
    {"path": "/embedding/*"}
  ]
}
```

**Vector Embedding Policy (for containers with embeddings):**
```json
{
  "vectorEmbeddings": [
    {
      "path": "/embedding",
      "dataType": "float32",
      "dimensions": 384,
      "distanceFunction": "cosine"
    }
  ]
}
```

---

## Schema Diagrams

### Entity Relationship Diagram (ASCII)

```
+------------------+       +------------------+       +------------------+
|   Heuristic      |       |    Outcome       |       | DomainKnowledge  |
+------------------+       +------------------+       +------------------+
| PK: id           |       | PK: id           |       | PK: id           |
| FK: project_id   |       | FK: project_id   |       | FK: project_id   |
|     agent        |       |     agent        |       |     agent        |
|     condition    |       |     task_type    |       |     domain       |
|     strategy     |       |     task_desc    |       |     fact         |
|     confidence   |       |     success      |       |     source       |
|     occurrence   |       |     strategy     |       |     confidence   |
|     success_cnt  |       |     duration_ms  |       |     last_verify  |
|     last_valid   |       |     error_msg    |       |     metadata     |
|     created_at   |       |     user_feedbk  |       |     embedding[]  |
|     metadata     |       |     timestamp    |       +------------------+
|     embedding[]  |       |     metadata     |
+------------------+       |     embedding[]  |
                          +------------------+

+------------------+       +------------------+
|  UserPreference  |       |   AntiPattern    |
+------------------+       +------------------+
| PK: id           |       | PK: id           |
| FK: user_id      |       | FK: project_id   |
|     category     |       |     agent        |
|     preference   |       |     pattern      |
|     source       |       |     why_bad      |
|     confidence   |       |     better_alt   |
|     timestamp    |       |     occurrence   |
|     metadata     |       |     last_seen    |
+------------------+       |     created_at   |
(no embedding)             |     metadata     |
                          |     embedding[]  |
                          +------------------+
```

### Partition Strategy Diagram

```
PostgreSQL / SQLite:
+---------------------------------------+
|              Database                 |
|  +----------------+  +-------------+  |
|  | project_A      |  | project_B   |  |
|  | - heuristics   |  | - heuristics|  |
|  | - outcomes     |  | - outcomes  |  |
|  | - knowledge    |  | - knowledge |  |
|  | - antipatterns |  | - antipattn |  |
|  +----------------+  +-------------+  |
|                                       |
|  +----------------------------------+ |
|  |       preferences (by user_id)   | |
|  +----------------------------------+ |
+---------------------------------------+

Azure Cosmos DB:
+-------------------------------------------+
|           alma-memory (database)          |
|  +----------------+  +-----------------+  |
|  | alma-heuristics|  | alma-outcomes   |  |
|  | partition:     |  | partition:      |  |
|  | /project_id    |  | /project_id     |  |
|  +----------------+  +-----------------+  |
|  +----------------+  +-----------------+  |
|  | alma-knowledge |  | alma-antipattrns|  |
|  | partition:     |  | partition:      |  |
|  | /project_id    |  | /project_id     |  |
|  +----------------+  +-----------------+  |
|  +-------------------------------------+  |
|  |         alma-preferences            |  |
|  |         partition: /user_id         |  |
|  +-------------------------------------+  |
+-------------------------------------------+
```

### Vector Search Flow Diagram

```
                              Query Embedding
                                    |
                                    v
+------------------------------------------------------------------+
|                        Storage Backend                            |
+------------------------------------------------------------------+
                    |                   |                    |
                    v                   v                    v
            +-------------+     +-------------+      +-------------+
            | PostgreSQL  |     |   SQLite    |      | Cosmos DB   |
            | + pgvector  |     | + FAISS     |      | + DiskANN   |
            +-------------+     +-------------+      +-------------+
                    |                   |                    |
                    v                   v                    v
            +-----------+       +-----------+        +-----------+
            | IVFFlat   |       | IndexFlatIP|       | VectorDist|
            | Cosine    |       | Cosine    |        | Cosine    |
            +-----------+       +-----------+        +-----------+
                    |                   |                    |
                    +-------------------+--------------------+
                                        |
                                        v
                              Top-K Similar Results
```

---

## Configuration Reference

**Source:** `.alma/templates/config.yaml.template`

### Embedding Dimensions by Provider

| Provider | Model | Dimensions |
|----------|-------|------------|
| local | all-MiniLM-L6-v2 | 384 |
| local | all-mpnet-base-v2 | 768 |
| azure | text-embedding-3-small | 1536 |
| azure | text-embedding-3-large | 3072 |
| mock | (hash-based) | 384 |

**Important:** The `embedding_dim` configuration must match the model's output dimensions across all backends.

---

*End of Schema Documentation*
