# ALMA-Memory Database Audit Report

**Generated:** 2026-01-28
**Audit Type:** Storage Backend Comparison and Technical Debt Assessment

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Storage Backend Comparison Matrix](#storage-backend-comparison-matrix)
3. [Data Consistency Concerns](#data-consistency-concerns)
4. [Performance Considerations](#performance-considerations)
5. [Migration Path Analysis](#migration-path-analysis)
6. [Vector Search Implementation Differences](#vector-search-implementation-differences)
7. [Technical Debt Analysis](#technical-debt-analysis)
8. [Recommendations](#recommendations)

---

## Executive Summary

ALMA-Memory implements a multi-backend storage architecture supporting PostgreSQL, SQLite, Azure Cosmos DB, and file-based storage. This audit identifies **12 schema inconsistencies**, **8 performance concerns**, and **6 areas of technical debt** that should be addressed for production readiness.

### Key Findings

| Category | Severity | Count |
|----------|----------|-------|
| Schema Inconsistencies | Medium | 12 |
| Missing Indexes | High | 4 |
| Performance Issues | Medium | 8 |
| Technical Debt | Medium | 6 |
| Migration Risks | High | 3 |

---

## Storage Backend Comparison Matrix

### Feature Comparison

| Feature | PostgreSQL | SQLite | Azure Cosmos DB | File-Based |
|---------|------------|--------|-----------------|------------|
| **Vector Search** | Native (pgvector) | External (FAISS) | Native (DiskANN) | None |
| **Connection Pooling** | Yes (psycopg_pool) | No | Yes (SDK) | N/A |
| **Transactions** | Full ACID | Full ACID | Limited (item-level) | None |
| **Scalability** | High | Low | Very High | Very Low |
| **Offline Support** | No | Yes | No | Yes |
| **Schema Migrations** | Manual | Manual | Automatic | N/A |
| **Embedding Storage** | VECTOR type | BLOB (separate table) | Native array | JSON array |
| **Metadata Type** | JSONB | TEXT (JSON string) | Native object | Native object |

### Query Capability Comparison

| Operation | PostgreSQL | SQLite | Azure Cosmos DB | File-Based |
|-----------|------------|--------|-----------------|------------|
| Exact match queries | Optimized | Optimized | Optimized | O(n) scan |
| Range queries | Optimized | Optimized | Optimized | O(n) scan |
| Vector similarity | Native (ORDER BY <=>) | FAISS (app-level) | Native (VectorDistance) | Not supported |
| Full-text search | Not implemented | Not implemented | Not implemented | Not implemented |
| Aggregations | Full SQL | Full SQL | Limited | None |
| Cross-partition queries | N/A | N/A | Expensive | N/A |

### Operational Characteristics

| Metric | PostgreSQL | SQLite | Azure Cosmos DB | File-Based |
|--------|------------|--------|-----------------|------------|
| Setup Complexity | Medium | Low | High | Very Low |
| Maintenance | Medium | Low | Low | Very Low |
| Cost | Medium | Free | High | Free |
| Backup/Restore | Standard tools | File copy | Azure built-in | File copy |
| Multi-instance | Yes | No | Yes | No |

---

## Data Consistency Concerns

### DC-1: Timestamp Type Inconsistency

**Severity:** Medium

**Issue:** Different backends use different timestamp representations.

| Backend | Type | Format |
|---------|------|--------|
| PostgreSQL | TIMESTAMPTZ | Native timezone-aware |
| SQLite | TEXT | ISO8601 string |
| Cosmos DB | String | ISO8601 string |
| File-Based | String | ISO8601 string |

**Code References:**
- PostgreSQL: `alma/storage/postgresql.py:183` - `TIMESTAMPTZ DEFAULT NOW()`
- SQLite: `alma/storage/sqlite_local.py:117` - `TEXT`
- Cosmos DB: `alma/storage/azure_cosmos.py:253-258` - ISO string conversion

**Impact:** Timezone handling may vary between backends. UTC is assumed but not enforced.

---

### DC-2: Boolean Type Inconsistency

**Severity:** Medium

**Issue:** SQLite uses INTEGER (0/1) while other backends use native booleans.

| Backend | Type | Values |
|---------|------|--------|
| PostgreSQL | BOOLEAN | true/false |
| SQLite | INTEGER | 0/1 |
| Cosmos DB | Boolean | true/false |
| File-Based | Boolean | true/false |

**Code References:**
- PostgreSQL: `alma/storage/postgresql.py:201` - `BOOLEAN DEFAULT FALSE`
- SQLite: `alma/storage/sqlite_local.py:135` - `INTEGER DEFAULT 0`

**Impact:** Conversion logic required in SQLite backend (`alma/storage/sqlite_local.py:858`).

---

### DC-3: Metadata Type Inconsistency

**Severity:** Low

**Issue:** PostgreSQL uses JSONB while SQLite uses TEXT.

| Backend | Type | Features |
|---------|------|----------|
| PostgreSQL | JSONB | Indexed, queryable |
| SQLite | TEXT | String only |
| Cosmos DB | Object | Native document |
| File-Based | Object | Native JSON |

**Code References:**
- PostgreSQL: `alma/storage/postgresql.py:184` - `JSONB`
- SQLite: `alma/storage/sqlite_local.py:119` - `TEXT`

**Impact:** PostgreSQL can query inside metadata; SQLite cannot.

---

### DC-4: Embedding Storage Inconsistency

**Severity:** High

**Issue:** Each backend stores embeddings differently.

| Backend | Storage | Format |
|---------|---------|--------|
| PostgreSQL (pgvector) | VECTOR column | `[1.0, 2.0, ...]` string |
| PostgreSQL (fallback) | BYTEA column | numpy bytes |
| SQLite | Separate embeddings table | numpy bytes blob |
| Cosmos DB | Document field | Native array |
| File-Based | Document field | Native array |

**Code References:**
- PostgreSQL: `alma/storage/postgresql.py:295-322` - `_embedding_to_db()`, `_embedding_from_db()`
- SQLite: `alma/storage/sqlite_local.py:210-219` - Separate embeddings table
- Cosmos DB: `alma/storage/azure_cosmos.py:260` - Native array in document

**Impact:** Migration between backends requires embedding format conversion.

---

### DC-5: Partition Key Differences

**Severity:** Medium

**Issue:** Different partition strategies between backends.

| Backend | Strategy |
|---------|----------|
| PostgreSQL | No partitioning (relies on indexes) |
| SQLite | No partitioning |
| Cosmos DB | project_id (heuristics, outcomes, knowledge, antipatterns), user_id (preferences) |
| File-Based | Single file per memory type |

**Code References:**
- Cosmos DB: `alma/storage/azure_cosmos.py:155-179` - Container configs with partition keys

**Impact:** Cosmos DB cross-partition queries are expensive and must enable `enable_cross_partition_query=True`.

---

### DC-6: Table Naming Inconsistency

**Severity:** Low

**Issue:** Table/container names differ between backends.

| Memory Type | PostgreSQL | SQLite | Cosmos DB |
|-------------|------------|--------|-----------|
| Heuristics | alma_heuristics | heuristics | alma-heuristics |
| Outcomes | alma_outcomes | outcomes | alma-outcomes |
| Preferences | alma_preferences | preferences | alma-preferences |
| Domain Knowledge | alma_domain_knowledge | domain_knowledge | alma-knowledge |
| Anti-Patterns | alma_anti_patterns | anti_patterns | alma-antipatterns |

**Code References:**
- PostgreSQL: `alma/storage/postgresql.py:173` - `{schema}.alma_heuristics`
- SQLite: `alma/storage/sqlite_local.py:108` - `heuristics`
- Cosmos DB: `alma/storage/azure_cosmos.py:74-80` - `CONTAINER_NAMES` dict

---

### DC-7: Missing Embedding Field in UserPreference

**Severity:** Low

**Issue:** UserPreference model does not support embeddings.

**Code References:**
- Types: `alma/types.py:94-108` - No `embedding` field
- PostgreSQL: `alma/storage/postgresql.py:221-232` - No embedding column
- Cosmos DB: `alma/storage/azure_cosmos.py:163-168` - `vector_indexes: False`

**Impact:** Cannot perform semantic search on user preferences. This may be intentional (preferences are looked up by user_id/category), but limits future capabilities.

---

### DC-8: SQLite Embeddings Table Naming Mismatch

**Severity:** High (Bug)

**Issue:** Delete operations use singular memory type names but the table stores plural names.

**Code References:**
- Save: `alma/storage/sqlite_local.py:364` - `_add_to_index("heuristics", ...)`
- Delete: `alma/storage/sqlite_local.py:947` - `DELETE FROM embeddings WHERE memory_type = 'heuristic'`

**Impact:** Embeddings are never deleted from the embeddings table for SQLite backend because the memory_type doesn't match.

---

### DC-9: Missing UPSERT Logic in File-Based Storage

**Severity:** Medium

**Issue:** File-based storage appends records without checking for duplicates.

**Code References:**
- `alma/storage/file_based.py:76-83` - `save_heuristic()` just appends

**Impact:** Duplicate records can accumulate if `save_heuristic()` is called with the same ID.

---

### DC-10: Inconsistent Default Values

**Severity:** Low

**Issue:** Default values for optional fields vary by backend.

| Field | PostgreSQL | SQLite | Cosmos DB | File-Based |
|-------|------------|--------|-----------|------------|
| confidence (heuristic) | 0.0 | 0.0 | 0.0 | 0.0 |
| confidence (preference) | 1.0 | 1.0 | 1.0 | 1.0 |
| occurrence_count | 0 | 0 | 0 (not set) | 0 |

**Code References:**
- Cosmos DB: `alma/storage/azure_cosmos.py:902-911` - Uses `.get()` with defaults

---

### DC-11: Azure Cosmos DB Update Requires Cross-Partition Query

**Severity:** Medium

**Issue:** Update operations must find the document first using cross-partition query.

**Code Reference:** `alma/storage/azure_cosmos.py:658-681`

```python
query = "SELECT * FROM c WHERE c.id = @id"
items = list(
    container.query_items(
        query=query,
        parameters=[{"name": "@id", "value": heuristic_id}],
        enable_cross_partition_query=True,  # Expensive!
    )
)
```

**Impact:** Update/delete by ID without knowing project_id is expensive (RU cost).

---

### DC-12: No Foreign Key Constraints

**Severity:** Low

**Issue:** No referential integrity between memory types.

**Impact:** Orphaned outcomes could reference non-existent heuristics. This is acceptable for a document-oriented design but limits relational analysis.

---

## Performance Considerations

### PC-1: Missing Index on timestamp (Outcomes)

**Severity:** High

**Issue:** Time-range queries on outcomes don't have timestamp indexes.

**Code Reference:** `alma/storage/postgresql.py:211-218` - Only project_agent and task_type indexes

**Impact:** `delete_outcomes_older_than()` performs full table scans.

**Recommendation:** Add timestamp index:
```sql
CREATE INDEX idx_outcomes_timestamp ON alma_outcomes(project_id, timestamp);
```

---

### PC-2: Missing Index on confidence (Heuristics)

**Severity:** Medium

**Issue:** Confidence-based filtering doesn't have dedicated index.

**Code Reference:** `alma/storage/postgresql.py:535-536` - `WHERE confidence >= %s`

**Impact:** Queries with `min_confidence` filter scan all rows for a project.

**Recommendation:** Add partial index:
```sql
CREATE INDEX idx_heuristics_confidence ON alma_heuristics(project_id, confidence)
WHERE confidence > 0.0;
```

---

### PC-3: IVFFlat Index Requires Data

**Severity:** Medium

**Issue:** IVFFlat indexes require existing data to build effectively.

**Code Reference:** `alma/storage/postgresql.py:280-291`
```python
try:
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_embedding
        ON {self.schema}.{table}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
except Exception:
    # IVFFlat requires data to build, skip if empty
    pass
```

**Impact:** Fresh databases have no vector indexes until data is added and manually reindexed.

**Recommendation:** Use HNSW index or implement automatic index rebuilding after data threshold.

---

### PC-4: SQLite FAISS Index Rebuild on Delete

**Severity:** Medium

**Issue:** Deleting items triggers full FAISS index rebuild.

**Code Reference:** `alma/storage/sqlite_local.py:954-958`
```python
if cursor.rowcount > 0:
    if "heuristic" in self._indices:
        self._load_faiss_indices()  # Full rebuild!
    return True
```

**Impact:** Delete operations are O(n) where n is total embeddings.

---

### PC-5: No Connection Pooling in SQLite

**Severity:** Low

**Issue:** SQLite creates new connection per operation.

**Code Reference:** `alma/storage/sqlite_local.py:87-99`
```python
def _get_connection(self):
    conn = sqlite3.connect(self.db_path)
    # ...
    conn.close()
```

**Impact:** Higher overhead for frequent operations.

---

### PC-6: File-Based Storage Full File Read/Write

**Severity:** Medium

**Issue:** Every operation reads and writes the entire JSON file.

**Code Reference:** `alma/storage/file_based.py:462-473`

**Impact:** O(n) for every operation. Not suitable for any production use.

---

### PC-7: No Batch Operations

**Severity:** Medium

**Issue:** No bulk insert/update methods exist in the interface.

**Code Reference:** `alma/storage/base.py:33-56` - Only single-item save methods

**Impact:** Saving multiple heuristics requires n separate transactions.

**Recommendation:** Add batch save methods to StorageBackend interface.

---

### PC-8: Azure Cosmos DB RU Cost for Updates

**Severity:** Medium

**Issue:** Updates require read-modify-write pattern with cross-partition query.

**Code Reference:** `alma/storage/azure_cosmos.py:683-711`

**Impact:** Each update costs ~2x RUs (read + write).

**Recommendation:** Consider point reads when project_id is known.

---

## Migration Path Analysis

### From File-Based to SQLite

**Complexity:** Low
**Risk:** Low

**Steps:**
1. Create SQLite database
2. Read JSON files
3. Insert records with proper datetime conversion
4. Generate embeddings if not present

**Concerns:**
- Embeddings may need regeneration if not stored in JSON

---

### From SQLite to PostgreSQL

**Complexity:** Medium
**Risk:** Medium

**Steps:**
1. Export SQLite tables to CSV/JSON
2. Convert timestamp strings to TIMESTAMPTZ
3. Convert embedding blobs to VECTOR format
4. Create PostgreSQL tables
5. Import data
6. Build pgvector indexes

**Concerns:**
- Embedding format conversion (numpy bytes to pgvector string)
- Index requires data to build IVFFlat

---

### From PostgreSQL to Azure Cosmos DB

**Complexity:** High
**Risk:** High

**Steps:**
1. Export PostgreSQL tables
2. Transform to document format (add `type` field)
3. Group by partition key
4. Batch upsert to containers
5. Configure vector indexing policies

**Concerns:**
- Partition strategy change (none to project_id)
- Document size limits (2MB)
- Cross-partition query cost during migration validation

---

### From Any Backend to Another

**Critical Path:**
1. Export all memory types
2. Normalize datetime to ISO8601 UTC
3. Normalize embeddings to float arrays
4. Import to target backend
5. Rebuild vector indexes
6. Validate counts match

**Migration Script Template:**
```python
# Not implemented - needs to be added to ALMA
def migrate_storage(source: StorageBackend, target: StorageBackend, project_ids: List[str]):
    for project_id in project_ids:
        # Get all data from source
        heuristics = source.get_heuristics(project_id, top_k=999999)
        outcomes = source.get_outcomes(project_id, top_k=999999)
        knowledge = source.get_domain_knowledge(project_id, top_k=999999)
        anti_patterns = source.get_anti_patterns(project_id, top_k=999999)

        # Save to target
        for h in heuristics:
            target.save_heuristic(h)
        # ... etc
```

---

## Vector Search Implementation Differences

### PostgreSQL (pgvector)

**Source:** `alma/storage/postgresql.py:520-570`

**Method:** Native SQL with `<=>` cosine distance operator

```sql
SELECT *, 1 - (embedding <=> %s::vector) as similarity
FROM alma_heuristics
WHERE project_id = %s AND confidence >= %s
ORDER BY similarity DESC
LIMIT %s
```

**Characteristics:**
- Index type: IVFFlat with 100 lists
- Distance: Cosine (via `vector_cosine_ops`)
- Fallback: Application-level numpy when pgvector unavailable
- Pros: Native integration, efficient for large datasets
- Cons: Requires extension installation

---

### SQLite (FAISS)

**Source:** `alma/storage/sqlite_local.py:225-333`

**Method:** External FAISS index with in-memory search

```python
# Normalize for cosine similarity (IndexFlatIP)
faiss.normalize_L2(query)
scores, indices = self._indices[memory_type].search(query, top_k)
```

**Characteristics:**
- Index type: IndexFlatIP (inner product on normalized vectors)
- Distance: Cosine (via normalization)
- Fallback: Numpy cosine similarity
- Pros: Works offline, no extensions needed
- Cons: Memory-resident index, rebuild on delete

---

### Azure Cosmos DB (DiskANN)

**Source:** `alma/storage/azure_cosmos.py:368-429`

**Method:** VectorDistance function in SQL query

```sql
SELECT TOP @top_k *
FROM c
WHERE c.project_id = @project_id
ORDER BY VectorDistance(c.embedding, @embedding)
```

**Characteristics:**
- Index type: DiskANN (configured via vector_embedding_policy)
- Distance: Cosine
- Fallback: None (requires vector-enabled account)
- Pros: Serverless scaling, integrated with Azure
- Cons: Expensive, region availability

---

### File-Based (None)

**Source:** `alma/storage/file_based.py:123-148`

**Method:** No vector search - ignores embedding parameter

```python
def get_heuristics(self, ..., embedding=None, ...):
    # embedding parameter is ignored
    # Returns confidence-sorted results
```

**Characteristics:**
- No semantic search capability
- Uses confidence/timestamp sorting only
- Suitable only for testing

---

### Vector Search Comparison Table

| Feature | PostgreSQL | SQLite | Cosmos DB | File-Based |
|---------|------------|--------|-----------|------------|
| Native Vector Type | Yes | No | Yes | No |
| Index Algorithm | IVFFlat | FlatIP | DiskANN | N/A |
| Approximate Search | Yes | No (exact) | Yes | N/A |
| Filter + Vector | Yes (native) | Yes (2-phase) | Yes (native) | N/A |
| Embedding in Same Table | Yes | No (separate) | Yes | Yes (unused) |
| Index Maintenance | Manual rebuild | Auto on add | Automatic | N/A |

---

## Technical Debt Analysis

### TD-1: No Migration System

**Severity:** High
**Effort:** Medium

**Issue:** No schema migration framework exists.

**Impact:**
- Adding new fields requires manual ALTER TABLE
- No version tracking for schema changes
- Difficult to roll back changes

**Recommendation:** Implement Alembic (PostgreSQL) or custom migration system.

---

### TD-2: SQLite Delete Bug (DC-8)

**Severity:** High
**Effort:** Low

**Issue:** Embeddings never deleted due to memory_type naming mismatch.

**Code Reference:** `alma/storage/sqlite_local.py:947, 965, 984, 1002`

**Fix:**
```python
# Change from:
DELETE FROM embeddings WHERE memory_type = 'heuristic'
# To:
DELETE FROM embeddings WHERE memory_type = 'heuristics'
```

---

### TD-3: No Interface for Batch Operations

**Severity:** Medium
**Effort:** Medium

**Issue:** Base interface lacks bulk operations.

**Code Reference:** `alma/storage/base.py`

**Recommendation:** Add to StorageBackend:
```python
def save_heuristics_batch(self, heuristics: List[Heuristic]) -> List[str]:
    """Batch save for performance."""
    pass
```

---

### TD-4: Inconsistent Error Handling

**Severity:** Medium
**Effort:** Low

**Issue:** Some operations return bool, others return counts, some raise exceptions.

**Examples:**
- `delete_heuristic()` returns `bool`
- `delete_outcomes_older_than()` returns `int`
- `save_*()` can raise on constraint violations

**Recommendation:** Standardize return types and exception handling.

---

### TD-5: No Observability

**Severity:** Medium
**Effort:** Medium

**Issue:** No metrics, tracing, or structured logging.

**Recommendations:**
- Add query timing to stats
- Implement OpenTelemetry traces
- Add structured logging for operations

---

### TD-6: Missing Abstract Methods

**Severity:** Low
**Effort:** Low

**Issue:** Some backends implement methods not in base interface.

**Example:** Azure Cosmos DB has `update_heuristic_confidence` but it's not called consistently.

**Recommendation:** Ensure all public methods are defined in `StorageBackend`.

---

## Recommendations

### Priority 1: Critical (Fix Immediately)

1. **Fix SQLite Delete Bug (TD-2)**
   - Change memory_type values in delete statements
   - Add unit tests for embedding deletion
   - **Effort:** 1 hour

2. **Add Timestamp Index (PC-1)**
   - Create index on outcomes timestamp
   - **Effort:** 30 minutes

### Priority 2: High (Fix This Sprint)

3. **Fix File-Based UPSERT (DC-9)**
   - Check for existing ID before append
   - **Effort:** 2 hours

4. **Implement Migration Framework (TD-1)**
   - Add Alembic for PostgreSQL
   - Add version tracking for SQLite
   - **Effort:** 1-2 days

5. **Add Batch Save Operations (TD-3)**
   - Add to base interface
   - Implement for each backend
   - **Effort:** 4 hours

### Priority 3: Medium (Plan for Next Release)

6. **Standardize Error Handling (TD-4)**
   - Define exception hierarchy
   - Update all backends
   - **Effort:** 1 day

7. **Add Confidence Index (PC-2)**
   - Create partial index
   - **Effort:** 30 minutes

8. **Optimize SQLite Delete (PC-4)**
   - Implement incremental index update
   - **Effort:** 4 hours

### Priority 4: Low (Backlog)

9. **Add Observability (TD-5)**
   - OpenTelemetry integration
   - **Effort:** 2 days

10. **Unify Table Names (DC-6)**
    - Standardize across backends
    - Requires migration
    - **Effort:** 1 day

11. **Document Migration Paths**
    - Create migration scripts
    - Add to documentation
    - **Effort:** 2 days

---

## Appendix: Code References Index

| File | Line Range | Description |
|------|------------|-------------|
| alma/storage/base.py | 1-373 | StorageBackend interface |
| alma/storage/postgresql.py | 1-1078 | PostgreSQL implementation |
| alma/storage/sqlite_local.py | 1-1014 | SQLite + FAISS implementation |
| alma/storage/azure_cosmos.py | 1-979 | Azure Cosmos DB implementation |
| alma/storage/file_based.py | 1-584 | File-based implementation |
| alma/types.py | 1-217 | Data model definitions |
| .alma/templates/config.yaml.template | 1-256 | Configuration template |

---

*End of Audit Report*
