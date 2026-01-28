# Database Specialist Review

**Document Reviewed:** `/docs/prd/technical-debt-DRAFT.md`
**Reviewer:** Database Specialist
**Review Date:** 2026-01-28
**Approval Status:** APPROVED WITH CORRECTIONS

---

## Executive Summary

I have reviewed all database-related findings in the technical debt DRAFT document against the actual source code. The majority of findings are **confirmed and accurate**. I am providing corrections, additional context, and remediation code snippets where needed.

**Validation Summary:**

| Finding ID | Status | Notes |
|------------|--------|-------|
| HIGH-001 | CONFIRMED | Critical bug - singular vs plural mismatch |
| HIGH-003 | CONFIRMED | Missing timestamp index on outcomes |
| HIGH-004 | CONFIRMED | IVFFlat empty table issue |
| MED-001 | CONFIRMED | File-based storage missing UPSERT |
| MED-003 | CONFIRMED | No batch operations interface |
| MED-004 | CONFIRMED | SQLite FAISS full rebuild on delete |
| MED-007 | CONFIRMED | Azure Cosmos cross-partition queries |

---

## Detailed Finding Validations

### HIGH-001: SQLite Embeddings Never Deleted (Bug)

**Status:** CONFIRMED - CRITICAL BUG

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/sqlite_local.py`:

**Save operations use PLURAL names (lines 364, 396, 448, 478):**
```python
# Line 364 - save_heuristic()
self._add_to_index("heuristics", heuristic.id, heuristic.embedding)

# Line 396 - save_outcome()
self._add_to_index("outcomes", outcome.id, outcome.embedding)

# Line 448 - save_domain_knowledge()
self._add_to_index("domain_knowledge", knowledge.id, knowledge.embedding)

# Line 478 - save_anti_pattern()
self._add_to_index("anti_patterns", anti_pattern.id, anti_pattern.embedding)
```

**Delete operations use SINGULAR names (lines 947, 965, 984, 1002):**
```python
# Line 947 - delete_heuristic()
DELETE FROM embeddings WHERE memory_type = 'heuristic' AND memory_id = ?

# Line 965 - delete_outcome()
DELETE FROM embeddings WHERE memory_type = 'outcome' AND memory_id = ?

# Line 984 - delete_domain_knowledge()
DELETE FROM embeddings WHERE memory_type = 'domain_knowledge' AND memory_id = ?
# Note: This one happens to match! But the pattern is inconsistent.

# Line 1002 - delete_anti_pattern()
DELETE FROM embeddings WHERE memory_type = 'anti_pattern' AND memory_id = ?
```

**Impact Analysis:**
- Embeddings for heuristics, outcomes, and anti_patterns are NEVER deleted
- Only `domain_knowledge` embeddings are correctly deleted (coincidental match)
- Database will grow unbounded with orphaned embedding data
- Vector index contains stale data, potentially returning deleted records in similarity searches

**Severity Adjustment:** Recommend upgrading to **CRITICAL** due to data integrity implications

**Remediation:**
```python
# In delete_heuristic() - line 947
conn.execute(
    "DELETE FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
    (heuristic_id,),
)

# In delete_outcome() - line 965
conn.execute(
    "DELETE FROM embeddings WHERE memory_type = 'outcomes' AND memory_id = ?",
    (outcome_id,),
)

# In delete_anti_pattern() - line 1002
conn.execute(
    "DELETE FROM embeddings WHERE memory_type = 'anti_patterns' AND memory_id = ?",
    (anti_pattern_id,),
)
```

**Additional Issue Found:** Lines 954-958, 973-976, 992-994, 1010-1012 check for singular names in `self._indices`:
```python
if "heuristic" in self._indices:  # Should be "heuristics"
    self._load_faiss_indices()
```

This means the index rebuild is also never triggered after deletion.

---

### HIGH-003: Missing Timestamp Index on Outcomes

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/postgresql.py`, lines 211-218:

```python
conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_outcomes_project_agent
    ON {self.schema}.alma_outcomes(project_id, agent)
""")
conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_outcomes_task_type
    ON {self.schema}.alma_outcomes(project_id, agent, task_type)
""")
```

**Missing Index:** No index on `timestamp` column, which is used by:
- `delete_outcomes_older_than()` (line 895): `WHERE ... AND timestamp < %s`
- `get_outcomes()` (line 612): `ORDER BY timestamp DESC`

**Impact:**
- Full table scans for time-based pruning operations
- Performance degrades linearly with data growth
- Memory pruning becomes expensive over time

**Remediation:**
```sql
-- Add to _init_database() after line 218
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
ON {schema}.alma_outcomes(project_id, timestamp DESC);

-- For composite queries (optional optimization)
CREATE INDEX IF NOT EXISTS idx_outcomes_agent_timestamp
ON {schema}.alma_outcomes(project_id, agent, timestamp DESC);
```

---

### HIGH-004: IVFFlat Index Not Built on Empty Tables

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/postgresql.py`, lines 280-291:

```python
if self._pgvector_available:
    for table in ["alma_heuristics", "alma_outcomes", "alma_domain_knowledge", "alma_anti_patterns"]:
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

**Technical Details:**
- IVFFlat (Inverted File with Flat) requires existing data to compute centroids
- On empty tables, the `CREATE INDEX` silently fails (caught by bare `except`)
- No retry mechanism after data is inserted
- No logging of the failure

**Impact:**
- Fresh deployments have no vector indexes
- First queries after data insertion use sequential scans
- No automatic index creation when data threshold is reached

**Severity Adjustment:** Maintain HIGH - this affects all new PostgreSQL deployments

**Remediation Options:**

**Option 1: Use HNSW index (recommended)**
```python
# HNSW doesn't require existing data
conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_{table}_embedding
    ON {self.schema}.{table}
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
""")
```

**Option 2: Deferred IVFFlat with threshold**
```python
def _ensure_vector_indexes(self):
    """Create IVFFlat indexes if sufficient data exists."""
    THRESHOLD = 1000  # IVFFlat performs well with lists=100 when data > 1000

    with self._get_connection() as conn:
        for table in ["alma_heuristics", "alma_outcomes", "alma_domain_knowledge", "alma_anti_patterns"]:
            # Check row count
            result = conn.execute(f"SELECT COUNT(*) FROM {self.schema}.{table}").fetchone()
            row_count = result[0] if result else 0

            if row_count >= THRESHOLD:
                try:
                    # Check if index exists
                    conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table}_embedding
                        ON {self.schema}.{table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
                    logger.info(f"Created IVFFlat index on {table}")
                except Exception as e:
                    logger.warning(f"Could not create IVFFlat index on {table}: {e}")
```

---

### MED-001: File-Based Storage Missing UPSERT Logic

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/file_based.py`, lines 76-83:

```python
def save_heuristic(self, heuristic: Heuristic) -> str:
    """Save a heuristic."""
    data = self._read_json(self._files["heuristics"])
    record = self._to_dict(heuristic)
    data.append(record)  # Always appends, never checks for existing ID
    self._write_json(self._files["heuristics"], data)
    logger.debug(f"Saved heuristic: {heuristic.id}")
    return heuristic.id
```

Same pattern exists for all save methods (lines 85-119).

**Impact:**
- Duplicate records accumulate if same ID is saved twice
- Data integrity issues in test/development environments
- Comparison: SQLite uses `INSERT OR REPLACE`, PostgreSQL uses `ON CONFLICT DO UPDATE`

**Remediation:**
```python
def save_heuristic(self, heuristic: Heuristic) -> str:
    """Save a heuristic (upsert semantics)."""
    data = self._read_json(self._files["heuristics"])
    record = self._to_dict(heuristic)

    # Find and replace existing, or append new
    found = False
    for i, existing in enumerate(data):
        if existing.get("id") == heuristic.id:
            data[i] = record
            found = True
            break

    if not found:
        data.append(record)

    self._write_json(self._files["heuristics"], data)
    logger.debug(f"Saved heuristic: {heuristic.id}")
    return heuristic.id
```

---

### MED-003: No Batch Operations Interface

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/base.py`, the abstract interface defines only single-item operations:
- `save_heuristic(self, heuristic: Heuristic) -> str`
- `save_outcome(self, outcome: Outcome) -> str`
- etc.

No batch methods exist:
- `save_heuristics(self, heuristics: List[Heuristic]) -> List[str]`
- `save_outcomes(self, outcomes: List[Outcome]) -> List[str]`

**Impact:**
- N separate transactions/connections for N memories
- Inefficient for bulk imports or aggregation operations
- PostgreSQL and Azure Cosmos both support bulk operations natively

**Remediation:**

Add to `base.py`:
```python
def save_heuristics(
    self,
    heuristics: List[Heuristic],
) -> List[str]:
    """
    Save multiple heuristics in a single batch operation.

    Default implementation calls save_heuristic() in a loop.
    Backends should override for bulk optimization.
    """
    return [self.save_heuristic(h) for h in heuristics]

def save_outcomes(
    self,
    outcomes: List[Outcome],
) -> List[str]:
    """Save multiple outcomes in a single batch operation."""
    return [self.save_outcome(o) for o in outcomes]
```

PostgreSQL implementation example:
```python
def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
    """Batch save heuristics using executemany."""
    with self._get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.schema}.alma_heuristics (...)
                VALUES (%s, %s, ...)
                ON CONFLICT (id) DO UPDATE SET ...
                """,
                [(h.id, h.agent, ...) for h in heuristics]
            )
        conn.commit()
    return [h.id for h in heuristics]
```

---

### MED-004: SQLite FAISS Index Full Rebuild on Delete

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/sqlite_local.py`, lines 954-958:

```python
if cursor.rowcount > 0:
    # Rebuild index if we had one
    if "heuristic" in self._indices:  # Note: wrong key, should be "heuristics"
        self._load_faiss_indices()
    return True
```

And `_load_faiss_indices()` (lines 225-258) reloads ALL embeddings from the database:

```python
def _load_faiss_indices(self):
    """Load or create FAISS indices for each memory type."""
    memory_types = ["heuristics", "outcomes", "domain_knowledge", "anti_patterns"]

    for memory_type in memory_types:
        # Create new index
        self._indices[memory_type] = faiss.IndexFlatIP(self.embedding_dim)
        self._id_maps[memory_type] = []

        # Load ALL embeddings from database
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_id, embedding FROM embeddings WHERE memory_type = ?",
                (memory_type,),
            )
            rows = cursor.fetchall()
            # ... adds all rows to index
```

**Impact:**
- Every delete triggers O(n) rebuild where n = total embeddings
- For 10,000 embeddings, every delete reloads all 10,000
- Memory pressure and latency spikes on delete operations

**Remediation:**

**Option 1: Lazy rebuild (simple)**
```python
def delete_heuristic(self, heuristic_id: str) -> bool:
    """Delete a heuristic by ID."""
    with self._get_connection() as conn:
        # Delete from embeddings table
        conn.execute(
            "DELETE FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
            (heuristic_id,),
        )
        cursor = conn.execute(
            "DELETE FROM heuristics WHERE id = ?",
            (heuristic_id,),
        )
        if cursor.rowcount > 0:
            # Mark index as dirty, rebuild on next search
            self._index_dirty["heuristics"] = True
            return True
        return False

def _search_index(self, memory_type: str, query_embedding: List[float], top_k: int):
    # Rebuild if dirty
    if self._index_dirty.get(memory_type, False):
        self._rebuild_single_index(memory_type)
        self._index_dirty[memory_type] = False
    # ... continue with search
```

**Option 2: Use IndexIDMap (FAISS native)**
```python
# IndexIDMap supports removal by ID
self._indices[memory_type] = faiss.IndexIDMap(
    faiss.IndexFlatIP(self.embedding_dim)
)

# Remove specific embedding
self._indices[memory_type].remove_ids(np.array([internal_id]))
```

---

### MED-007: Azure Cosmos Cross-Partition Query for Updates

**Status:** CONFIRMED

**Code Verification:**

In `/Users/friendlyaifi/Documents/GitHub/ALMA-memory/alma/storage/azure_cosmos.py`, lines 658-681:

```python
def update_heuristic(
    self,
    heuristic_id: str,
    updates: Dict[str, Any],
) -> bool:
    """Update a heuristic's fields."""
    container = self._get_container("heuristics")

    # We need project_id to read the item (partition key)
    # First try to find the heuristic
    query = "SELECT * FROM c WHERE c.id = @id"
    items = list(
        container.query_items(
            query=query,
            parameters=[{"name": "@id", "value": heuristic_id}],
            enable_cross_partition_query=True,  # EXPENSIVE!
        )
    )
```

Same pattern in `increment_heuristic_occurrence()` (lines 692-699) and `delete_heuristic()` (lines 806-812).

**Impact:**
- Cross-partition queries consume significantly more RU (Request Units)
- Azure Cosmos bills by RU consumption
- Latency increases as data grows across partitions
- Single-partition queries are 5-10x more efficient

**Remediation:**

**Option 1: Require partition key in API**
```python
def update_heuristic(
    self,
    heuristic_id: str,
    project_id: str,  # Add partition key parameter
    updates: Dict[str, Any],
) -> bool:
    """Update a heuristic's fields."""
    container = self._get_container("heuristics")

    # Point read (1 RU) instead of cross-partition query (many RU)
    try:
        doc = container.read_item(item=heuristic_id, partition_key=project_id)
    except exceptions.CosmosResourceNotFoundError:
        return False

    # Apply updates
    for key, value in updates.items():
        if isinstance(value, datetime):
            doc[key] = value.isoformat()
        else:
            doc[key] = value

    container.replace_item(item=heuristic_id, body=doc)
    return True
```

**Option 2: Composite ID encoding**
```python
# Store partition key in ID: "{project_id}:{heuristic_id}"
# Allows extraction without cross-partition query
```

---

## Additional Issues Found

### NEW-001: SQLite Missing Timestamp Index on Outcomes

**Severity:** MEDIUM
**Location:** `alma/storage/sqlite_local.py:127-151`

Similar to PostgreSQL, SQLite also lacks a timestamp index on the outcomes table. The `delete_outcomes_older_than()` method (line 741) performs a full table scan:

```python
query = "DELETE FROM outcomes WHERE project_id = ? AND timestamp < ?"
```

**Remediation:**
```python
# Add after line 151
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp "
    "ON outcomes(project_id, timestamp)"
)
```

---

### NEW-002: Azure Cosmos Missing update_heuristic_confidence and update_knowledge_confidence

**Severity:** HIGH (already noted as HIGH-002 in DRAFT)
**Location:** `alma/storage/azure_cosmos.py`

Confirmed that these methods are indeed missing from the Azure Cosmos backend. The base interface (`base.py` lines 212-246) requires:
- `update_heuristic_confidence()`
- `update_knowledge_confidence()`

These are implemented in SQLite (lines 916-940) and PostgreSQL (lines 817-843) but NOT in Azure Cosmos.

**Remediation:**
```python
def update_heuristic_confidence(
    self,
    heuristic_id: str,
    new_confidence: float,
) -> bool:
    """Update confidence score for a heuristic."""
    return self.update_heuristic(heuristic_id, {"confidence": new_confidence})

def update_knowledge_confidence(
    self,
    knowledge_id: str,
    new_confidence: float,
) -> bool:
    """Update confidence score for domain knowledge."""
    container = self._get_container("knowledge")

    # Cross-partition query needed (see MED-007)
    query = "SELECT * FROM c WHERE c.id = @id"
    items = list(
        container.query_items(
            query=query,
            parameters=[{"name": "@id", "value": knowledge_id}],
            enable_cross_partition_query=True,
        )
    )

    if not items:
        return False

    doc = items[0]
    doc["confidence"] = new_confidence
    doc["last_verified"] = datetime.now(timezone.utc).isoformat()

    container.replace_item(item=knowledge_id, body=doc)
    return True
```

---

### NEW-003: Bare Exception Handling in PostgreSQL

**Severity:** LOW
**Location:** `alma/storage/postgresql.py:289-291`

```python
except Exception:
    # IVFFlat requires data to build, skip if empty
    pass
```

This catches ALL exceptions, masking potential configuration errors or permission issues.

**Remediation:**
```python
except psycopg.errors.InvalidTableDefinition as e:
    # IVFFlat requires data to build, skip if empty
    logger.debug(f"Skipping IVFFlat index creation (likely empty table): {e}")
except Exception as e:
    logger.warning(f"Unexpected error creating vector index: {e}")
```

---

## Summary of Required Changes

### Immediate (Sprint 1)

| Priority | File | Change |
|----------|------|--------|
| P0 | sqlite_local.py | Fix plural/singular mismatch in delete operations (lines 947, 965, 1002) |
| P0 | sqlite_local.py | Fix index key checks (lines 956, 974, 1010) |
| P1 | postgresql.py | Add timestamp index on alma_outcomes |
| P1 | sqlite_local.py | Add timestamp index on outcomes |

### Short-term (Sprint 2)

| Priority | File | Change |
|----------|------|--------|
| P1 | postgresql.py | Switch to HNSW or implement deferred IVFFlat |
| P1 | azure_cosmos.py | Implement missing confidence update methods |
| P2 | file_based.py | Add UPSERT logic to save methods |
| P2 | base.py | Add batch operation interface |

### Medium-term (Sprint 3+)

| Priority | File | Change |
|----------|------|--------|
| P2 | azure_cosmos.py | Refactor to avoid cross-partition queries |
| P2 | sqlite_local.py | Implement lazy index rebuild or use IndexIDMap |
| P3 | postgresql.py | Improve exception handling specificity |

---

## Approval

**Status:** APPROVED WITH CORRECTIONS

The DRAFT document accurately identifies the database-related technical debt. I have verified all findings against the source code and provided:

1. Confirmation of all listed issues
2. Specific line number references
3. Remediation code snippets
4. Two additional issues discovered during review
5. Prioritized implementation recommendations

The document is ready to proceed to the next review phase after incorporating the corrections noted above.

---

**Signed:** Database Specialist
**Date:** 2026-01-28
