# Story ALMA-002: Fix SQLite Embeddings Delete Bug

**Story ID:** ALMA-002
**Epic:** ALMA Technical Debt Resolution
**Priority:** P0 (Critical)
**Points:** 1
**Sprint:** 1

---

## User Story

**As a** developer using SQLite storage
**I want** deleted memories to be fully removed including embeddings
**So that** the database doesn't grow unbounded and stale data doesn't affect searches

## Background

Delete operations use singular memory type names (`'heuristic'`) but the embeddings table stores plural names (`'heuristics'`), causing embeddings to never be deleted. Additionally, index rebuild checks use the wrong key names.

## Technical Details

**File:** `alma/storage/sqlite_local.py`

### Bug 1: Delete SQL uses wrong memory_type
**Lines:** 947, 965, 984, 1002

**Current Code (line 947):**
```python
conn.execute(
    "DELETE FROM embeddings WHERE memory_type = 'heuristic' AND memory_id = ?",
    (heuristic_id,),
)
```

**Correct Code:**
```python
conn.execute(
    "DELETE FROM embeddings WHERE memory_type = 'heuristics' AND memory_id = ?",
    (heuristic_id,),
)
```

### Bug 2: Index rebuild uses wrong key
**Lines:** 954, 973, 992, 1010

**Current Code (line 954):**
```python
if "heuristic" in self._indices:
    self._load_faiss_indices()
```

**Correct Code:**
```python
if "heuristics" in self._indices:
    self._load_faiss_indices()
```

## Acceptance Criteria

- [ ] All delete operations use plural memory_type names
- [ ] All index rebuild checks use plural key names
- [ ] Embeddings are actually deleted from database
- [ ] FAISS index is rebuilt after deletion
- [ ] Unit tests verify embedding deletion
- [ ] Unit tests verify index key matching

## Implementation

### Changes Required

| Line | Current | Correct |
|------|---------|---------|
| 947 | `'heuristic'` | `'heuristics'` |
| 954 | `"heuristic"` | `"heuristics"` |
| 965 | `'outcome'` | `'outcomes'` |
| 973 | `"outcome"` | `"outcomes"` |
| 1002 | `'anti_pattern'` | `'anti_patterns'` |
| 1010 | `"anti_pattern"` | `"anti_patterns"` |

Note: `domain_knowledge` at lines 984, 992 is already correct.

## Test Cases

1. Delete heuristic and verify embedding removed
2. Delete outcome and verify embedding removed
3. Delete anti_pattern and verify embedding removed
4. Verify FAISS index is rebuilt after deletion
5. Verify vector search doesn't return deleted items

## Definition of Done

- [ ] All 6 string corrections made
- [ ] Unit tests added for embedding deletion
- [ ] Integration tests pass
- [ ] Vector search tests verify no stale data

## Files to Modify

- `alma/storage/sqlite_local.py` (6 string changes)
- `tests/test_storage/test_sqlite_local.py` (add deletion tests)

---

**Estimated Effort:** 30 minutes
