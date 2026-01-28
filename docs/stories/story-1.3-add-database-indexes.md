# Story ALMA-003: Add Missing Database Indexes

**Story ID:** ALMA-003
**Epic:** ALMA Technical Debt Resolution
**Priority:** P1 (High)
**Points:** 1
**Sprint:** 1

---

## User Story

**As a** developer deploying ALMA in production
**I want** proper indexes on frequently queried columns
**So that** time-based queries perform efficiently as data grows

## Background

Both PostgreSQL and SQLite backends lack timestamp indexes on the outcomes table, causing full table scans for `delete_outcomes_older_than()` and time-based queries.

## Technical Details

### PostgreSQL
**File:** `alma/storage/postgresql.py`
**Lines:** 211-218 (add after existing indexes)

**Missing Index:**
```sql
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
ON alma_outcomes(project_id, timestamp DESC);
```

### SQLite
**File:** `alma/storage/sqlite_local.py`
**Lines:** 127-151 (add after existing table creation)

**Missing Index:**
```sql
CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
ON outcomes(project_id, timestamp);
```

## Acceptance Criteria

- [ ] PostgreSQL backend creates timestamp index on init
- [ ] SQLite backend creates timestamp index on init
- [ ] Existing databases get index on next connection
- [ ] `delete_outcomes_older_than()` uses index (verify with EXPLAIN)
- [ ] No errors on database upgrade

## Implementation

### PostgreSQL (postgresql.py)

Add after line 218:
```python
conn.execute(f"""
    CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
    ON {self.schema}.alma_outcomes(project_id, timestamp DESC)
""")
```

### SQLite (sqlite_local.py)

Add after line 151:
```python
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp "
    "ON outcomes(project_id, timestamp)"
)
```

## Test Cases

1. Fresh PostgreSQL database creates index
2. Fresh SQLite database creates index
3. Existing database adds index on upgrade
4. Time-based queries use index (EXPLAIN ANALYZE)
5. No duplicate index errors on re-init

## Definition of Done

- [ ] PostgreSQL index added
- [ ] SQLite index added
- [ ] Unit tests verify index creation
- [ ] Performance test shows improvement

## Files to Modify

- `alma/storage/postgresql.py` (add 4 lines)
- `alma/storage/sqlite_local.py` (add 4 lines)
- `tests/test_storage/` (add index verification tests)

---

**Estimated Effort:** 30 minutes
