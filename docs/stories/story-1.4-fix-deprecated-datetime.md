# Story ALMA-004: Fix Deprecated datetime.utcnow()

**Story ID:** ALMA-004
**Epic:** ALMA Technical Debt Resolution
**Priority:** P1 (High)
**Points:** 1
**Sprint:** 1

---

## User Story

**As a** developer maintaining ALMA
**I want** all datetime usage to be Python 3.12+ compatible
**So that** the codebase remains future-proof

## Background

`datetime.utcnow()` is deprecated since Python 3.12 and will be removed in a future version. The recommended replacement is `datetime.now(timezone.utc)`.

## Technical Details

**File:** `alma/types.py`
**Lines:** 88, 106, 124, 144

**Current Code:**
```python
timestamp: datetime = field(default_factory=datetime.utcnow)
```

**Correct Code:**
```python
from datetime import datetime, timezone
timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

### Affected Types

| Line | Type | Field |
|------|------|-------|
| 88 | Outcome | timestamp |
| 106 | UserPreference | timestamp |
| 124 | DomainKnowledge | last_verified |
| 144 | AntiPattern | created_at |

## Acceptance Criteria

- [ ] All 4 occurrences updated to use `datetime.now(timezone.utc)`
- [ ] No `DeprecationWarning` in Python 3.12+
- [ ] Timestamps are timezone-aware (UTC)
- [ ] Existing code remains compatible
- [ ] Tests verify timezone awareness

## Implementation

### Step 1: Add Import
At top of `alma/types.py`:
```python
from datetime import datetime, timezone
```

### Step 2: Update All Occurrences

```python
# Line 88 (Outcome)
timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Line 106 (UserPreference)
timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Line 124 (DomainKnowledge)
last_verified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# Line 144 (AntiPattern)
created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

### Optional: Create Helper
```python
def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)
```

## Test Cases

1. Create Outcome and verify timestamp is timezone-aware
2. Create UserPreference and verify timestamp is timezone-aware
3. Create DomainKnowledge and verify last_verified is timezone-aware
4. Create AntiPattern and verify created_at is timezone-aware
5. Run with Python 3.12+ and verify no deprecation warnings

## Definition of Done

- [ ] All 4 datetime usages updated
- [ ] Import added for timezone
- [ ] Unit tests verify timezone awareness
- [ ] No deprecation warnings in CI

## Files to Modify

- `alma/types.py` (4 line changes + 1 import)
- `tests/test_types.py` (add timezone tests)

---

**Estimated Effort:** 30 minutes
