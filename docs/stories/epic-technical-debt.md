# Epic: ALMA Technical Debt Resolution

**Epic ID:** ALMA-TECH-DEBT-001
**Priority:** P0 (Critical) / P1 (High)
**Status:** Ready for Planning
**Source:** Brownfield Discovery Assessment (2026-01-28)

---

## Epic Summary

Resolve technical debt identified in the ALMA-Memory brownfield discovery assessment, focusing on critical security vulnerabilities, data integrity bugs, and production blockers.

## Business Value

- **Eliminate security risk** from code execution vulnerability
- **Prevent data integrity issues** from unbounded storage growth
- **Enable production deployment** on Azure Cosmos DB
- **Reduce debugging time** by 60% with proper exception handling
- **Future-proof** codebase for Python 3.13+

## Acceptance Criteria

1. All CRITICAL and HIGH severity issues resolved
2. Unit tests pass for all fixed issues
3. No regression in existing functionality
4. Documentation updated where applicable
5. Security review passed for CRIT-001 fix

---

## Story Breakdown

### Sprint 1: Critical Security & Data Integrity (4 hours total)

| Story ID | Title | Points | Priority |
|----------|-------|--------|----------|
| ALMA-001 | Fix eval() Security Vulnerability | 2 | P0 |
| ALMA-002 | Fix SQLite Embeddings Delete Bug | 1 | P0 |
| ALMA-003 | Add Missing Database Indexes | 1 | P1 |
| ALMA-004 | Fix Deprecated datetime.utcnow() | 1 | P1 |

### Sprint 2: Production Readiness (18 hours total)

| Story ID | Title | Points | Priority |
|----------|-------|--------|----------|
| ALMA-005 | Implement Azure Cosmos Update Methods | 5 | P1 |
| ALMA-006 | Fix PostgreSQL IVFFlat Index Issue | 3 | P1 |
| ALMA-007 | Create Exception Hierarchy | 5 | P1 |
| ALMA-008 | Improve Config Template Discoverability | 2 | P2 |

### Backlog: Medium Priority

| Story ID | Title | Points | Priority |
|----------|-------|--------|----------|
| ALMA-009 | Add Batch Operations Interface | 5 | P2 |
| ALMA-010 | Implement Schema Migration Framework | 8 | P2 |
| ALMA-011 | Add Observability (Metrics/Tracing) | 8 | P2 |
| ALMA-012 | Package Testing Utilities | 5 | P2 |
| ALMA-013 | Add Input Validation to MCP Tools | 3 | P2 |

---

## Dependencies

- ALMA-002 must complete before ALMA-006 (both affect vector indexes)
- ALMA-007 should complete before ALMA-005 (Azure uses new exceptions)

## Risks

| Risk | Mitigation |
|------|------------|
| Breaking changes in public API | Use deprecation warnings, maintain backwards compatibility |
| Azure testing requires cloud resources | Use mock tests, integration tests in CI only |
| Schema migration affects existing deployments | Provide migration scripts, document upgrade path |

---

## Definition of Done

- [ ] Code changes reviewed and approved
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced (for CRIT-001)
- [ ] Performance benchmarks maintained

---

## Related Documents

- [Technical Debt Assessment](../prd/technical-debt-assessment.md)
- [Executive Report](../reports/TECHNICAL-DEBT-REPORT.md)
- [System Architecture](../architecture/system-architecture.md)
- [Database Audit](../../supabase/docs/DB-AUDIT.md)
