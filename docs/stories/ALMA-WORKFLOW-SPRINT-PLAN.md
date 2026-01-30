# ALMA Workflow Context Layer - Sprint Plan

**Epic**: ALMA Enhancement for AGtestari Workflow Studio Integration
**PRP Reference**: `PRPs/ALMA_AGTESTARI_INTEGRATION_PRP.md`
**Total Effort**: ~16 days
**Sprint Duration**: 1 week each
**Start Date**: TBD
**Created By**: @pm (Morgan)

---

## Executive Summary

Transform ALMA from an "agent learning system" to an "enterprise workflow intelligence layer" by implementing:
- Workflow-scoped memory retrieval
- Automatic execution checkpointing
- State reducers for parallel branch merging
- Multi-tenant memory hierarchy
- Artifact-linked memory

---

## Team Assignments

| Agent | Role | Responsibilities |
|-------|------|------------------|
| @pm (Morgan) | Project Lead | Sprint planning, progress tracking, stakeholder communication |
| @architect (Aria) | Technical Lead | Architecture guidance, code reviews, API design validation |
| @data-analyst (Dana) | Database Lead | Schema design, Cloudflare provisioning, query optimization |
| @dev | Implementation | Code development following PRP specifications |
| @qa | Quality Assurance | Test creation, validation, coverage verification |
| @devops | Infrastructure | CI/CD, deployment, Cloudflare setup |

---

## Sprint Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Sprint 0   │  Sprint 1   │  Sprint 2   │  Sprint 3   │  Sprint 4  │
│  (3 days)   │  (5 days)   │  (5 days)   │  (5 days)   │  (3 days)  │
├─────────────┼─────────────┼─────────────┼─────────────┼────────────┤
│  Planning   │  Foundation │  Storage    │  Core API   │  Polish    │
│  & Schema   │  & Types    │  Layer      │  & Features │  & Launch  │
└─────────────────────────────────────────────────────────────────────┘
        │             │             │             │            │
        ▼             ▼             ▼             ▼            ▼
   Dana designs   Types &      All backends   Checkpoint   Docs &
   DB schemas    Reducers      updated       & Retrieval   Testing
```

---

## Sprint 0: Planning & Database Design (3 days)

**Goal**: Complete all prerequisite planning and database schema design

### Tasks

| ID | Task | Owner | Est | Dependencies |
|----|------|-------|-----|--------------|
| 0.1 | Review PRP with team, clarify requirements | @pm | 0.5d | — |
| 0.2 | Design checkpoint table schema | @data-analyst | 0.5d | 0.1 |
| 0.3 | Design workflow_outcomes table schema | @data-analyst | 0.5d | 0.1 |
| 0.4 | Design artifact_links table schema | @data-analyst | 0.5d | 0.1 |
| 0.5 | Provision Cloudflare Hyperdrive PostgreSQL | @data-analyst | 0.5d | 0.2-0.4 |
| 0.6 | Setup Cloudflare R2 bucket for artifacts | @devops | 0.5d | 0.1 |
| 0.7 | Create migration scripts skeleton | @data-analyst | 0.5d | 0.2-0.4 |

### Definition of Done
- [ ] All table schemas documented and approved by @architect
- [ ] Cloudflare Hyperdrive PostgreSQL instance running
- [ ] Cloudflare R2 bucket configured
- [ ] Migration scripts structure created
- [ ] Team has access to all infrastructure

### Milestone: **Infrastructure Ready** 🏗️

---

## Sprint 1: Foundation & Core Types (5 days)

**Goal**: Implement core types and state reducers module

### Tasks

| ID | Task | Owner | Est | Dependencies |
|----|------|-------|-----|--------------|
| 1.1 | Implement `RetrievalScope` enum | @dev | 0.5d | Sprint 0 |
| 1.2 | Implement `WorkflowContext` dataclass | @dev | 0.5d | 1.1 |
| 1.3 | Implement `Checkpoint` dataclass | @dev | 0.5d | 1.1 |
| 1.4 | Implement `ArtifactRef` dataclass | @dev | 0.25d | 1.1 |
| 1.5 | Implement `WorkflowOutcome` dataclass | @dev | 0.5d | 1.1 |
| 1.6 | Update `MemorySlice` with workflow_outcomes | @dev | 0.25d | 1.5 |
| 1.7 | Create `alma/workflow/__init__.py` | @dev | 0.25d | 1.1-1.6 |
| 1.8 | Implement `StateReducer` abstract class | @dev | 0.5d | 1.7 |
| 1.9 | Implement built-in reducers (6 types) | @dev | 1d | 1.8 |
| 1.10 | Implement `ReducerConfig` & `StateMerger` | @dev | 0.5d | 1.9 |
| 1.11 | Write unit tests for types | @qa | 0.5d | 1.1-1.6 |
| 1.12 | Write unit tests for reducers | @qa | 0.5d | 1.8-1.10 |
| 1.13 | Architecture review checkpoint | @architect | 0.25d | 1.1-1.10 |

### Definition of Done
- [ ] All new types in `alma/types.py` passing tests
- [ ] `alma/workflow/` module created with reducers
- [ ] `RetrievalScope` enum (NOT MemoryScope) implemented
- [ ] All 6 reducer types working
- [ ] Unit tests passing with >90% coverage on new code
- [ ] @architect approval on API design

### Milestone: **Types Complete** 📦

---

## Sprint 2: Storage Layer (5 days)

**Goal**: Update all storage backends with checkpoint and workflow support

### Tasks

| ID | Task | Owner | Est | Dependencies |
|----|------|-------|-----|--------------|
| 2.1 | Add `scope_filter` to `StorageBackend` base | @dev | 0.5d | Sprint 1 |
| 2.2 | Add checkpoint abstract methods to base | @dev | 0.5d | 2.1 |
| 2.3 | Add workflow_outcome abstract methods | @dev | 0.25d | 2.1 |
| 2.4 | Add artifact_links abstract methods | @dev | 0.25d | 2.1 |
| 2.5 | Implement SQLite checkpoint tables | @dev | 0.5d | 2.2, Dana schemas |
| 2.6 | Implement SQLite workflow_outcomes | @dev | 0.5d | 2.3 |
| 2.7 | Implement SQLite artifact_links | @dev | 0.25d | 2.4 |
| 2.8 | Implement PostgreSQL checkpoint tables | @dev | 0.5d | 2.2, Dana schemas |
| 2.9 | Implement PostgreSQL workflow_outcomes | @dev | 0.5d | 2.3 |
| 2.10 | Implement PostgreSQL artifact_links | @dev | 0.25d | 2.4 |
| 2.11 | Add pgvector index for workflow_outcomes | @data-analyst | 0.25d | 2.9 |
| 2.12 | Implement Cosmos DB containers | @dev | 0.5d | 2.2-2.4 |
| 2.13 | Create migration runner | @dev | 0.5d | 2.5-2.12 |
| 2.14 | Write migration scripts (001-004) | @data-analyst | 0.5d | 2.13 |
| 2.15 | Storage unit tests | @qa | 1d | 2.5-2.12 |
| 2.16 | Test migrations on all backends | @qa | 0.5d | 2.14 |

### Definition of Done
- [ ] All 4 backends (SQLite, PostgreSQL, Cosmos, FileBased) updated
- [ ] `scope_filter` parameter working on all read methods
- [ ] Checkpoint CRUD operations working
- [ ] Migration scripts tested and documented
- [ ] All storage tests passing
- [ ] @data-analyst approval on schema implementation

### Milestone: **Storage Ready** 💾

---

## Sprint 3: Core API & Features (5 days)

**Goal**: Implement checkpoint manager, retrieval updates, and core API methods

### Tasks

| ID | Task | Owner | Est | Dependencies |
|----|------|-------|-----|--------------|
| 3.1 | Implement `CheckpointManager` class | @dev | 1d | Sprint 2 |
| 3.2 | Add state size validation (1MB limit) | @dev | 0.25d | 3.1 |
| 3.3 | Implement sequence numbering logic | @dev | 0.25d | 3.1 |
| 3.4 | Implement branch tracking | @dev | 0.5d | 3.1 |
| 3.5 | Update retrieval engine with `scope` param | @dev | 0.5d | Sprint 2 |
| 3.6 | Implement `_build_scope_filter()` | @dev | 0.5d | 3.5 |
| 3.7 | Add workflow_outcomes to retrieval | @dev | 0.5d | 3.5 |
| 3.8 | Add `checkpoint()` to ALMA core | @dev | 0.25d | 3.1 |
| 3.9 | Add `get_resume_point()` to ALMA core | @dev | 0.25d | 3.1 |
| 3.10 | Add `merge_states()` to ALMA core | @dev | 0.25d | Sprint 1 reducers |
| 3.11 | Add `learn_from_workflow()` to ALMA core | @dev | 0.5d | 3.1, 3.5 |
| 3.12 | Add `link_artifact()` to ALMA core | @dev | 0.25d | Sprint 2 |
| 3.13 | Checkpoint manager unit tests | @qa | 0.5d | 3.1-3.4 |
| 3.14 | Concurrent checkpoint tests | @qa | 0.5d | 3.1 |
| 3.15 | Retrieval integration tests | @qa | 0.5d | 3.5-3.7 |
| 3.16 | Core API integration tests | @qa | 0.5d | 3.8-3.12 |

### Definition of Done
- [ ] `CheckpointManager` fully functional with size limits
- [ ] Retrieval respects `RetrievalScope` filtering
- [ ] All ALMA core methods implemented
- [ ] Concurrent checkpoint tests passing
- [ ] Integration tests achieving >80% coverage
- [ ] @architect approval on API implementation

### Milestone: **Core API Complete** ⚡

---

## Sprint 4: Polish & Launch (3 days)

**Goal**: MCP tools, documentation, final testing, and release

### Tasks

| ID | Task | Owner | Est | Dependencies |
|----|------|-------|-----|--------------|
| 4.1 | Add `alma_checkpoint` MCP tool | @dev | 0.25d | Sprint 3 |
| 4.2 | Add `alma_get_resume_point` MCP tool | @dev | 0.25d | Sprint 3 |
| 4.3 | Add `alma_merge_states` MCP tool | @dev | 0.25d | Sprint 3 |
| 4.4 | Add `alma_learn_workflow` MCP tool | @dev | 0.25d | Sprint 3 |
| 4.5 | Update `alma_retrieve` with scope param | @dev | 0.25d | Sprint 3 |
| 4.6 | MCP tools unit tests | @qa | 0.25d | 4.1-4.5 |
| 4.7 | Update README with workflow docs | @dev | 0.5d | Sprint 3 |
| 4.8 | Write migration guide | @dev | 0.25d | Sprint 2 |
| 4.9 | Create `agtestari_integration.py` example | @dev | 0.5d | Sprint 3 |
| 4.10 | FileBasedStorage update (low priority) | @dev | 0.5d | Sprint 2 |
| 4.11 | Full regression test suite | @qa | 0.5d | All |
| 4.12 | Performance benchmarks | @qa | 0.25d | All |
| 4.13 | Final code review | @architect | 0.25d | All |
| 4.14 | Release preparation | @devops | 0.25d | 4.13 |

### Definition of Done
- [ ] All MCP tools working and tested
- [ ] README updated with workflow context documentation
- [ ] Migration guide complete
- [ ] Example integration script runs successfully
- [ ] All tests passing (>80% coverage overall)
- [ ] Performance targets met (checkpoint <50ms, retrieval <200ms)
- [ ] Release tagged and documented

### Milestone: **v0.6.0 Released** 🚀

---

## Dependency Graph

```
Sprint 0 (Dana)
     │
     ├─► 0.2 Checkpoint Schema ──────────────────────────────┐
     ├─► 0.3 Workflow Outcome Schema ────────────────────────┤
     ├─► 0.4 Artifact Links Schema ──────────────────────────┤
     │                                                       │
     ▼                                                       ▼
Sprint 1 (@dev)                                    Sprint 2 (@dev)
     │                                                       │
     ├─► 1.1-1.6 Core Types ──────────────────────►  2.1-2.4 Base Interface
     ├─► 1.8-1.10 Reducers                           2.5-2.7 SQLite
     │                                               2.8-2.11 PostgreSQL
     │                                               2.12 Cosmos
     │                                               2.13-2.14 Migrations
     │                                                       │
     └───────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                          Sprint 3 (@dev)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              3.1-3.4      3.5-3.7      3.8-3.12
              Checkpoint   Retrieval    Core API
              Manager      Updates      Methods
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                          Sprint 4 (@dev)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              4.1-4.6      4.7-4.9      4.10-4.14
              MCP Tools    Docs &       Polish &
                          Examples      Release
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cloudflare Hyperdrive pgvector unavailable | Low | High | Fallback to app-level similarity (already in codebase) |
| Schema migration breaks existing data | Medium | High | Test on staging first, backup procedures documented |
| Concurrent checkpoint race conditions | Medium | Medium | Extensive concurrent tests in Sprint 3 |
| Sprint 2 storage work takes longer | Medium | Medium | Buffer in Sprint 3, can defer FileBasedStorage |
| @data-analyst (Dana) availability | Low | High | Schema designs front-loaded in Sprint 0 |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Checkpoint write latency | < 50ms p95 | Performance tests |
| Checkpoint read latency | < 20ms p95 | Performance tests |
| Scoped retrieval latency | < 200ms p95 | Performance tests |
| Test coverage (new code) | > 80% | pytest --cov |
| All existing tests pass | 100% | CI pipeline |
| Zero breaking changes | 0 | API compatibility tests |

---

## Communication Plan

| Event | Frequency | Participants | Format |
|-------|-----------|--------------|--------|
| Sprint Planning | Start of sprint | All agents | Sync meeting |
| Daily Standup | Daily | @dev, @qa, @pm | Async update |
| Sprint Review | End of sprint | All agents | Demo + discussion |
| Architecture Review | End of Sprint 1, 3 | @architect, @dev | Code review |
| Schema Review | Sprint 0 | @data-analyst, @architect | Design review |

---

## Approval

| Role | Agent | Approved | Date |
|------|-------|----------|------|
| Project Lead | @pm (Morgan) | ☐ | |
| Technical Lead | @architect (Aria) | ☐ | |
| Database Lead | @data-analyst (Dana) | ☐ | |
| Product Owner | @po | ☐ | |

---

## Next Actions

1. **@pm**: Share this plan with all agents for review
2. **@data-analyst (Dana)**: Begin schema design (Sprint 0 tasks)
3. **@devops**: Provision Cloudflare infrastructure
4. **@architect**: Review and approve sprint plan
5. **All**: Sprint 0 kickoff meeting

---

*— Morgan, planejando o futuro 📊*
