# ALMA - Agent Learning Memory Architecture

## Product Requirements Document

**Version**: 1.0
**Date**: 2026-01-23
**Author**: Claude Opus 4.5 + RBKunnela
**Status**: Draft - Pending Approval

---

## Executive Summary

ALMA (Agent Learning Memory Architecture) is a persistent memory system that enables AI agents to learn and improve over time without model weight updates. By structuring memory into scoped layers (heuristics, outcomes, preferences, domain knowledge), agents accumulate intelligence across runs while remaining constrained to their defined responsibilities.

### Core Insight

> "Agents that get better over time will be able to log and update their strategies, their heuristics, their domain knowledge if we can construct our memory systems appropriately and scope them right."

This is not training with weights. This happens entirely in memory instruction layers.

---

## Problem Statement

### Current State
- AI agents repeat the same mistakes across sessions
- No mechanism to capture "what worked" vs "what failed"
- User preferences must be re-explained every conversation
- Domain knowledge doesn't accumulate
- Context balloons when trying to inject historical knowledge

### Desired State
- Agents improve with each interaction
- Successful strategies are remembered and reused
- User preferences persist without re-explanation
- Domain knowledge accumulates within scoped boundaries
- Only relevant memory slices are injected per-call

---

## Goals

| Goal | Success Metric |
|------|----------------|
| Agents learn from outcomes | Heuristic retrieval improves task success rate by 20%+ |
| Memory stays scoped | Zero cross-domain knowledge bleed between agents |
| Context efficiency | Per-call memory injection < 2000 tokens |
| Performance | Memory retrieval < 200ms p95 |
| Security | Zero secrets in config files; all via Key Vault |

---

## Non-Goals (Out of Scope)

- Fine-tuning or weight updates
- Real-time collaborative memory between agents
- Memory sharing across organizations
- Automated heuristic generation without validation

---

## Architecture Overview

### Memory Stack

```
┌─────────────────────────────────────────────────────────┐
│                   PER-CALL CONTEXT                      │
│  (minimal - only inject relevant slices)                │
└─────────────────────────────────────────────────────────┘
                         ↑ retrieval
┌─────────────────────────────────────────────────────────┐
│              MEMORY RETRIEVAL LAYER                     │
│  Semantic search + recency + success-weighting + cache  │
└─────────────────────────────────────────────────────────┘
                         ↑
┌──────────────┬──────────────┬──────────────────────────┐
│   HEURISTICS │   OUTCOMES   │   USER PREFERENCES       │
│   MEMORY     │   MEMORY     │   MEMORY                 │
├──────────────┼──────────────┼──────────────────────────┤
│ "When X,     │ "Task Y      │ "User prefers verbose    │
│  strategy A  │  succeeded   │  output, dislikes        │
│  works 80%"  │  with Z"     │  emojis in code"         │
└──────────────┴──────────────┴──────────────────────────┘
                         ↑ consolidation + forgetting
┌─────────────────────────────────────────────────────────┐
│              RAW EXECUTION LOGS                         │
│  Every run's decisions, outcomes, feedback              │
└─────────────────────────────────────────────────────────┘
```

### Azure Resources

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| Azure Cosmos DB | Vector storage for memories | Containers partitioned by project_id/agent_name |
| Azure OpenAI | Embedding generation | text-embedding-3-small (1536 dims) |
| Azure Key Vault | Secrets management | All connection strings, API keys |

### Memory Types

| Type | Stores | Update Frequency | Partition Key |
|------|--------|------------------|---------------|
| Heuristics | "When X, do Y" learned rules | After validation (min 3 occurrences) | project_id |
| Outcomes | Task success/failure records | Every completed task | agent_name |
| User Preferences | Communication style, constraints | On user correction | user_id |
| Domain Knowledge | Accumulated domain facts | Curated updates | project_id |
| Anti-patterns | What NOT to do | After validated failures | agent_name |

---

## Technical Design

### Core Components

```
alma/
├── retrieval/
│   ├── engine.py         # Semantic search + ranking
│   ├── cache.py          # Redis/in-memory caching
│   └── scoring.py        # Recency + success weighting
├── learning/
│   ├── protocols.py      # When/how to update memories
│   ├── validation.py     # Scope checking before commit
│   └── forgetting.py     # Prune low-confidence/stale
├── storage/
│   ├── base.py           # Abstract interface
│   ├── azure_cosmos.py   # Production storage
│   ├── sqlite_local.py   # Local dev with FAISS
│   └── file_based.py     # Simplest fallback
├── integration/
│   └── claude_agents.py  # Hooks for Helena, Victor, etc.
└── config/
    ├── loader.py         # Env vars + Key Vault
    └── defaults.yaml     # Sensible defaults
```

### Learning Protocol

```python
def learn_from_execution(agent, task, outcome, feedback=None):
    """
    1. VALIDATE scope - agent can only learn within defined boundaries
    2. CHECK occurrence count - need min 3 similar outcomes for heuristic
    3. COMPARE with existing - don't contradict high-confidence heuristics
    4. COMMIT if validated - write to appropriate memory store
    5. TRIGGER forgetting - prune if memory exceeds thresholds
    """
```

### Retrieval Protocol

```python
def get_relevant_memories(task_description, agent, user, project_id):
    """
    1. EMBED task description
    2. QUERY each memory type with vector similarity
    3. FILTER by scope (agent's allowed domains)
    4. RANK by recency + success_rate + similarity
    5. CACHE result for repeated queries
    6. RETURN compact MemorySlice (< 2000 tokens)
    """
```

### Scoping Configuration

```yaml
# .alma/config.yaml (per-project)
alma:
  project_id: "agenticTestari-app"
  storage: azure  # or "local" for dev

  agents:
    helena:
      can_learn:
        - testing_strategies
        - selector_patterns
        - ui_component_patterns
      cannot_learn:
        - backend_logic
        - database_operations
      min_occurrences_for_heuristic: 3

    victor:
      can_learn:
        - api_testing_patterns
        - database_validation
        - error_handling_patterns
      cannot_learn:
        - frontend_logic
        - ui_testing
      min_occurrences_for_heuristic: 3
```

---

## Implementation Phases

### Phase 1: Core Abstractions (Week 1)
- [ ] Define storage interface (`storage/base.py`)
- [ ] Define memory types (Heuristic, Outcome, Preference, etc.)
- [ ] Define learning protocol interface
- [ ] Set up pytest infrastructure

### Phase 2: Local Storage (Week 1)
- [ ] Implement SQLite storage with FAISS for vectors
- [ ] Implement file-based fallback
- [ ] Write unit tests with mock data

### Phase 3: Retrieval Engine (Week 2)
- [ ] Implement embedding generation (local model for dev)
- [ ] Implement semantic search
- [ ] Implement scoring (recency + success weighting)
- [ ] Implement caching layer
- [ ] Integration tests

### Phase 4: Learning Protocols (Week 2)
- [ ] Implement scope validation
- [ ] Implement occurrence counting
- [ ] Implement heuristic extraction
- [ ] Implement forgetting mechanism
- [ ] Validation tests

### Phase 5: Agent Integration - Helena + Victor (Week 3)
- [ ] Create integration hooks for Claude Code agents
- [ ] Integrate Helena (frontend QA)
- [ ] Integrate Victor (backend QA)
- [ ] End-to-end testing with real tasks
- [ ] Document integration patterns

### Phase 6: Azure Integration (Week 3-4)
- [ ] Set up Azure Cosmos DB with vector search
- [ ] Set up Azure Key Vault
- [ ] Implement `azure_cosmos.py` storage
- [ ] Implement Key Vault config loader
- [ ] Migration from local to Azure

### Phase 7: Cache Layer (Week 4)
- [ ] Implement Redis cache (optional) or in-memory
- [ ] Cache invalidation strategies
- [ ] Performance benchmarks

### Phase 8: Forgetting Mechanism (Week 4)
- [ ] Implement confidence decay
- [ ] Implement staleness detection
- [ ] Implement pruning policies
- [ ] Automated cleanup jobs

---

## Security Requirements

| Requirement | Implementation |
|-------------|----------------|
| No secrets in code/config | All secrets via Azure Key Vault |
| Scoped access | Each agent only accesses own memories |
| Audit logging | All memory writes logged with timestamp/agent |
| Data isolation | Project memories isolated via partition keys |

---

## Success Criteria

### MVP (Phases 1-5)
1. Helena and Victor can write outcomes to local storage
2. Agents can retrieve relevant heuristics via semantic search
3. Scope validation prevents cross-domain learning
4. All tests pass (>80% coverage)

### Production (Phases 6-8)
1. Azure Cosmos DB operational with vector search
2. Secrets managed via Key Vault
3. Memory retrieval < 200ms p95
4. Forgetting mechanism prevents unbounded growth

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory bloat | Slow retrieval, high costs | Forgetting mechanism, TTL policies |
| Stale heuristics | Wrong advice | Confidence decay, validation checks |
| Scope creep | Agents learn outside boundaries | Config-enforced scopes, validation |
| Azure costs | Budget overrun | Start with local, monitor usage |

---

## Appendix: Memory Schema Examples

### Heuristic
```json
{
  "id": "heur_001",
  "agent": "helena",
  "project_id": "agenticTestari-app",
  "condition": "form with multiple required fields",
  "strategy": "test happy path first, then individual field validation",
  "confidence": 0.85,
  "occurrence_count": 7,
  "last_validated": "2026-01-20T14:30:00Z",
  "embedding": [0.123, -0.456, ...]
}
```

### Outcome
```json
{
  "id": "out_042",
  "agent": "victor",
  "project_id": "agenticTestari-app",
  "task_type": "api_validation",
  "task_description": "Validate authentication endpoint returns 401 for invalid token",
  "success": true,
  "strategy_used": "Test with expired token, malformed token, missing token",
  "duration_ms": 1250,
  "timestamp": "2026-01-23T10:15:00Z",
  "embedding": [0.789, -0.012, ...]
}
```

### User Preference
```json
{
  "id": "pref_003",
  "user_id": "RBKunnela",
  "category": "communication",
  "preference": "No emojis in code or documentation",
  "source": "explicit_instruction",
  "timestamp": "2026-01-15T09:00:00Z"
}
```

---

## Approval

- [ ] Architecture approved
- [ ] Security review passed
- [ ] Azure resources approved
- [ ] Ready for Phase 1

**Approver**: ________________
**Date**: ________________
