# ALMA + MemPalace Merge Strategy

## Phase 0 Status Check

Before merging, you need to understand what you're actually working with.

### MemPalace Codebase Inventory

**Storage Layer (Ready to adopt):**
- `mempalace/backends/chroma.py` - ChromaDB wrapper (production-hardened)
- `mempalace/palace.py` - Collection access patterns
- `mempalace/config.py` - Configuration management

**Retrieval & Search (Ready to adopt):**
- `mempalace/searcher.py` - Semantic search with metadata filtering
- `mempalace/query_sanitizer.py` - Input safety
- `mempalace/palace_graph.py` - Graph traversal for wings/rooms/halls

**Data Processing (Ready to adopt):**
- `mempalace/convo_miner.py` - Extract conversations from chat exports
- `mempalace/miner.py` - File-based memory mining
- `mempalace/entity_detector.py` - NER and entity linking
- `mempalace/entity_registry.py` - Deduplication and linking
- `mempalace/normalize.py` - Text normalization
- `mempalace/dedup.py` - Deduplication logic
- `mempalace/spellcheck.py` - Typo correction

**Compression & Dialects (Consider adopting):**
- `mempalace/dialect.py` - AAAK abbreviation system (experimental, 84.2% retrieval)
- `mempalace/split_mega_files.py` - Large file handling

**Knowledge Graph (Partial overlap with ALMA):**
- `mempalace/knowledge_graph.py` - Entity relationships (SQLite backend only)
- `mempalace/layers.py` - Multi-layer memory (L0 critical facts, L1 context)

**Integration (Ready to adopt):**
- `mempalace/mcp_server.py` - MCP server (55KB, fully featured)
- `mempalace/hooks/` - Automatic save hooks for Claude Code

**Utilities (Ready to adopt):**
- `mempalace/migrate.py` - Data migration
- `mempalace/repair.py` - Database consistency repair
- `mempalace/exporter.py` - Export memories

**CLI (Consider replacing with ALMA's):**
- `mempalace/cli.py` - Command-line interface
- `mempalace/hooks_cli.py` - Hooks management
- `mempalace/instructions_cli.py` - Instructions management

**Testing:**
- 11,286 lines of tests with high coverage
- Benchmarks against LongMemEval, ConvoMem, etc.
- Fixtures and utilities for integration testing

---

## Architecture Merge Plan

### Level 1: Keep MemPalace's Storage Layer, Add ALMA's Governance

```
New Architecture (Target)

┌─────────────────────────────────────────────────────────────────┐
│                    ALMA Governance Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Agent Scoping  │  Anti-Pattern Learning  │  Multi-Agent Share  │
├─────────────────────────────────────────────────────────────────┤
│           MemPalace Palace Structure (Wings/Halls/Rooms)         │
├─────────────────────────────────────────────────────────────────┤
│         Five Memory Types (ALMA) → Palace Organization          │
│  Heuristic → Hall::Heuristics                                   │
│  Outcome → Hall::Outcomes                                       │
│  Preference → Hall::Preferences (scoped by agent)               │
│  Domain Knowledge → Hall::Facts                                 │
│  Anti-Pattern → Hall::AntiPatterns (scoped by agent)            │
├─────────────────────────────────────────────────────────────────┤
│              ChromaDB + Entity Registry (MemPalace)             │
├─────────────────────────────────────────────────────────────────┤
│  Chroma (vectors) │ SQLite (entity graph) │ Files (raw text)    │
└─────────────────────────────────────────────────────────────────┘
```

**Decision Rationale:**
- Keep MemPalace's ChromaDB because it's proven at 96.6% accuracy
- Add ALMA's scoped learning on top without replacing search
- Palace structure becomes the enforcer of ALMA scopes
- Risk: Scoping adds latency if not indexed properly

---

### Level 2: Storage Backend Unification

**MemPalace Reality:**
- Only ChromaDB is tested
- PostgreSQL interface exists but untested
- Qdrant, Pinecone, Chroma interfaces are unused

**ALMA Reality:**
- 6 backend interfaces designed, none hardened

**What to do:**
1. Keep ChromaDB as default (proven)
2. Implement PostgreSQL+pgvector as second backend (high priority)
3. Add Qdrant after PostgreSQL works at 90%+ on benchmarks
4. Archive other backends as "experimental"

**Why this order:**
- PostgreSQL: standard production database, pgvector is stable
- Qdrant: better UX than Pinecone for self-hosted
- Others: can be added later once architecture stabilizes

---

### Level 3: Memory Type Consolidation

**MemPalace Model:**
```
Drawer (drawerID, content, wing, hall, room, embeddings, metadata)
```

**ALMA Model:**
```
Memory (type, agent, category, content, embedding, metadata)
type = heuristic | outcome | preference | domain_knowledge | anti_pattern
metadata = {agent_id, can_learn, cannot_learn, shared_from, ...}
```

**Merge Strategy:**
```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  
  -- MemPalace structure
  wing TEXT NOT NULL,           -- person/project
  hall TEXT NOT NULL,           -- heuristics|outcomes|preferences|facts|antipatterns
  room TEXT NOT NULL,           -- topic name
  
  -- ALMA governance
  agent_id TEXT NOT NULL,       -- which agent "learned" this
  memory_type TEXT NOT NULL,    -- heuristic|outcome|preference|domain_knowledge|anti_pattern
  
  -- Content & search
  content TEXT NOT NULL,        -- verbatim (MemPalace approach)
  embedding VECTOR(384),        -- ChromaDB embedding
  
  -- Metadata
  metadata JSONB,               -- {can_learn, cannot_learn, shared_from, ...}
  created_at TIMESTAMP,
  accessed_count INT DEFAULT 0,
  last_accessed TIMESTAMP,
  
  -- Lineage
  source_file TEXT,
  source_mtime FLOAT,
  merged_from TEXT[],           -- consolidation provenance
  
  FOREIGN KEY (agent_id) REFERENCES agents(id),
  INDEX ON (wing, hall, room),  -- palace navigation
  INDEX ON (agent_id, memory_type),  -- ALMA scoping
);

CREATE TABLE agents (
  id TEXT PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  domain TEXT NOT NULL,         -- coding|research|sales|support|general
  can_learn TEXT[],             -- categories allowed
  cannot_learn TEXT[],          -- hard boundaries
  inherit_from TEXT[],          -- parent agent IDs
  share_with TEXT[],            -- child/peer agent IDs
  created_at TIMESTAMP,
);
```

**Implementation Plan:**
1. Extend MemPalace's drawerID to include agent scoping
2. Add memory_type classification (can auto-detect from hall)
3. Migrate existing MemPalace drawers → memories with agent=DEFAULT
4. Keep all MemPalace metadata intact (source_file, source_mtime, etc.)

---

### Level 4: Retrieval with Governance

**Current MemPalace Retrieval:**
```python
search_memories(query, wing=None, hall=None, room=None, top_k=5)
# Returns: drawers with highest cosine similarity
```

**New ALMA-aware Retrieval:**
```python
search_memories(
    query,
    agent_id,                 # Required: which agent is asking
    include_shared=False,     # Include shared_from memories?
    memory_types=None,        # Filter by type (default: all agent's types)
    wing=None,                # Palace navigation
    top_k=5
)
# Filter logic:
# 1. Semantic search in ChromaDB
# 2. Filter: agent_id MATCHES can_learn OR agent_id IN shared_agents
# 3. Filter: memory_type NOT IN cannot_learn
# 4. Sort: similarity * recency_boost * access_frequency
# 5. Return: top_k
```

**Risk Analysis:**
- Extra filtering might hurt latency (need benchmarks)
- Solution: Index on (agent_id, memory_type) + pre-cache scopes

---

### Level 5: Learning with Scopes

**Current MemPalace Learning:**
```python
learn(agent_id, task, outcome, strategy_used)
# Creates: new drawer in hall_outcomes with wing={agent_id}
```

**New ALMA-aware Learning:**
```python
learn(
    agent_id,
    task,
    outcome,                  # success|partial|failure
    strategy_used,
    category=None,            # infer from task or require explicit
    anti_pattern=False        # is this a negative example?
)
# Validation:
# 1. Check category IN agents.can_learn[agent_id]
# 2. Check category NOT IN agents.cannot_learn[agent_id]
# 3. If anti_pattern: store with explicit anti_pattern flag
# 4. Trigger consolidation if memory_count > threshold
```

**Critical Decision:**
Auto-classification (task → category) vs Manual assignment

- MemPalace: no classification (all verbatim)
- ALMA: explicit classification (agent learns "X")
- Compromise: optional LLM classifier (Claude Haiku) if category not provided
  - Cost: ~$0.001 per learn() call
  - Benefit: catch category violations early

---

## Execution Roadmap

### Sprint 1: Storage Unification (Week 1)

**Goal:** Single unified schema that supports both MemPalace and ALMA operations

**Tasks:**
1. Design schema (use SQL above)
2. Create migration script: MemPalace drawers → ALMA memories
3. Test migration on MemPalace's test dataset
4. Verify all MemPalace tests pass with new schema
5. Benchmark: retrieval speed unchanged (target: <100ms for top 5)

**Definition of Done:**
- All MemPalace tests pass (11,286 lines)
- ConvoMem score >= 92% (was 92.9%)
- LongMemEval score >= 95% (was 96.6%, small regression acceptable)

**Risk:** Schema changes break existing integrations (mitigate: backward compat layer)

---

### Sprint 2: Agent Governance Layer (Week 2)

**Goal:** Implement scoped learning and retrieval

**Tasks:**
1. Create `alma/governance/agent.py` - Agent model + scope validation
2. Create `alma/governance/scopes.py` - can_learn/cannot_learn enforcement
3. Modify retrieval to filter by agent scopes
4. Modify learning to validate against scopes
5. Add tests: can_learn enforcement, cannot_learn rejection
6. Benchmark: retrieval latency with scoping (target: <150ms for top 5)

**Definition of Done:**
- Agent scoping tests pass
- Scope violations caught and rejected
- Retrieval latency < 150ms with filtering
- ConvoMem >= 90%, LongMemEval >= 94%

**Risk:** Filtering by agent_id adds significant latency

---

### Sprint 3: Multi-Agent Sharing (Week 3)

**Goal:** Implement inherit_from and share_with

**Tasks:**
1. Create `alma/governance/sharing.py` - shared memory retrieval
2. Implement inherit_from query (fetch memories from parent agents)
3. Implement share_with query (fetch memories shared with this agent)
4. Add metadata tracking: shared_from field
5. Test: junior agent inherits senior agent's memories
6. Benchmark: multi-agent retrieval performance

**Definition of Done:**
- Sharing queries work
- Origin tracking (shared_from) verified
- ConvoMem >= 90%, LongMemEval >= 94%
- Junior agent retrieval includes senior's shared memories

**Risk:** Multi-agent queries could be expensive (need careful indexing)

---

### Sprint 4: Anti-Pattern Learning (Week 4)

**Goal:** Explicit capture of "what not to do"

**Tasks:**
1. Create `alma/learning/anti_patterns.py` - anti-pattern capture and retrieval
2. Extend learn() to accept anti_pattern=True
3. Create retrieval variant: get_antipatterns(agent_id, category, context)
4. Inject anti-patterns into prompts (format: "DO NOT: {pattern} because {reason}")
5. Test: anti-pattern blocking in agent decisions
6. Benchmark: does anti-pattern awareness improve success rate?

**Definition of Done:**
- Anti-patterns storable and retrievable
- Prompt injection works
- A/B test: agent with anti-patterns vs without (need test dataset)

**Risk:** Anti-patterns require manual curation or LLM classification (cost + latency)

---

### Sprint 5: Memory Consolidation (Week 5)

**Goal:** Deduplicate and merge similar memories with ALMA governance aware

**Tasks:**
1. Adapt MemPalace's consolidation logic to new schema
2. Ensure consolidation respects agent scopes (don't merge across domains)
3. Add provenance tracking: merged_from field
4. Test: consolidation doesn't hurt retrieval
5. Benchmark: merged vs unmerged retrieval quality

**Definition of Done:**
- Consolidation preserves scope boundaries
- ConvoMem >= 92%, LongMemEval >= 95%
- No regression from consolidation

---

### Sprint 6: MCP Integration & Shipping (Week 6)

**Goal:** Expose merged system to Claude Code

**Tasks:**
1. Adapt MemPalace's mcp_server.py to new schema
2. Add MCP tools: alma_retrieve, alma_learn, alma_manage_scopes
3. Test MCP integration with Claude Code
4. Create hooks for automatic context capture
5. Release v1.0.0 to PyPI
6. Documentation + examples

**Definition of Done:**
- MCP server passes all MemPalace tests
- Claude Code integration works
- v1.0.0 published
- README updated with benchmark results

---

## Critical Success Metrics

### Must-Have
- LongMemEval >= 95% (accept 1-2% regression for ALMA features)
- ConvoMem >= 90% (accept 2-3% regression)
- Retrieval latency < 200ms (with scoping overhead)
- All MemPalace tests pass

### Nice-to-Have
- PostgreSQL backend implemented and tested
- Anti-pattern A/B test showing improvement
- Multi-agent sharing deployed with 2+ real agents
- Memory consolidation reduces storage by 10%+

### Stretch Goals
- Qdrant backend working and benchmarked
- Domain-specific schemas working for 3+ domains
- User-facing dashboard showing memory growth/access patterns

---

## Code Structure Post-Merge

```
alma-memory/
├── alma/
│   ├── core/                          # MemPalace core (unchanged)
│   │   ├── storage/
│   │   │   ├── chroma.py              # from MemPalace
│   │   │   ├── postgres.py            # NEW - pgvector backend
│   │   │   └── abstract.py
│   │   ├── search/
│   │   │   ├── searcher.py            # from MemPalace
│   │   │   ├── query_sanitizer.py     # from MemPalace
│   │   │   └── palace_graph.py        # from MemPalace
│   │   ├── mining/
│   │   │   ├── convo_miner.py         # from MemPalace
│   │   │   ├── entity_detector.py     # from MemPalace
│   │   │   ├── normalize.py           # from MemPalace
│   │   │   └── dedup.py               # from MemPalace
│   │   └── config.py                  # from MemPalace
│   │
│   ├── governance/                    # ALMA scoping (NEW)
│   │   ├── agent.py
│   │   ├── scopes.py
│   │   ├── sharing.py
│   │   └── __init__.py
│   │
│   ├── learning/                      # ALMA learning (NEW)
│   │   ├── anti_patterns.py
│   │   ├── consolidation.py           # from ALMA + enhanced
│   │   └── __init__.py
│   │
│   ├── integration/                   # Adapters
│   │   ├── mcp_server.py              # from MemPalace + ALMA tools
│   │   ├── hooks.py                   # from MemPalace
│   │   └── __init__.py
│   │
│   ├── domain/                        # Pre-built schemas
│   │   ├── coding.py
│   │   ├── research.py
│   │   ├── sales.py
│   │   └── __init__.py
│   │
│   ├── __init__.py                    # Main ALMA class
│   └── cli.py                         # CLI (MemPalace + new commands)
│
├── tests/
│   ├── test_storage.py                # MemPalace storage tests
│   ├── test_retrieval.py              # MemPalace retrieval tests
│   ├── test_governance.py             # NEW - scoping tests
│   ├── test_learning.py               # NEW - ALMA learning tests
│   ├── test_anti_patterns.py          # NEW
│   ├── test_mcp_integration.py        # MemPalace MCP tests
│   ├── conftest.py
│   └── fixtures/
│       └── migrations/                # Test data + migration scripts
│
├── benchmarks/
│   ├── longmemeval/                   # from MemPalace
│   ├── convomem/                      # from MemPalace
│   ├── anti_patterns/                 # NEW - test anti-pattern effectiveness
│   ├── scoping/                       # NEW - test scope enforcement
│   └── multi_agent/                   # NEW - test sharing overhead
│
├── docs/
│   ├── architecture.md                # NEW - merged design
│   ├── migration.md                   # Migration guide from pure MemPalace
│   ├── governance.md                  # Agent scoping guide
│   ├── examples/
│   │   ├── single_agent.py            # Helena example
│   │   ├── multi_agent.py             # Junior + Senior example
│   │   └── domain_specific.py         # Domain factory example
│   └── benchmarks.md                  # Results from both systems
│
├── pyproject.toml                     # Updated with new dependencies
├── README.md                          # Updated with merged feature set
└── CHANGELOG.md                       # Document the merge
```

---

## What Gets Cut/Archived

**From MemPalace (lower priority):**
- AAAK compression dialect (too experimental, 12.4% regression)
- Room detector local (can be readded later)
- Split mega files (edge case, keep simple first)

**From ALMA (unproven):**
- Event webhooks (too much surface area, MVP doesn't need async)
- TypeScript SDK (Python first, JS later)
- Domain factory pre-built schemas (except coding - ship with one domain)
- Graph memory (Neo4j, Memgraph, Kuzu) - keep in-memory only for MVP

**Result:** Smaller, faster-to-ship core with everything you need for scoped learning

---

## Risk Assessment

### High Risk
- Scope filtering adds retrieval latency → mitigate with indexing strategy
- Multi-agent queries might be expensive → pre-cache sharing graphs
- Consolidation could break retrieval → test thoroughly

**Mitigation:** Benchmark after each sprint; if regression > 2%, revert

### Medium Risk
- Migration from MemPalace drawers could lose metadata → backup first
- Agent classification (for anti-patterns) adds LLM cost → make optional

**Mitigation:** Test migration on small dataset first; implement LLM fallback as optional

### Low Risk
- PostgreSQL backend not ready for MVP → keep Chroma default
- Domain factory limited in V1 → extend after shipping

---

## First Cut: MVP (6 weeks)

**What ships in v1.0:**
- MemPalace retrieval (96%+) + ALMA scoping (new)
- ChromaDB + Chroma backend
- Agent scopes (can_learn/cannot_learn)
- Basic multi-agent sharing (inherit_from/share_with)
- Anti-pattern learning (manual annotation)
- Memory consolidation (scope-aware)
- MCP server (Claude Code integration)
- PyPI release

**What's deferred to v1.1:**
- PostgreSQL backend
- LLM-based anti-pattern classification
- Domain factory
- Qdrant backend
- Event webhooks
- TypeScript SDK

**Timeline:** 6 weeks if you're focused

---

## Claude Code Setup

**To use this in Claude Code, you need:**

```bash
# Clone both repos
git clone https://github.com/RBKunnela/ALMA-memory.git alma-work
cd alma-work

# Add MemPalace as subtree (preserves history)
git subtree add --prefix mempalace https://github.com/MemPalace/mempalace.git main

# Or copy for faster iteration (loses history)
cp -r /path/to/mempalace-repo ./mempalace-reference
```

**Then you can:**
1. Build new schema in `alma/schema.py`
2. Create migration script: `alma/migrate.py`
3. Adapt MemPalace searcher: `alma/core/search/searcher.py`
4. Build governance layer: `alma/governance/`
5. Run benchmarks from both systems

---

## Next Steps

1. **Read MemPalace's test suite** (11K lines) - it's your specification
2. **Read MemPalace's mcp_server.py** (55KB) - understand the integration
3. **Run MemPalace's benchmarks** - get baseline (96.6% LongMemEval)
4. **Start with schema migration** - Sprint 1 above
5. **Do NOT redesign architecture** - you did that already. Execute.

The merge will work if you:
- Keep MemPalace's retrieval (proven)
- Add ALMA's governance (unproven but valuable)
- Benchmark ruthlessly after each change
- Ship when you hit 95%+ on both benchmarks

Do not get fancy. Do not add features. Do not redesign the graph layer.

Merge. Benchmark. Ship.
