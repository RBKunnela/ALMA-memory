# Week 1 Execution Guide - Storage Schema & Migration

## The Goal This Week
Get MemPalace's data working in ALMA's schema while keeping 96%+ retrieval accuracy.

This is not design work. This is implementation. You already designed it. Now execute.

---

## Task 1: Understand MemPalace's Current Schema (Day 1)

### What MemPalace Stores

**ChromaDB Collection (where the action is):**
```python
# mempalace/mcp_server.py line ~400
collection.add(
    ids=[drawer_id],
    embeddings=[embedding],
    documents=[content],
    metadatas=[{
        "wing": "kai",
        "hall": "events",
        "room": "auth-migration",
        "source_file": "/path/to/export.json",
        "source_mtime": 1698765432.0,
        "created_at": "2024-01-15T10:30:00Z",
        "content_length": 1234,
    }]
)
```

**The Drawer ID:**
```
format: {wing}/{hall}/{room}/{uuid4}
example: "kai/events/auth-migration/a1b2c3d4-e5f6-7890"
```

**MemPalace's Hall Types (from config defaults):**
```python
HALLS = [
    "facts",          # what happened
    "decisions",      # what was decided
    "advice",         # what was recommended
    "code",           # code snippets
    "errors",         # what went wrong
    "patterns",       # repeated behaviors
    "quotes",         # exact memorable statements
    "todos",          # action items
]
```

**SQLite Knowledge Graph (secondary storage):**
```python
# mempalace/knowledge_graph.py - optional entity relationships
# Not critical for MVP - skip for Week 1
```

### Your Task Today
1. Open `mempalace-repo/tests/test_mcp_server.py` (30KB file)
2. Find the test cases that create drawers
3. Trace one drawer creation from storage → ChromaDB → query
4. Document what you find in `schema_notes.md`

**Deliverable by EOD:**
```
schema_notes.md
- MemPalace drawer structure (copy/paste from code)
- Example drawer IDs
- Hall types used
- Metadata fields present
- ChromaDB collection schema
```

---

## Task 2: Design ALMA's Memory Table (Day 1-2)

### Starting Point (from merge strategy)
```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  
  -- MemPalace structure (KEEP)
  wing TEXT NOT NULL,
  hall TEXT NOT NULL,
  room TEXT NOT NULL,
  
  -- ALMA governance (ADD)
  agent_id TEXT NOT NULL,
  memory_type TEXT NOT NULL,
  
  -- Content (CHANGE SLIGHTLY)
  content TEXT NOT NULL,
  embedding VECTOR(384),
  
  -- Metadata (EXTEND)
  metadata JSONB,
  created_at TIMESTAMP,
  
  FOREIGN KEY (agent_id) REFERENCES agents(id),
);
```

### Your Task
1. Decide: SQLite or PostgreSQL for Week 1?
   - **Answer:** SQLite (MemPalace uses it, Chroma uses it, no new dependencies)
   - Add PostgreSQL in Sprint 2

2. Create `alma/schema.py`:
```python
# alma/schema.py
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime

@dataclass
class Memory:
    """Unified memory model - MemPalace structure + ALMA governance"""
    
    # Identifiers
    id: str                      # UUID, unique across system
    
    # Palace structure (from MemPalace)
    wing: str                    # person/project name
    hall: str                    # facts|decisions|advice|code|errors|patterns|quotes|todos|heuristics|outcomes|preferences|antipatterns
    room: str                    # specific topic
    
    # ALMA governance
    agent_id: str               # which agent created/learned this
    memory_type: str            # heuristic|outcome|preference|domain_knowledge|anti_pattern
    
    # Content
    content: str                # verbatim text
    embedding: Optional[List[float]] = None  # 384-dim from all-MiniLM-L6-v2
    
    # Retrieval tracking
    created_at: datetime = None
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Metadata
    metadata: Dict = None       # {can_learn, cannot_learn, shared_from, ...}
    
    # Lineage
    source_file: Optional[str] = None
    source_mtime: Optional[float] = None
    merged_from: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.merged_from is None:
            self.merged_from = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Agent:
    """ALMA Agent with scoping rules"""
    
    id: str
    name: str
    domain: str                 # coding|research|sales|support|general
    
    # Learning boundaries
    can_learn: List[str]        # ["testing_strategies", "selector_patterns"]
    cannot_learn: List[str]     # ["backend_logic", "database_queries"]
    
    # Multi-agent sharing
    inherit_from: List[str] = None  # [senior_dev_id]
    share_with: List[str] = None    # [junior_dev_id, qa_lead_id]
    
    created_at: datetime = None
    
    def __post_init__(self):
        if self.inherit_from is None:
            self.inherit_from = []
        if self.share_with is None:
            self.share_with = []
        if self.created_at is None:
            self.created_at = datetime.now()
```

3. Create `alma/storage/schema.sql`:
```sql
-- ALMA + MemPalace unified schema

CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    can_learn TEXT[],
    cannot_learn TEXT[],
    inherit_from TEXT[],
    share_with TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    
    -- Palace structure
    wing TEXT NOT NULL,
    hall TEXT NOT NULL,
    room TEXT NOT NULL,
    
    -- ALMA governance
    agent_id TEXT NOT NULL REFERENCES agents(id),
    memory_type TEXT NOT NULL,
    
    -- Content
    content TEXT NOT NULL,
    embedding BYTEA,  -- Store as binary for efficiency
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Lineage
    source_file TEXT,
    source_mtime REAL,
    merged_from TEXT[],
    
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Indexes for common queries
CREATE INDEX idx_memories_wing_hall_room ON memories(wing, hall, room);
CREATE INDEX idx_memories_agent_type ON memories(agent_id, memory_type);
CREATE INDEX idx_memories_wing_agent ON memories(wing, agent_id);
CREATE INDEX idx_memories_created ON memories(created_at DESC);
```

**Deliverable by EOD Day 2:**
```
alma/schema.py          - Memory + Agent dataclasses
alma/storage/schema.sql - Complete DDL
```

---

## Task 3: Create Migration Script (Day 2-3)

**Goal:** Convert MemPalace drawers → ALMA memories without losing any data

### The Migration Logic

```python
# alma/migrate.py

def migrate_mempalace_to_alma(
    chroma_path: str,
    alma_db_path: str,
    dry_run: bool = True
) -> dict:
    """
    Read from MemPalace's ChromaDB, write to ALMA's SQLite
    
    Args:
        chroma_path: path to MemPalace's .chroma directory
        alma_db_path: path for new ALMA database
        dry_run: if True, don't write, just report what would happen
    
    Returns:
        {
            "total_drawers": 1234,
            "migrated": 1234,
            "failed": 0,
            "errors": [],
            "hall_mapping": {...},
            "agent_mapping": {...},
        }
    """
    
    # Step 1: Connect to MemPalace's ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_collection("mempalace_drawers")
    
    # Step 2: Get all drawers
    drawers = collection.get(include=["embeddings", "documents", "metadatas"])
    
    # Step 3: Create ALMA database
    if not dry_run:
        init_alma_db(alma_db_path)
    
    # Step 4: Migrate each drawer
    stats = {
        "total_drawers": len(drawers["ids"]),
        "migrated": 0,
        "failed": 0,
        "errors": [],
        "hall_to_memory_type_mapping": {},
    }
    
    for drawer_id, embedding, content, metadata in zip(
        drawers["ids"],
        drawers["embeddings"],
        drawers["documents"],
        drawers["metadatas"]
    ):
        try:
            # Parse drawer ID: wing/hall/room/uuid
            parts = drawer_id.split("/")
            wing, hall, room = parts[0], parts[1], parts[2]
            
            # Map MemPalace hall → ALMA memory_type
            memory_type = map_hall_to_memory_type(hall)
            stats["hall_to_memory_type_mapping"][hall] = memory_type
            
            # Create memory object
            memory = Memory(
                id=drawer_id,
                wing=wing,
                hall=hall,
                room=room,
                agent_id="DEFAULT",  # All existing drawers are "DEFAULT" agent
                memory_type=memory_type,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                source_file=metadata.get("source_file") if metadata else None,
                source_mtime=metadata.get("source_mtime") if metadata else None,
            )
            
            if not dry_run:
                store_memory(memory, alma_db_path)
            
            stats["migrated"] += 1
            
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({
                "drawer_id": drawer_id,
                "error": str(e)
            })
    
    return stats


def map_hall_to_memory_type(hall: str) -> str:
    """
    MemPalace halls → ALMA memory types
    
    facts, decisions, advice, code, errors, patterns, quotes, todos
    ↓
    domain_knowledge, outcomes, preference, heuristic, anti_pattern
    """
    mapping = {
        "facts": "domain_knowledge",
        "decisions": "outcome",
        "advice": "preference",
        "code": "heuristic",
        "errors": "anti_pattern",
        "patterns": "heuristic",
        "quotes": "domain_knowledge",
        "todos": "outcome",
        # ALMA halls (if any already exist)
        "heuristics": "heuristic",
        "outcomes": "outcome",
        "preferences": "preference",
        "antipatterns": "anti_pattern",
    }
    return mapping.get(hall, "domain_knowledge")  # safe default
```

### Your Task Day 2-3
1. Create `alma/migrate.py` with migration logic above
2. Create `alma/storage/sqlite.py` - SQLite backend (adapt from MemPalace's pattern)
3. Create `tests/test_migration.py`:
   - Test: drawer → memory conversion preserves all metadata
   - Test: hall → memory_type mapping is correct
   - Test: embeddings are preserved
   - Test: source_file and source_mtime are kept

**Deliverable by EOD Day 3:**
```
alma/migrate.py
alma/storage/sqlite.py
tests/test_migration.py
```

---

## Task 4: Adapt MemPalace's Searcher (Day 3-4)

**Goal:** Make retrieval work with ALMA schema while keeping accuracy

### Current MemPalace Searcher (simplified)
```python
def search_memories(query, wing=None, hall=None, room=None, top_k=5):
    results = collection.query(
        query_embeddings=[embed(query)],
        where={
            "$and": [
                {"wing": {"$eq": wing}} if wing else {},
                {"hall": {"$eq": hall}} if hall else {},
                {"room": {"$eq": room}} if room else {},
            ]
        },
        n_results=top_k
    )
    return results
```

### New ALMA-aware Searcher
```python
# alma/core/search/searcher.py

def search_memories(
    query: str,
    agent_id: str,
    memory_types: Optional[List[str]] = None,
    wing: Optional[str] = None,
    include_shared: bool = False,
    top_k: int = 5,
) -> List[Memory]:
    """
    Search with ALMA governance enforcement
    
    1. Check agent's can_learn boundaries
    2. Include shared memories if agent inherits_from
    3. Apply semantic search
    4. Return top_k with highest similarity
    """
    
    # Step 1: Get agent's scopes
    agent = get_agent(agent_id)
    
    # Step 2: Build WHERE clause
    where_conditions = []
    
    # Agent must be able to learn this memory_type
    allowed_types = agent.can_learn if memory_types is None else memory_types
    where_conditions.append({"memory_type": {"$in": allowed_types}})
    
    # Exclude agent's cannot_learn categories
    where_conditions.append({"memory_type": {"$nin": agent.cannot_learn}})
    
    # Palace navigation (optional)
    if wing:
        where_conditions.append({"wing": {"$eq": wing}})
    
    # Handle shared memories
    if include_shared and agent.inherit_from:
        # OR: (agent_id == self) OR (agent_id in inherit_from)
        where_conditions.append({
            "$or": [
                {"agent_id": {"$eq": agent_id}},
                {"agent_id": {"$in": agent.inherit_from}},
            ]
        })
    else:
        where_conditions.append({"agent_id": {"$eq": agent_id}})
    
    # Step 3: Semantic search
    results = collection.query(
        query_embeddings=[embed(query)],
        where={"$and": where_conditions} if where_conditions else {},
        n_results=top_k,
        include=["embeddings", "documents", "metadatas", "distances"]
    )
    
    # Step 4: Convert to Memory objects
    memories = []
    for i in range(len(results["ids"])):
        mem = Memory(
            id=results["ids"][i],
            wing=results["metadatas"][i]["wing"],
            hall=results["metadatas"][i]["hall"],
            room=results["metadatas"][i]["room"],
            agent_id=results["metadatas"][i]["agent_id"],
            memory_type=results["metadatas"][i]["memory_type"],
            content=results["documents"][i],
            embedding=results["embeddings"][i],
            metadata=results["metadatas"][i],
        )
        memories.append(mem)
    
    return memories
```

### Your Task Day 3-4
1. Create `alma/core/search/searcher.py` with search logic above
2. Create `tests/test_retrieval.py`:
   - Test: agent's can_learn is respected
   - Test: cannot_learn boundaries enforced
   - Test: shared memories included when include_shared=True
   - Test: retrieval quality unchanged from MemPalace baseline
3. Run against MemPalace test dataset:
   - Target: >=95% accuracy (was 96.6%, 1-2% acceptable loss)

**Deliverable by EOD Day 4:**
```
alma/core/search/searcher.py
tests/test_retrieval.py
retrieval_benchmark_results.txt (showing accuracy >=95%)
```

---

## Task 5: Agent Scoping & Learning (Day 4-5)

### Create Agent Model
```python
# alma/governance/agent.py

class AgentManager:
    """Manage agent creation, scoping, and validation"""
    
    def create_agent(
        self,
        agent_id: str,
        name: str,
        domain: str,
        can_learn: List[str],
        cannot_learn: List[str],
        inherit_from: Optional[List[str]] = None,
        share_with: Optional[List[str]] = None,
    ) -> Agent:
        """Create new agent with scopes"""
        # Validate: can_learn and cannot_learn don't overlap
        overlap = set(can_learn) & set(cannot_learn)
        if overlap:
            raise ValueError(f"can_learn and cannot_learn overlap: {overlap}")
        
        agent = Agent(
            id=agent_id,
            name=name,
            domain=domain,
            can_learn=can_learn,
            cannot_learn=cannot_learn,
            inherit_from=inherit_from or [],
            share_with=share_with or [],
        )
        
        # Store in DB
        self.store_agent(agent)
        return agent
    
    def validate_learning(
        self,
        agent_id: str,
        memory_type: str,
        category: Optional[str] = None,
    ) -> bool:
        """Check if agent can learn this memory_type"""
        agent = self.get_agent(agent_id)
        
        # Check hard boundaries
        if memory_type in agent.cannot_learn:
            return False
        
        # Check soft boundaries (can_learn must include it or be empty)
        if agent.can_learn and memory_type not in agent.can_learn:
            return False
        
        return True
```

### Modify learn() function
```python
# alma/core/learning.py

def learn(
    agent_id: str,
    task: str,
    outcome: str,                # success|partial|failure
    strategy_used: str,
    memory_type: Optional[str] = None,  # auto-classify if None
    anti_pattern: bool = False,
) -> Memory:
    """
    Create a new memory with scoping enforcement
    
    Args:
        agent_id: which agent is learning
        task: what was the task
        outcome: result (success|partial|failure)
        strategy_used: how did you solve it
        memory_type: heuristic|outcome|preference|domain_knowledge|anti_pattern
        anti_pattern: if True, mark as what NOT to do
    """
    
    # Step 1: Infer memory_type if not provided
    if not memory_type:
        memory_type = infer_memory_type(outcome, anti_pattern)
        # heuristic if success, anti_pattern if failure, outcome always
    
    # Step 2: Validate agent can learn this
    agent_mgr = AgentManager()
    if not agent_mgr.validate_learning(agent_id, memory_type):
        raise PermissionError(
            f"Agent {agent_id} cannot learn {memory_type}. "
            f"can_learn={agent.can_learn}, cannot_learn={agent.cannot_learn}"
        )
    
    # Step 3: Create memory
    memory = Memory(
        id=str(uuid4()),
        wing=agent_id,
        hall=memory_type,
        room=extract_room_from_task(task),
        agent_id=agent_id,
        memory_type=memory_type,
        content=f"Task: {task}\nOutcome: {outcome}\nStrategy: {strategy_used}",
        metadata={
            "task": task,
            "outcome": outcome,
            "strategy_used": strategy_used,
            "anti_pattern": anti_pattern,
        }
    )
    
    # Step 4: Embed and store
    memory.embedding = embed(memory.content)
    store_memory(memory)
    
    return memory


def infer_memory_type(outcome: str, anti_pattern: bool) -> str:
    """Simple type inference"""
    if anti_pattern:
        return "anti_pattern"
    elif outcome == "success":
        return "heuristic"
    elif outcome == "failure":
        return "anti_pattern"
    else:
        return "outcome"
```

### Your Task Day 4-5
1. Create `alma/governance/agent.py` with AgentManager
2. Modify `alma/core/learning.py` with new learn() function
3. Create `tests/test_learning.py`:
   - Test: agent.can_learn enforced
   - Test: agent.cannot_learn enforced
   - Test: memory_type inference works
   - Test: anti_pattern marking works
4. Run MemPalace learning tests with new schema

**Deliverable by EOD Day 5:**
```
alma/governance/agent.py
alma/core/learning.py (modified)
tests/test_learning.py
```

---

## Task 6: Integration Test & Benchmark (Day 5-6)

### Create End-to-End Test
```python
# tests/test_e2e_alma_mempalace.py

def test_migrate_then_search_then_learn():
    """Full pipeline: migrate MemPalace → search with scoping → learn with validation"""
    
    # 1. Migrate MemPalace data
    stats = migrate_mempalace_to_alma("test_data/.chroma", ":memory:")
    assert stats["migrated"] > 0
    assert stats["failed"] == 0
    
    # 2. Create agents with different scopes
    agent_mgr = AgentManager()
    helena = agent_mgr.create_agent(
        agent_id="helena",
        name="Helena FE Tester",
        domain="coding",
        can_learn=["testing_strategies", "selector_patterns"],
        cannot_learn=["backend_logic", "database_queries"],
    )
    
    victor = agent_mgr.create_agent(
        agent_id="victor",
        name="Victor API Tester",
        domain="coding",
        can_learn=["api_patterns", "database_queries"],
        cannot_learn=["frontend_selectors"],
    )
    
    # 3. Search with scoping (Helena can't see Victor's API patterns)
    results = search_memories(
        query="how to mock external APIs",
        agent_id="helena",
    )
    # Helena shouldn't see backend-only patterns
    for result in results:
        assert result.memory_type not in helena.cannot_learn
    
    # 4. Helena learns a new heuristic
    mem = learn(
        agent_id="helena",
        task="Test form validation",
        outcome="success",
        strategy_used="Tested empty fields, invalid email, valid submission",
        memory_type="heuristic",
    )
    assert mem.agent_id == "helena"
    assert mem.memory_type == "heuristic"
    
    # 5. Helena tries to learn backend logic (should fail)
    with pytest.raises(PermissionError):
        learn(
            agent_id="helena",
            task="Set up database connection pool",
            outcome="success",
            strategy_used="Used pgbouncer",
            memory_type="domain_knowledge",
            # This violates cannot_learn
        )
    
    # 6. Victor can learn it
    mem = learn(
        agent_id="victor",
        task="Set up database connection pool",
        outcome="success",
        strategy_used="Used pgbouncer",
        memory_type="domain_knowledge",
    )
    assert mem.agent_id == "victor"
    
    # 7. Verify scope isolation
    helena_search = search_memories("pgbouncer", agent_id="helena", include_shared=False)
    assert len(helena_search) == 0  # Helena can't see Victor's memories
```

### Run Benchmarks
```bash
# In Week 1: just get baseline working
pytest tests/test_migration.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_learning.py -v
pytest tests/test_e2e_alma_mempalace.py -v

# Benchmark: retrieval accuracy
python benchmarks/longmemeval_bench.py --baseline mempalace --new alma
# Expected output:
# MemPalace baseline: 96.6%
# ALMA with scoping: >=95.0% (1-2% regression acceptable)
```

**Deliverable by EOD Day 6:**
```
tests/test_e2e_alma_mempalace.py (passing)
benchmarks/results_week1.txt (showing >=95% accuracy)
WEEK1_COMPLETE.md (status report)
```

---

## Week 1 Checklist

- [ ] `schema_notes.md` - MemPalace schema documented
- [ ] `alma/schema.py` - Memory + Agent dataclasses
- [ ] `alma/storage/schema.sql` - Database DDL
- [ ] `alma/migrate.py` - Migration script (dry-run tested)
- [ ] `alma/storage/sqlite.py` - SQLite backend
- [ ] `tests/test_migration.py` - Migration validation tests (passing)
- [ ] `alma/core/search/searcher.py` - ALMA-aware retrieval
- [ ] `tests/test_retrieval.py` - Retrieval tests (>=95% accuracy)
- [ ] `alma/governance/agent.py` - Agent scoping manager
- [ ] `alma/core/learning.py` - Updated learn() with validation
- [ ] `tests/test_learning.py` - Learning validation tests (passing)
- [ ] `tests/test_e2e_alma_mempalace.py` - End-to-end test (passing)
- [ ] Benchmarks running and showing >=95% accuracy
- [ ] `WEEK1_COMPLETE.md` - What got done, what didn't, what's next

---

## Week 1 Success Criteria

### Must Have
- All migrations complete without data loss
- Retrieval accuracy >= 95% (was 96.6%)
- Agent scoping enforced (can_learn / cannot_learn)
- Learning restricted by scopes
- All tests passing

### Nice to Have
- PostgreSQL backend partially implemented (separate branch)
- Benchmark numbers documented
- README updated

### If You Have Extra Time
- Start Sprint 2: Multi-agent sharing
- Draft anti-pattern learning interface

---

## Git Workflow for Week 1

```bash
cd alma-work

# Create feature branch
git checkout -b feature/mempalace-merge-week1

# Daily commits
git add alma/schema.py
git commit -m "feat: Memory + Agent dataclasses"

git add alma/migrate.py
git commit -m "feat: MemPalace → ALMA migration script"

git add alma/core/search/searcher.py
git commit -m "feat: ALMA-aware retrieval with scoping"

# At end of week
git push origin feature/mempalace-merge-week1

# Create PR for review (even if self-reviewing)
# Title: "Week 1: Storage unification + ALMA scoping foundation"
```

---

## What NOT to Do This Week

- Don't redesign the schema
- Don't add PostgreSQL (save for Sprint 2)
- Don't implement consolidation (save for Sprint 5)
- Don't touch MCP server (save for Sprint 6)
- Don't over-optimize queries yet (just make them correct)
- Don't add event webhooks (too much surface area)
- Don't implement multi-agent sharing fully (just skeleton)

---

## By Friday EOD

You should be able to:
1. Migrate MemPalace's test data to ALMA schema
2. Search with agent scoping enforced
3. Learn with boundaries checked
4. See >=95% retrieval accuracy
5. Have all tests passing

That's Week 1. Then you move to Sprint 2 (multi-agent sharing).

No more design. Just code. You know what to build.
