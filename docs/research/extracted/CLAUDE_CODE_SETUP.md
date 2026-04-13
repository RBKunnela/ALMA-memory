# Claude Code Setup - Ready to Go

## Your Repositories Are Ready

Located at:
- `/home/claude/mempalace-repo/` - MemPalace (reference)
- `github.com/RBKunnela/ALMA-memory` - Your ALMA repo (where you work)

## Prepare ALMA Repo for Merge

```bash
cd /path/to/ALMA-memory

# Create feature branch for merge
git checkout -b feature/mempalace-merge-sprint1

# Create directories for new code
mkdir -p alma/core/{search,mining,learning}
mkdir -p alma/storage
mkdir -p alma/governance
mkdir -p alma/integration
mkdir -p tests/fixtures
mkdir -p benchmarks/{longmemeval,convomem,governance}
```

## Download Your Four Documents

Files are in `/mnt/user-data/outputs/`:
1. `00_START_HERE.md` - Read first
2. `mempalace_vs_alma_analysis.md` - Why merge
3. `alma_mempalace_merge_strategy.md` - What to build
4. `week1_execution_guide.md` - What to code
5. `technical_patterns_reference.md` - Code patterns

Download to: `/path/to/ALMA-memory/docs/merge-planning/`

## Week 1 Checklist

### Day 1: Schema
- [ ] Read `week1_execution_guide.md` Task 1
- [ ] Create `alma/schema.py` (Memory + Agent dataclasses)
- [ ] Document MemPalace drawer format in `schema_notes.md`

### Day 2: SQL
- [ ] Create `alma/storage/schema.sql` (complete DDL)
- [ ] Finalize unified schema design
- [ ] Review with any teammates

### Day 3: Migration
- [ ] Create `alma/migrate.py` (MemPalace → ALMA converter)
- [ ] Create `alma/storage/sqlite.py` (SQLite backend)
- [ ] Dry-run migration on test data

### Day 4: Retrieval
- [ ] Create `alma/core/search/searcher.py` (ALMA-aware search)
- [ ] Add agent scoping to WHERE clause
- [ ] Copy `alma/core/search/query_sanitizer.py` from MemPalace

### Day 5: Learning
- [ ] Create `alma/governance/agent.py` (AgentManager)
- [ ] Modify `alma/core/learning.py` (scoped learn())
- [ ] Add validation for can_learn/cannot_learn

### Day 6: Testing
- [ ] Create `tests/test_migration.py`
- [ ] Create `tests/test_retrieval.py`
- [ ] Create `tests/test_learning.py`
- [ ] Create `tests/test_e2e_alma_mempalace.py`
- [ ] Run all tests - target: all passing
- [ ] Benchmark: LongMemEval >= 95%

## File Structure After Week 1

```
ALMA-memory/
├── alma/
│   ├── __init__.py
│   ├── schema.py                     # NEW: Memory + Agent
│   ├── config.py                     # MODIFIED: Add ALMA config
│   ├── core/
│   │   ├── __init__.py
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── searcher.py           # NEW: ALMA-aware
│   │   │   ├── query_sanitizer.py    # COPY from MemPalace
│   │   │   └── embedder.py           # NEW: Embedding wrapper
│   │   ├── mining/
│   │   │   ├── __init__.py
│   │   │   ├── entity_detector.py    # COPY from MemPalace
│   │   │   ├── normalize.py          # COPY from MemPalace
│   │   │   └── convo_miner.py        # ADAPT from MemPalace
│   │   └── learning.py               # MODIFIED: Add scoping
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── schema.sql                # NEW: DDL
│   │   ├── sqlite.py                 # NEW: SQLite backend
│   │   └── chroma.py                 # ADAPT from MemPalace
│   └── governance/
│       ├── __init__.py
│       └── agent.py                  # NEW: AgentManager
│
├── tests/
│   ├── conftest.py                   # MODIFY: Add ALMA fixtures
│   ├── test_migration.py             # NEW
│   ├── test_retrieval.py             # NEW
│   ├── test_learning.py              # NEW
│   ├── test_e2e_alma_mempalace.py    # NEW
│   └── fixtures/
│       └── sample_data/
│
├── benchmarks/
│   ├── longmemeval_with_scoping.py   # NEW
│   ├── results_week1.txt             # NEW (after running)
│   └── (copy MemPalace benchmarks)
│
├── docs/
│   ├── merge-planning/               # YOUR FOUR DOCUMENTS
│   │   ├── 00_START_HERE.md
│   │   ├── mempalace_vs_alma_analysis.md
│   │   ├── alma_mempalace_merge_strategy.md
│   │   ├── week1_execution_guide.md
│   │   └── technical_patterns_reference.md
│   ├── architecture.md               # SPRINT 2+
│   └── examples/
│
├── pyproject.toml                    # MODIFIED: Add dependencies
├── README.md                         # UNCHANGED (update Week 6)
└── WEEK1_COMPLETE.md                 # NEW (status report)
```

## Claude Code Integration

### MCP Setup (Sprint 6, not Week 1)
```json
{
  "mcpServers": {
    "alma-memory": {
      "command": "python",
      "args": ["-m", "alma.mcp", "--config", ".alma/config.yaml"]
    }
  }
}
```

### Testing in Claude Code
```bash
# Week 1: Just get things working
python -m pytest tests/ -v --tb=short

# Week 6: Full integration test
python -m alma.mcp &
# Then interact with Claude
```

## Dependencies to Add

```toml
# pyproject.toml

[project]
dependencies = [
    "chromadb>=0.4.0",
    "sentence-transformers>=2.2.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "numpy>=1.23.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
postgres = ["psycopg>=3.1.0", "pgvector>=0.1.0"]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black>=23.0.0", "ruff>=0.1.0"]
```

## Git Workflow

```bash
# Start Week 1
git checkout -b feature/mempalace-merge-week1

# Daily commits
git add alma/schema.py
git commit -m "feat: Memory + Agent dataclasses"

git add alma/migrate.py
git commit -m "feat: MemPalace to ALMA migration"

# End of week
git push origin feature/mempalace-merge-week1
git pull-request

# Self-review, then merge to main
```

## Environment Variables (Optional)

```bash
# .env (optional, for later sprints)
ALMA_PROJECT_ID=default
ALMA_STORAGE_DIR=.alma
ALMA_EMBEDDING_PROVIDER=local
PINECONE_API_KEY=...  # Sprint 2+
POSTGRES_URL=...      # Sprint 2+
```

## Running Tests

```bash
# Week 1: Just unit tests
pytest tests/test_migration.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_learning.py -v
pytest tests/test_e2e_alma_mempalace.py -v

# Week 1: Check one benchmark
python -m pytest benchmarks/longmemeval_with_scoping.py -v

# Week 6: Full test suite
pytest tests/ -v --cov=alma --cov-report=html
```

## Your First 30 Minutes

1. Read `00_START_HERE.md` (10 min)
2. Read first half of `alma_mempalace_merge_strategy.md` (10 min)
3. Skim `week1_execution_guide.md` to understand scope (10 min)

Then start coding Task 1 from `week1_execution_guide.md`.

## Success Looks Like (Friday EOD)

You can run:
```bash
python -c "
from alma.schema import Memory, Agent
from alma.core.search import search_memories
from alma.core.learning import learn
from alma.migrate import migrate_mempalace_to_alma

# Migrate MemPalace data
stats = migrate_mempalace_to_alma('test.db', ':memory:')
print(f'Migrated {stats[\"migrated\"]} memories')

# Create agent
agent = Agent(
    id='helena',
    name='Helena',
    domain='coding',
    can_learn=['testing_strategies'],
    cannot_learn=['backend_logic'],
)

# Learn something
mem = learn(
    agent_id='helena',
    task='Test form',
    outcome='success',
    strategy_used='Incremental validation',
)

# Search with scoping
results = search_memories('validation patterns', agent_id='helena')
print(f'Found {len(results)} relevant memories')
"
```

And it all works.

That's Week 1. Then you move to Sprint 2.

---

## Final Checklist

- [ ] MemPalace repo cloned
- [ ] ALMA repo ready
- [ ] Four documents downloaded
- [ ] Directories created
- [ ] Read START_HERE.md
- [ ] First coding task understood
- [ ] Dependencies added to pyproject.toml
- [ ] Ready to start Monday morning

You are set. Now execute.
