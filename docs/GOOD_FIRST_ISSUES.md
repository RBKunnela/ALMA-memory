# Good First Issues for ALMA

Copy these to create issues on GitHub. These are designed to attract contributors after your professor's video.

---

## Issue 1: Add PostgreSQL Storage Backend

**Title:** `[GOOD FIRST ISSUE] Add PostgreSQL storage backend`

**Labels:** `good-first-issue`, `help-wanted`, `storage`

**Body:**

### Overview
Add a PostgreSQL storage backend option alongside the existing SQLite, Azure Cosmos, and file-based backends.

### Why This Matters
Many teams already use PostgreSQL and would prefer to use their existing database infrastructure rather than adding SQLite or Azure.

### Difficulty Level
- [x] Medium (2-4 hours, requires understanding of storage abstraction)

### Steps to Complete

1. Create `alma/storage/postgres.py`
2. Implement `PostgresStorage` class extending `StorageBackend`
3. Implement all required methods (save_heuristic, get_heuristics, etc.)
4. Use `psycopg2` or `asyncpg` for database access
5. Add pgvector extension support for vector search
6. Add tests in `tests/test_postgres_storage.py`
7. Update documentation

### Files to Modify
- Create `alma/storage/postgres.py`
- Modify `alma/storage/__init__.py` to export new class
- Modify `alma/core.py` to support `storage: postgres` config
- Create `tests/test_postgres_storage.py`

### Resources
- [Existing SQLite implementation](alma/storage/sqlite_local.py) - Use as reference
- [StorageBackend base class](alma/storage/base.py)
- [pgvector documentation](https://github.com/pgvector/pgvector)

### Mentorship
Tag @RBKunnela if you have questions. Happy to pair program!

---

## Issue 2: Add Ollama Support for Local LLM Extraction

**Title:** `[GOOD FIRST ISSUE] Add Ollama support for fact extraction`

**Labels:** `good-first-issue`, `help-wanted`, `extraction`

**Body:**

### Overview
Add support for Ollama as an LLM provider for fact extraction, enabling fully offline operation.

### Why This Matters
Currently fact extraction requires OpenAI or Anthropic API keys. Adding Ollama support allows users to run ALMA completely offline with local models like Llama, Mistral, or Phi.

### Difficulty Level
- [x] Easy (1-2 hours)

### Steps to Complete

1. Add `OllamaExtractor` class in `alma/extraction/extractor.py`
2. Use the `ollama` Python package
3. Support configurable model selection (llama3, mistral, etc.)
4. Add to `create_extractor()` factory function
5. Add tests
6. Update documentation

### Files to Modify
- `alma/extraction/extractor.py` - Add OllamaExtractor class
- `alma/extraction/__init__.py` - Export new class
- `tests/test_extraction.py` - Add tests

### Example Implementation
```python
class OllamaExtractor(FactExtractor):
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        
    def extract(self, messages, agent_context=None, existing_facts=None):
        import ollama
        # Implementation here
```

### Resources
- [Ollama Python library](https://github.com/ollama/ollama-python)
- [Existing LLMFactExtractor](alma/extraction/extractor.py)

---

## Issue 3: Improve Test Coverage for Retrieval Engine

**Title:** `[GOOD FIRST ISSUE] Add tests for retrieval edge cases`

**Labels:** `good-first-issue`, `help-wanted`, `testing`

**Body:**

### Overview
Add comprehensive tests for the retrieval engine, focusing on edge cases and error handling.

### Why This Matters
Better test coverage makes ALMA more reliable and easier to maintain. Currently retrieval has gaps in edge case testing.

### Difficulty Level
- [x] Easy (1-2 hours)

### Test Cases Needed

1. Retrieval with empty database
2. Retrieval with no matching results
3. Retrieval with special characters in query
4. Retrieval with very long query strings
5. Cache hit/miss scenarios
6. Score threshold filtering
7. Agent scope filtering

### Files to Modify
- `tests/test_retrieval.py` - Add new test cases

### Example Test
```python
def test_retrieval_empty_database():
    """Retrieval should return empty results gracefully."""
    storage = InMemoryStorage()  # Empty
    engine = RetrievalEngine(storage)
    
    result = engine.retrieve(
        query="test query",
        agent="helena",
        project_id="test",
    )
    
    assert result.heuristics == []
    assert result.outcomes == []
    assert result.total_items == 0
```

### Mentorship
Great issue for learning the codebase! Tag @RBKunnela for guidance.

---

## Issue 4: Add Examples Directory with Use Cases

**Title:** `[GOOD FIRST ISSUE] Create examples directory with practical use cases`

**Labels:** `good-first-issue`, `help-wanted`, `documentation`

**Body:**

### Overview
Create an `examples/` directory with practical, runnable examples showing ALMA in action.

### Why This Matters
Examples help new users understand how to use ALMA in real scenarios. Currently we only have the demo.py file.

### Difficulty Level
- [x] Easy (2-3 hours, mostly writing code examples)

### Examples Needed

1. `examples/basic_usage.py` - Simple retrieve/learn cycle
2. `examples/multi_agent.py` - Multiple agents with scoped learning
3. `examples/auto_learning.py` - Using AutoLearner for automatic extraction
4. `examples/graph_memory.py` - Using graph memory with Neo4j
5. `examples/mcp_integration.py` - Using ALMA with Claude via MCP
6. `examples/README.md` - Overview of all examples

### Requirements for Each Example
- Self-contained (can run independently)
- Well-commented
- Includes expected output
- Uses mock/in-memory storage (no external dependencies)

### Files to Create
- `examples/README.md`
- `examples/basic_usage.py`
- `examples/multi_agent.py`
- `examples/auto_learning.py`
- `examples/graph_memory.py`

---

## Issue 5: Add TypeScript/JavaScript SDK

**Title:** `[FEATURE] TypeScript SDK for ALMA`

**Labels:** `help-wanted`, `enhancement`, `typescript`

**Body:**

### Overview
Create a TypeScript SDK for ALMA, enabling JavaScript/TypeScript developers to use ALMA in Node.js and browser environments.

### Why This Matters
Many AI applications are built in JavaScript/TypeScript. A native SDK would significantly expand ALMA's reach.

### Difficulty Level
- [x] Hard (10+ hours, requires TypeScript expertise)

### Scope

**Phase 1 - API Client**
- REST API client for ALMA Cloud (when available)
- TypeScript types for all ALMA concepts
- Basic retrieve/learn operations

**Phase 2 - Local Implementation**
- Port core ALMA logic to TypeScript
- Support for browser-based storage (IndexedDB)
- Support for Node.js storage (SQLite via better-sqlite3)

### Suggested Structure
```
alma-ts/
├── src/
│   ├── index.ts
│   ├── types.ts
│   ├── client.ts
│   ├── storage/
│   │   ├── memory.ts
│   │   └── indexeddb.ts
│   └── retrieval/
│       └── engine.ts
├── package.json
├── tsconfig.json
└── README.md
```

### Resources
- [Python ALMA implementation](alma/) - Reference
- [Mem0 TypeScript SDK](https://github.com/mem0ai/mem0/tree/main/mem0-ts) - Inspiration

### Mentorship
This is a larger project. @RBKunnela will provide guidance on architecture decisions.

---

## Issue 6: Add Documentation Site

**Title:** `[GOOD FIRST ISSUE] Set up documentation site with MkDocs`

**Labels:** `good-first-issue`, `help-wanted`, `documentation`

**Body:**

### Overview
Set up a documentation website using MkDocs and Material theme, deployed to GitHub Pages.

### Why This Matters
A proper documentation site is more accessible than README files and provides better navigation, search, and organization.

### Difficulty Level
- [x] Easy (2-3 hours)

### Steps to Complete

1. Set up MkDocs with Material theme
2. Organize existing documentation into sections
3. Add navigation structure
4. Configure GitHub Actions for auto-deployment
5. Add API reference generation (mkdocstrings)

### Suggested Structure
```
docs/
├── index.md (Home)
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── concepts/
│   ├── memory-types.md
│   ├── scoped-learning.md
│   └── harness-pattern.md
├── features/
│   ├── extraction.md
│   ├── graph-memory.md
│   └── mcp-server.md
├── api/
│   └── reference.md
└── contributing.md
```

### Files to Create/Modify
- `mkdocs.yml` - Configuration
- `docs/` - Documentation files
- `.github/workflows/docs.yml` - Auto-deploy workflow

### Resources
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)

---

## Issue 7: Add Benchmarks Against Mem0

**Title:** `[FEATURE] Add benchmarks comparing ALMA vs Mem0`

**Labels:** `help-wanted`, `enhancement`, `benchmarks`

**Body:**

### Overview
Create a benchmarking suite that compares ALMA's performance against Mem0 on standard tasks.

### Why This Matters
Objective benchmarks help users make informed decisions and demonstrate ALMA's strengths.

### Metrics to Measure

1. **Retrieval accuracy** - Relevance of retrieved memories
2. **Retrieval latency** - Time to retrieve memories
3. **Learning speed** - Time to learn from outcomes
4. **Memory efficiency** - Storage size per memory
5. **Token usage** - LLM tokens used for extraction

### Benchmark Scenarios

1. Single-hop retrieval (direct fact lookup)
2. Multi-hop retrieval (reasoning across memories)
3. Temporal queries (time-based retrieval)
4. Scoped vs unscoped retrieval
5. Anti-pattern matching

### Files to Create
- `benchmarks/README.md`
- `benchmarks/retrieval_benchmark.py`
- `benchmarks/learning_benchmark.py`
- `benchmarks/results/` - Store results

### Resources
- [LOCOMO benchmark](https://arxiv.org/abs/2504.19413) - Used by Mem0

---

## How to Use This File

1. Go to https://github.com/RBKunnela/ALMA-memory/issues/new
2. Copy the content for each issue
3. Add appropriate labels
4. Publish

Create issues #1, #2, #3, #4, and #6 first (easiest for newcomers).
Save #5 and #7 for after you have some momentum.
