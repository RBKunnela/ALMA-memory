# ALMA - Agent Learning Memory Architecture

[![PyPI version](https://badge.fury.io/py/alma-memory.svg)](https://pypi.org/project/alma-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/RBKunnela/ALMA-memory/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Persistent memory for AI agents that learn and improve over time - without model weight updates.

---

## Why ALMA? Key Differentiators

ALMA isn't just another memory framework. Here's what sets it apart from alternatives like Mem0:

| Feature | ALMA | Mem0 | Why It Matters |
|---------|------|------|----------------|
| **Memory Scoping** | `can_learn` / `cannot_learn` per agent | Basic user/session isolation | Prevents agents from learning outside their domain |
| **Anti-Pattern Learning** | Explicit with `why_bad` + `better_alternative` | None | Agents learn what NOT to do |
| **Harness Pattern** | Decouples agent from domain memory | None | Reusable agent architecture |
| **MCP Integration** | Native stdio/HTTP server | None | Direct Claude Code integration |
| **Domain Memory Factory** | 6 pre-built schemas | None | Instant setup for any domain |
| **Multi-Factor Scoring** | Similarity + Recency + Success + Confidence | Primarily vector + recency | More nuanced retrieval |
| **Session Handoff** | Full context continuity | Minimal | No "starting fresh" problem |
| **Progress Tracking** | Built-in work item management | None | Agents know what to do next |

**Bottom line:** ALMA is purpose-built for AI agents that need to learn, remember, and improve - not just store and retrieve.

---

## What's New in v0.4.0

### Security Fixes
- **CRIT-001**: Fixed `eval()` vulnerability in Neo4j graph store - now uses safe `json.loads()`
- Input validation added to all MCP tools

### Bug Fixes
- **CRIT-002**: Fixed SQLite embeddings delete bug (memory_type naming mismatch)
- Fixed Azure Cosmos DB missing `update_heuristic_confidence()` and `update_knowledge_confidence()` methods
- Fixed PostgreSQL IVFFlat index creation on empty tables

### Performance Improvements
- Added HNSW indexes for PostgreSQL vector search (faster than IVFFlat)
- Lazy FAISS index rebuild for SQLite (only when needed)
- Added timestamp indexes for efficient temporal queries
- Connection pooling improvements for PostgreSQL

### API Improvements
- New `batch_save_*()` methods for bulk operations
- Comprehensive exception hierarchy (`ALMAError`, `StorageError`, `ValidationError`, etc.)
- Input validation with clear error messages

### Deprecation Fixes
- Replaced all `datetime.utcnow()` with timezone-aware `datetime.now(timezone.utc)`
- Python 3.13+ compatibility ensured

See [CHANGELOG.md](CHANGELOG.md) for the complete list.

---

## What is ALMA?

ALMA is a **memory framework** that makes AI agents appear to "learn" by:
1. **Storing** outcomes, strategies, and knowledge from past tasks
2. **Retrieving** relevant memories before each new task
3. **Injecting** that knowledge into prompts
4. **Learning** from new outcomes to improve future performance

**No fine-tuning. No model changes. Just smarter prompts.**

```
+---------------------------------------------------------------------+
|  BEFORE TASK: Retrieve relevant memories                            |
|  +-- "Last time you tested forms, incremental validation worked"    |
|  +-- "User prefers verbose output"                                  |
|  +-- "Don't use sleep() - causes flaky tests"                       |
+---------------------------------------------------------------------+
|  DURING TASK: Agent executes with injected knowledge                |
+---------------------------------------------------------------------+
|  AFTER TASK: Learn from outcome                                     |
|  +-- Success? -> New heuristic. Failure? -> Anti-pattern.           |
+---------------------------------------------------------------------+
```

---

## Installation

```bash
pip install alma-memory
```

**With optional backends:**
```bash
pip install alma-memory[local]     # SQLite + FAISS + local embeddings
pip install alma-memory[postgres]  # PostgreSQL + pgvector
pip install alma-memory[azure]     # Azure Cosmos DB + Azure OpenAI
pip install alma-memory[all]       # Everything
```

---

## Quick Start

### 1. Set Up Configuration

```bash
cp config.yaml.example .alma/config.yaml
```

Edit `.alma/config.yaml` to configure your project ID, storage backend, and agents.

### 2. Use ALMA in Your Code

```python
from alma import ALMA

# Initialize
alma = ALMA.from_config(".alma/config.yaml")

# Before task: Get relevant memories
memories = alma.retrieve(
    task="Test the login form validation",
    agent="helena",
    top_k=5
)

# Inject into your prompt
prompt = f"""
## Your Task
Test the login form validation

## Knowledge from Past Runs
{memories.to_prompt()}
"""

# After task: Learn from outcome
alma.learn(
    agent="helena",
    task="Test login form",
    outcome="success",
    strategy_used="Tested empty fields, invalid email, valid submission",
)
```

---

## Core Features

### Five Memory Types

| Type | What It Stores | Example |
|------|----------------|---------|
| **Heuristic** | Learned strategies | "For forms with >5 fields, test validation incrementally" |
| **Outcome** | Task results | "Login test succeeded using JWT token strategy" |
| **Preference** | User constraints | "User prefers verbose test output" |
| **Domain Knowledge** | Accumulated facts | "Login uses OAuth 2.0 with 24h token expiry" |
| **Anti-pattern** | What NOT to do | "Don't use sleep() for async waits - causes flaky tests" |

### Scoped Learning

Agents only learn within their defined domains. Helena (frontend tester) cannot learn backend logic:

```yaml
agents:
  helena:
    domain: coding
    can_learn:
      - testing_strategies
      - selector_patterns
    cannot_learn:
      - backend_logic
      - database_queries
```

### Storage Backends

| Backend | Use Case | Vector Search | Production Ready |
|---------|----------|---------------|------------------|
| **SQLite + FAISS** | Local development | Yes | Yes |
| **PostgreSQL + pgvector** | Production, HA | Yes (HNSW) | Yes |
| **Azure Cosmos DB** | Enterprise, Azure-native | Yes (DiskANN) | Yes |
| **File-based** | Testing | No | No |

---

## The Harness Pattern

ALMA implements a generalized harness pattern for any tool-using agent:

```
+---------------------------------------------------------------------+
|  1. SETTING        Fixed environment: tools, constraints            |
+---------------------------------------------------------------------+
|  2. CONTEXT        Ephemeral per-run inputs: task, user             |
+---------------------------------------------------------------------+
|  3. AGENT          The executor with scoped intelligence            |
+---------------------------------------------------------------------+
|  4. MEMORY SCHEMA  Domain-specific learning structure               |
+---------------------------------------------------------------------+
```

```python
from alma import create_harness, Context

# Create domain-specific harness
harness = create_harness("coding", "helena", alma)

# Run with automatic memory injection
result = harness.run(Context(
    task="Test the login form validation",
    project_id="my-app",
))
```

---

## MCP Server Integration

Expose ALMA to Claude Code or any MCP-compatible client:

```bash
python -m alma.mcp --config .alma/config.yaml
```

```json
// .mcp.json
{
  "mcpServers": {
    "alma-memory": {
      "command": "python",
      "args": ["-m", "alma.mcp", "--config", ".alma/config.yaml"]
    }
  }
}
```

**Available MCP Tools:**
| Tool | Description |
|------|-------------|
| `alma_retrieve` | Get memories for a task |
| `alma_learn` | Record task outcome |
| `alma_add_preference` | Add user preference |
| `alma_add_knowledge` | Add domain knowledge |
| `alma_forget` | Prune stale memories |
| `alma_stats` | Get memory statistics |
| `alma_health` | Health check |

---

## Advanced Features

### Domain Memory Factory

Create ALMA instances for any domain - not just coding:

```python
from alma.domains import DomainMemoryFactory, get_research_schema

# Pre-built schemas
factory = DomainMemoryFactory()
alma = factory.create_alma(get_research_schema(), "my-research-project")

# Or create custom domains
schema = factory.create_schema("sales", {
    "entity_types": [
        {"name": "lead", "attributes": ["stage", "value"]},
        {"name": "objection", "attributes": ["type", "response"]},
    ],
    "learning_categories": [
        "objection_handling",
        "closing_techniques",
        "qualification_patterns",
    ],
})
```

**Pre-built schemas:** `coding`, `research`, `sales`, `general`, `customer_support`, `content_creation`

### Progress Tracking

Track work items and get intelligent next-task suggestions:

```python
from alma.progress import ProgressTracker

tracker = ProgressTracker("my-project")

# Create work items
item = tracker.create_work_item(
    title="Fix authentication bug",
    description="Login fails on mobile devices",
    priority=80,
    agent="Victor",
)

# Update status
tracker.update_status(item.id, "in_progress")

# Get next task (by priority, quick wins, or unblock others)
next_task = tracker.get_next_item(strategy="priority")
```

### Session Handoff

Maintain context across sessions - no more "starting fresh":

```python
from alma.session import SessionManager

manager = SessionManager("my-project")

# Start session (loads previous context)
context = manager.start_session(agent="Helena", goal="Complete auth testing")

# Previous session info is available
if context.previous_handoff:
    print(f"Last action: {context.previous_handoff.last_action}")
    print(f"Blockers: {context.previous_handoff.blockers}")

# End session with handoff for next time
manager.create_handoff("Helena", context.session_id,
    last_action="completed_oauth_tests",
    last_outcome="success",
    next_steps=["Test refresh tokens", "Add error cases"],
)
```

### LLM-Powered Fact Extraction

Automatically extract and learn from conversations:

```python
from alma.extraction import AutoLearner

alma = ALMA.from_config(".alma/config.yaml")
auto_learner = AutoLearner(alma)

# After a conversation, automatically extract learnings
results = auto_learner.learn_from_conversation(
    messages=[
        {"role": "user", "content": "Test the login form"},
        {"role": "assistant", "content": "I used incremental validation which worked well..."},
    ],
    agent="helena",
)

print(f"Extracted {results['extracted_count']} facts")
```

### Graph Memory

Capture entity relationships for complex reasoning:

```python
from alma.graph import create_graph_store, EntityExtractor

# Connect to Neo4j
graph = create_graph_store(
    "neo4j",
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j",
    password="your-password",
)

# Extract entities and relationships from text
extractor = EntityExtractor()
entities, relationships = extractor.extract(
    "Alice from Acme Corp reviewed the PR that Bob submitted."
)

# Store in graph
for entity in entities:
    graph.add_entity(entity)
for rel in relationships:
    graph.add_relationship(rel)
```

### Confidence Engine

Assess strategies before trying them:

```python
from alma.confidence import ConfidenceEngine

engine = ConfidenceEngine(alma)

# Assess a strategy before trying it
signal = engine.assess_strategy(
    strategy="Use incremental validation",
    context="Testing a 5-field registration form",
    agent="Helena",
)

print(f"Confidence: {signal.confidence_score:.0%}")
print(f"Recommendation: {signal.recommendation}")
# -> Confidence: 78%
# -> Recommendation: yes
```

---

## Architecture

```
+-------------------------------------------------------------------------+
|                          ALMA v0.4.0                                    |
+-------------------------------------------------------------------------+
|  HARNESS LAYER                                                          |
|  +-----------+  +-----------+  +-----------+  +----------------+        |
|  | Setting   |  | Context   |  |  Agent    |  | MemorySchema   |        |
|  +-----------+  +-----------+  +-----------+  +----------------+        |
+-------------------------------------------------------------------------+
|  EXTENSION MODULES                                                      |
|  +-------------+  +---------------+  +------------------+               |
|  | Progress    |  | Session       |  | Domain Memory    |               |
|  | Tracking    |  | Handoff       |  | Factory          |               |
|  +-------------+  +---------------+  +------------------+               |
|  +-------------+  +---------------+  +------------------+               |
|  | Auto        |  | Confidence    |  | Graph            |               |
|  | Learner     |  | Engine        |  | Memory           |               |
|  +-------------+  +---------------+  +------------------+               |
+-------------------------------------------------------------------------+
|  CORE LAYER                                                             |
|  +-------------+  +-------------+  +-------------+  +------------+      |
|  | Retrieval   |  |  Learning   |  |  Caching    |  | Forgetting |      |
|  |  Engine     |  |  Protocol   |  |   Layer     |  | Mechanism  |      |
|  +-------------+  +-------------+  +-------------+  +------------+      |
+-------------------------------------------------------------------------+
|  STORAGE LAYER                                                          |
|  +---------------+  +------------------+  +---------------+             |
|  | SQLite+FAISS  |  | PostgreSQL+pgvec |  | Azure Cosmos  |             |
|  +---------------+  +------------------+  +---------------+             |
+-------------------------------------------------------------------------+
|  INTEGRATION LAYER                                                      |
|  +-------------------------------------------------------------------+  |
|  |                         MCP Server                                 |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

---

## Configuration

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite  # sqlite | postgres | azure | file
  embedding_provider: local  # local | azure | mock
  storage_dir: .alma
  db_name: alma.db
  embedding_dim: 384

  agents:
    helena:
      domain: coding
      can_learn:
        - testing_strategies
        - selector_patterns
      cannot_learn:
        - backend_logic
      min_occurrences_for_heuristic: 3

    victor:
      domain: coding
      can_learn:
        - api_patterns
        - database_queries
      cannot_learn:
        - frontend_selectors
```

### Embedding Providers

| Provider | Model | Dimensions | Cost | Best For |
|----------|-------|------------|------|----------|
| **local** | all-MiniLM-L6-v2 | 384 | Free | Development, offline |
| **azure** | text-embedding-3-small | 1536 | ~$0.02/1M | Production |
| **mock** | (hash-based) | 384 | Free | Testing only |

---

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| Core Abstractions | Memory types, scopes | Done |
| Local Storage | SQLite + FAISS | Done |
| Retrieval Engine | Semantic search, scoring | Done |
| Learning Protocols | Heuristic formation | Done |
| Agent Integration | Harness pattern | Done |
| Azure Cosmos DB | Enterprise storage | Done |
| PostgreSQL | Production storage | Done |
| Cache Layer | Performance optimization | Done |
| Forgetting Mechanism | Memory pruning | Done |
| MCP Server | Claude Code integration | Done |
| Progress Tracking | Work item management | Done |
| Session Handoff | Context continuity | Done |
| Domain Factory | Any-domain support | Done |
| Confidence Engine | Strategy assessment | Done |
| Technical Debt Sprint | Security & performance fixes | Done |

---

## Troubleshooting

### Common Issues

**ImportError: sentence-transformers is required**
```bash
pip install alma-memory[local]
```

**pgvector extension not found**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Embeddings dimension mismatch**
- Ensure `embedding_dim` in config matches your embedding provider
- Local: 384, Azure text-embedding-3-small: 1536

### Debug Logging

```python
import logging
logging.getLogger("alma").setLevel(logging.DEBUG)
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**What we need most:**
- Documentation improvements
- Test coverage for edge cases
- Storage backend integrations (MongoDB, Pinecone)
- TypeScript/JavaScript SDK
- LLM provider integrations (Ollama, Groq)

---

## Roadmap

Based on our [competitive analysis](docs/architecture/competitive-analysis-mem0.md):

**Next Quarter:**
- Multi-agent memory sharing
- Memory consolidation engine
- Event system / webhooks
- TypeScript SDK

**Future:**
- Expanded vector database support (Qdrant, Pinecone, Chroma)
- Memory compression / summarization
- Temporal reasoning
- Proactive memory suggestions

---

## License

MIT

---

## Star History

If ALMA helps your AI agents get smarter, consider giving us a star!

---

**Built for AI agents that get better with every task.**
