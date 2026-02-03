# ALMA - Agent Learning Memory Architecture

[![PyPI version](https://badge.fury.io/py/alma-memory.svg)](https://pypi.org/project/alma-memory/)
[![npm version](https://img.shields.io/badge/npm-v0.6.0-cb3837?logo=npm)](https://github.com/RBKunnela/ALMA-memory/pkgs/npm/alma-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/RBKunnela/ALMA-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/RBKunnela/ALMA-memory/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-alma--memory.pages.dev-blue)](https://alma-memory.pages.dev)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/aiagentsprp)

<div align="center">

### 🧠 AI Agents That Actually Learn

**Persistent memory for AI agents that improves over time - no fine-tuning required.**

[**🌐 Visit alma-memory.pages.dev →**](https://alma-memory.pages.dev)

</div>

---

**[📖 Documentation](https://alma-memory.pages.dev)** · **[🔧 Technical Reference](docs/TECHNICAL.md)** · **[📦 PyPI](https://pypi.org/project/alma-memory/)** · **[📦 npm](https://github.com/RBKunnela/ALMA-memory/pkgs/npm/alma-memory)** · **[☕ Support](https://buymeacoffee.com/aiagentsprp)**

---

## Looking for a Mem0 Alternative? LangChain Memory Replacement?

**ALMA is the answer.** If you've tried Mem0 or LangChain Memory and found them lacking for production AI agents, ALMA was built specifically to solve those gaps:

| If you need... | Mem0 | LangChain | **ALMA** |
|----------------|------|-----------|----------|
| Scoped learning (agents only learn their domain) | ❌ | ❌ | ✅ |
| Anti-pattern tracking (what NOT to do) | ❌ | ❌ | ✅ |
| Multi-agent knowledge sharing | ❌ | ❌ | ✅ |
| TypeScript/JavaScript SDK | ❌ | ✅ | ✅ |
| MCP integration (Claude Code) | ❌ | ❌ | ✅ |
| 6 vector database backends | Limited | Limited | ✅ |
| Graph memory for relationships | Limited | Limited | ✅ |
| Workflow checkpoints & state merging | ❌ | ❌ | ✅ |

**See detailed comparisons:** [ALMA vs Mem0](https://alma-memory.pages.dev/comparison/mem0-vs-alma.html) · [ALMA vs LangChain Memory](https://alma-memory.pages.dev/comparison/langchain-memory-vs-alma.html)

---

## The Problem I Solved

I was building AI agents for automated testing. Helena for frontend QA, Victor for backend verification. They worked great... until they didn't.

**The same mistakes kept happening:**

- Helena would use `sleep(5000)` for waits, causing flaky tests
- Victor would forget that the API uses JWT with 24-hour expiry
- Both agents would repeat failed strategies session after session

Every conversation started fresh. No memory. No learning. Just an expensive LLM making the same mistakes I'd already corrected.

I tried **Mem0**. It stores memories, but no way to scope what an agent can learn, no anti-pattern tracking, no multi-agent sharing. I looked at **LangChain Memory**. It's for conversation context, not long-term learning.

**Nothing fit. So I built ALMA.**

The core insight: AI agents don't need to modify their weights to "learn." They need **smart prompts** built from **relevant past experiences.**

---

## Why ALMA? Key Differentiators

ALMA isn't just another memory framework. Here's what sets it apart from alternatives like Mem0:

| Feature | ALMA | Mem0 | Why It Matters |
|---------|------|------|----------------|
| **Memory Scoping** | `can_learn` / `cannot_learn` per agent | Basic user/session isolation | Prevents agents from learning outside their domain |
| **Anti-Pattern Learning** | Explicit with `why_bad` + `better_alternative` | None | Agents learn what NOT to do |
| **Multi-Agent Sharing** | `inherit_from` + `share_with` scopes | None | Agents can share knowledge hierarchically |
| **Memory Consolidation** | LLM-powered deduplication | Basic | Intelligent merge of similar memories |
| **Event System** | Webhooks + in-process callbacks | None | React to memory changes in real-time |
| **TypeScript SDK** | Full-featured client library | None | First-class JavaScript/TypeScript support |
| **Vector DB Support** | 6 backends (PostgreSQL, Qdrant, Pinecone, Chroma, SQLite, Azure) | Limited | Deploy anywhere |
| **Graph Memory** | Pluggable backends (Neo4j, Memgraph, Kuzu, In-memory) | Limited | Entity relationship tracking |
| **Harness Pattern** | Decouples agent from domain memory | None | Reusable agent architecture |
| **MCP Integration** | Native stdio/HTTP server | None | Direct Claude Code integration |
| **Domain Memory Factory** | 6 pre-built schemas | None | Instant setup for any domain |
| **Multi-Factor Scoring** | Similarity + Recency + Success + Confidence | Primarily vector + recency | More nuanced retrieval |

**Bottom line:** ALMA is purpose-built for AI agents that need to learn, remember, and improve - not just store and retrieve.

### Quick Comparison: ALMA vs Mem0 vs Graphiti

| Feature | ALMA | Mem0 | Graphiti |
|---------|------|------|----------|
| Memory Scoping | ✅ `can_learn`/`cannot_learn` | ❌ | ❌ |
| Anti-Pattern Learning | ✅ `why_bad` + `better_alternative` | ❌ | ❌ |
| Multi-Agent Inheritance | ✅ `inherit_from`/`share_with` | ❌ | ❌ |
| Multi-Factor Scoring | ✅ 4 factors (similarity + recency + success + confidence) | ❌ similarity only | ❌ similarity only |
| MCP Integration | ✅ 16 tools | ❌ | ❌ |
| Workflow Checkpoints | ✅ | ❌ | ❌ |
| TypeScript SDK | ✅ | ❌ | ❌ |
| Graph + Vector Hybrid | ✅ | Limited | ✅ |

**The key insight:** Most solutions treat memory as "store embeddings, retrieve similar." ALMA treats it as "teach agents to improve within safe boundaries."

---

## What's New in v0.6.0

### Workflow Context Layer

The major theme of v0.6.0 is **multi-agent workflow support** - enabling agents to coordinate across long-running tasks with checkpoints, state merging, and artifact tracking.

- **Checkpoint & Resume** (`alma/workflow/`)
  - Save workflow state at any point: `alma.checkpoint(workflow_id, state, metadata)`
  - Resume from checkpoints after failures or handoffs: `alma.resume(workflow_id)`
  - Automatic cleanup of old checkpoints: `alma.cleanup_checkpoints(older_than_days=7)`

- **State Reducers for Multi-Agent Workflows**
  - Merge states from parallel agents: `alma.merge_states(workflow_id, states, reducer)`
  - Built-in reducers: `latest_wins`, `merge_lists`, `priority_agent`
  - Custom reducer functions for complex merge logic

- **Artifact Linking**
  - Link outputs to workflows: `alma.link_artifact(workflow_id, artifact_type, ref)`
  - Artifact types: `code`, `test`, `document`, `config`, `deployment`
  - Retrieve all artifacts: `alma.get_artifacts(workflow_id)`

- **Scoped Retrieval**
  - Filter memories by workflow context: `alma.retrieve_scoped(query, scope)`
  - Scopes: `workflow_only`, `agent_only`, `project_wide`

- **Session Persistence**
  - Session handoffs now persist to storage backend
  - Lazy loading for performance
  - Survives process restarts

- **MCP Workflow Tools** (8 new tools)
  - `alma_consolidate`, `alma_checkpoint`, `alma_resume`
  - `alma_merge_states`, `alma_workflow_learn`
  - `alma_link_artifact`, `alma_get_artifacts`
  - `alma_cleanup_checkpoints`, `alma_retrieve_scoped`

- **TypeScript SDK v0.6.0** (`packages/alma-memory-js/`)
  - Full workflow API parity with Python SDK
  - 9 new methods: `consolidate()`, `checkpoint()`, `resume()`, `mergeStates()`, `workflowLearn()`, `linkArtifact()`, `getArtifacts()`, `cleanupCheckpoints()`, `retrieveScoped()`
  - 25+ new TypeScript types for workflow context
  - Published to GitHub Packages: `@rbkunnela/alma-memory`

### Previous Releases

<details>
<summary>v0.5.0 - Vector Database Backends</summary>

- **Qdrant Backend** - Full vector similarity search with metadata filtering
- **Pinecone Backend** - Serverless spec support, namespace organization
- **Chroma Backend** - Persistent, client-server, and ephemeral modes
- **Graph Database Abstraction** - Neo4j, Memgraph, Kuzu, In-memory backends
- **Testing Module** - MockStorage, MockEmbedder, factory functions
- **Memory Consolidation Engine** - LLM-powered deduplication
- **Event System** - Webhooks + in-process callbacks
- **TypeScript SDK** - Initial release with core API
- **Multi-Agent Memory Sharing** - inherit_from, share_with

</details>

See [CHANGELOG.md](CHANGELOG.md) for the complete history.

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
# Local development
pip install alma-memory[local]     # SQLite + FAISS + local embeddings

# Production databases
pip install alma-memory[postgres]  # PostgreSQL + pgvector
pip install alma-memory[qdrant]    # Qdrant vector database
pip install alma-memory[pinecone]  # Pinecone vector database
pip install alma-memory[chroma]    # ChromaDB

# Enterprise
pip install alma-memory[azure]     # Azure Cosmos DB + Azure OpenAI

# Everything
pip install alma-memory[all]
```

**TypeScript/JavaScript (via GitHub Packages):**
```bash
# Configure npm for the scope (one-time)
echo "@rbkunnela:registry=https://npm.pkg.github.com" >> ~/.npmrc

# Install
npm install @rbkunnela/alma-memory
# or
yarn add @rbkunnela/alma-memory
```

---

## Quick Start

### Python

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

### TypeScript/JavaScript

```typescript
import { ALMA } from '@rbkunnela/alma-memory';

// Create client
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project'
});

// Retrieve memories
const memories = await alma.retrieve({
  query: 'authentication flow',
  agent: 'dev-agent',
  topK: 5
});

// Learn from outcomes
await alma.learn({
  agent: 'dev-agent',
  task: 'Implement OAuth',
  outcome: 'success',
  strategyUsed: 'Used passport.js middleware'
});

// Add preferences and knowledge
await alma.addPreference({
  userId: 'user-123',
  category: 'code_style',
  preference: 'Use TypeScript strict mode'
});

await alma.addKnowledge({
  agent: 'dev-agent',
  domain: 'authentication',
  fact: 'API uses JWT with 24h expiry'
});
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

### Multi-Agent Memory Sharing

Enable agents to share knowledge hierarchically:

```yaml
agents:
  senior_dev:
    can_learn: [architecture, best_practices]
    share_with: [junior_dev, qa_agent]  # Others can read my memories

  junior_dev:
    can_learn: [coding_patterns]
    inherit_from: [senior_dev]  # I can read senior's memories
```

```python
# Junior dev retrieves memories (includes senior's shared memories)
memories = alma.retrieve(
    task="Implement user authentication",
    agent="junior_dev",
    include_shared=True  # Include inherited memories
)

# Shared memories are marked with origin
for mem in memories.heuristics:
    if mem.metadata.get('shared_from'):
        print(f"Learned from {mem.metadata['shared_from']}: {mem.strategy}")
```

### Storage Backends

| Backend | Use Case | Vector Search | Production Ready |
|---------|----------|---------------|------------------|
| **SQLite + FAISS** | Local development | Yes | Yes |
| **PostgreSQL + pgvector** | Production, HA | Yes (HNSW) | Yes |
| **Qdrant** | Managed vector DB | Yes (HNSW) | Yes |
| **Pinecone** | Serverless vector DB | Yes | Yes |
| **Chroma** | Lightweight local | Yes | Yes |
| **Azure Cosmos DB** | Enterprise, Azure-native | Yes (DiskANN) | Yes |
| **File-based** | Testing | No | No |

---

## Memory Consolidation

Automatically deduplicate and merge similar memories using LLM intelligence:

```python
from alma.consolidation import ConsolidationEngine

engine = ConsolidationEngine(
    storage=alma.storage,
    llm_client=my_llm_client  # Optional: for intelligent merging
)

# Consolidate heuristics for an agent
result = await engine.consolidate(
    agent="helena",
    project_id="my-project",
    memory_type="heuristics",
    similarity_threshold=0.85,  # Group memories above this similarity
    use_llm=True,               # Use LLM for intelligent merging
    dry_run=False               # Set True to preview without changes
)

print(f"Merged {result.merged_count} memories from {result.groups_found} groups")
```

**Features:**
- Cosine similarity-based grouping
- LLM-powered intelligent merging (preserves important nuances)
- Provenance tracking (`merged_from` metadata)
- Support for heuristics, domain_knowledge, and anti_patterns

---

## Event System

React to memory changes with webhooks or in-process callbacks:

### In-Process Callbacks

```python
from alma.events import get_emitter, MemoryEventType

def on_memory_created(event):
    print(f"Memory created: {event.memory_id} by {event.agent}")
    # Trigger downstream processes, update caches, etc.

emitter = get_emitter()
emitter.subscribe(MemoryEventType.CREATED, on_memory_created)
emitter.subscribe(MemoryEventType.CONSOLIDATED, on_consolidation)
```

### Webhooks

```python
from alma.events import WebhookConfig, WebhookManager, get_emitter

manager = WebhookManager()
manager.add_webhook(WebhookConfig(
    url="https://your-app.com/alma-webhook",
    events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
    secret="your-webhook-secret",  # For HMAC signature verification
    retry_count=3,
    retry_delay=5.0
))
manager.start(get_emitter())
```

**Event Types:**
- `CREATED` - New memory stored
- `UPDATED` - Memory modified
- `DELETED` - Memory removed
- `ACCESSED` - Memory retrieved
- `CONSOLIDATED` - Memories merged

---

## Graph Memory

Capture entity relationships for complex reasoning:

```python
from alma.graph import create_graph_backend, BackendGraphStore, EntityExtractor

# Create graph backend - multiple options available:

# Neo4j (production, hosted)
backend = create_graph_backend(
    "neo4j",
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j",
    password="your-password"
)

# Memgraph (high-performance, in-memory)
# backend = create_graph_backend("memgraph", uri="bolt://localhost:7687")

# Kuzu (embedded, no server required)
# backend = create_graph_backend("kuzu", database_path="./my_graph_db")

# In-memory (testing)
# backend = create_graph_backend("memory")

# Create store with backend
graph = BackendGraphStore(backend)

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

# Query relationships
alice_relations = graph.get_relationships("alice", relationship_type="WORKS_FOR")
```

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

**Available MCP Tools (16 total):**

| Core Tools | Description |
|------------|-------------|
| `alma_retrieve` | Get memories for a task |
| `alma_learn` | Record task outcome |
| `alma_add_preference` | Add user preference |
| `alma_add_knowledge` | Add domain knowledge |
| `alma_forget` | Prune stale memories |
| `alma_stats` | Get memory statistics |
| `alma_health` | Health check |

| Workflow Tools (v0.6.0) | Description |
|-------------------------|-------------|
| `alma_consolidate` | Merge similar memories |
| `alma_checkpoint` | Save workflow state |
| `alma_resume` | Resume from checkpoint |
| `alma_merge_states` | Merge parallel agent states |
| `alma_workflow_learn` | Learn with workflow context |
| `alma_link_artifact` | Link output to workflow |
| `alma_get_artifacts` | Get workflow artifacts |
| `alma_cleanup_checkpoints` | Clean old checkpoints |
| `alma_retrieve_scoped` | Scoped memory retrieval |

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
|                          ALMA v0.6.0                                    |
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
|  | Auto        |  | Confidence    |  | Memory           |               |
|  | Learner     |  | Engine        |  | Consolidation    |               |
|  +-------------+  +---------------+  +------------------+               |
|  +-------------+  +---------------+                                     |
|  | Event       |  | TypeScript    |                                     |
|  | System      |  | SDK           |                                     |
|  +-------------+  +---------------+                                     |
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
|  +---------------+  +------------------+  +---------------+             |
|  |    Qdrant     |  |    Pinecone      |  |    Chroma     |             |
|  +---------------+  +------------------+  +---------------+             |
+-------------------------------------------------------------------------+
|  GRAPH LAYER                                                            |
|  +---------------+  +------------------+  +---------------+             |
|  |    Neo4j      |  |    Memgraph      |  |     Kuzu      |             |
|  +---------------+  +------------------+  +---------------+             |
|  +---------------+                                                      |
|  |   In-Memory   |                                                      |
|  +---------------+                                                      |
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
  storage: sqlite  # sqlite | postgres | qdrant | pinecone | chroma | azure | file
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
      share_with: [qa_lead]  # Share memories with QA lead

    victor:
      domain: coding
      can_learn:
        - api_patterns
        - database_queries
      cannot_learn:
        - frontend_selectors
      inherit_from: [senior_architect]  # Learn from senior architect
```

### Storage Backend Configuration

**PostgreSQL + pgvector:**
```yaml
storage: postgres
postgres:
  host: localhost
  port: 5432
  database: alma
  user: alma_user
  password: ${POSTGRES_PASSWORD}
  vector_index_type: hnsw  # hnsw | ivfflat
```

**Qdrant:**
```yaml
storage: qdrant
qdrant:
  url: http://localhost:6333
  api_key: ${QDRANT_API_KEY}  # Optional for cloud
  collection_prefix: alma
```

**Pinecone:**
```yaml
storage: pinecone
pinecone:
  api_key: ${PINECONE_API_KEY}
  environment: us-east-1-aws
  index_name: alma-memories
```

**Chroma:**
```yaml
storage: chroma
chroma:
  persist_directory: .alma/chroma
  # Or for client-server mode:
  # host: localhost
  # port: 8000
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
| Multi-Agent Sharing | Cross-agent memory access | Done |
| Memory Consolidation | LLM-powered deduplication | Done |
| Event System | Webhooks and callbacks | Done |
| TypeScript SDK | JavaScript/TypeScript client | Done |
| Qdrant Backend | Vector database support | Done |
| Pinecone Backend | Serverless vector DB | Done |
| Chroma Backend | Lightweight vector DB | Done |
| Graph Abstraction | Pluggable graph backends | Done |
| Testing Module | Mocks and factories for testing | Done |
| Workflow Context | Checkpoints, state merging, artifacts | Done |
| Session Persistence | Persistent session handoffs | Done |
| Scoped Retrieval | Filter by workflow/agent/project | Done |
| MCP Workflow Tools | 8 additional MCP tools | Done |
| TypeScript SDK v0.6.0 | Full workflow API support | Done |

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

**Qdrant connection refused**
```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant
```

**Pinecone index not found**
- Ensure your index exists in the Pinecone console
- Check that `index_name` in config matches your index

**Embeddings dimension mismatch**
- Ensure `embedding_dim` in config matches your embedding provider
- Local: 384, Azure text-embedding-3-small: 1536

**Memgraph connection refused**
```bash
# Start Memgraph with Docker
docker run -p 7687:7687 memgraph/memgraph-mage
```

**Kuzu database locked**
- Ensure only one process accesses the database at a time
- Use `read_only=True` for concurrent read access

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
- Additional LLM provider integrations (Ollama, Groq)
- Frontend dashboard for memory visualization

---

## Roadmap

**Completed (v0.6.0):**
- Workflow context layer (checkpoints, state merging, artifacts)
- Session persistence
- Scoped retrieval
- MCP workflow tools (8 new tools)
- TypeScript SDK v0.6.0 with full workflow support

**Completed (v0.5.0):**
- Multi-agent memory sharing
- Memory consolidation engine
- Event system / webhooks
- TypeScript SDK (initial)
- Qdrant, Pinecone, Chroma backends
- Graph database abstraction

**Next:**
- Memory compression / summarization
- Temporal reasoning (time-aware retrieval)
- Proactive memory suggestions
- Visual memory explorer dashboard
- Additional graph backends (Neptune, TigerGraph)

---

## License

MIT

---

## Support the Project

If ALMA helps your AI agents get smarter:

- **Star this repo** - It helps others discover ALMA
- **[Buy me a coffee](https://buymeacoffee.com/aiagentsprp)** - Support continued development
- **[Sponsor on GitHub](https://github.com/sponsors/RBKunnela)** - Become an official sponsor
- **Contribute** - PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Links

- **Documentation:** [alma-memory.pages.dev](https://alma-memory.pages.dev)
- **PyPI:** [pypi.org/project/alma-memory](https://pypi.org/project/alma-memory/)
- **npm:** [@rbkunnela/alma-memory](https://github.com/RBKunnela/ALMA-memory/pkgs/npm/alma-memory)
- **Issues:** [GitHub Issues](https://github.com/RBKunnela/ALMA-memory/issues)

---

**Built for AI agents that get better with every task.**

*Created by [@RBKunnela](https://github.com/RBKunnela)*

---

## Technical Documentation

For detailed technical reference including architecture diagrams, API specifications, storage backend configuration, and performance tuning, see the **[Technical Documentation](docs/TECHNICAL.md)**.

---

## Share ALMA

Help other developers discover ALMA:

- **Twitter/X:** "Just found @ALMA_Memory - finally an AI agent memory framework that actually works. Scoped learning, anti-patterns, multi-agent sharing. Way better than Mem0. https://alma-memory.pages.dev"
- **LinkedIn:** Share how ALMA improved your AI agent workflows
- **Reddit:** Post in r/MachineLearning, r/LocalLLaMA, r/ClaudeAI, r/artificial
- **Hacker News:** Submit the landing page - we'd love your upvotes!
- **Dev.to / Medium:** Write about your experience using ALMA

---

## Keywords

*AI agent memory, persistent memory for LLMs, Mem0 alternative, LangChain memory replacement, agent learning framework, MCP memory server, Claude Code memory, vector database for agents, pgvector memory, Qdrant memory, Pinecone memory, multi-agent memory sharing, AI memory architecture, semantic memory for AI, long-term memory AI agents, memory-augmented AI, retrieval-augmented generation, RAG memory, agentic memory system*
