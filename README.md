# ALMA - Agent Learning Memory Architecture

[![PyPI version](https://badge.fury.io/py/alma-memory.svg)](https://pypi.org/project/alma-memory/)
[![npm version](https://img.shields.io/npm/v/@rbkunnela/alma-memory?logo=npm&color=cb3837)](https://www.npmjs.com/package/@rbkunnela/alma-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/RBKunnela/ALMA-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/RBKunnela/ALMA-memory/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-alma--memory.pages.dev-blue)](https://alma-memory.pages.dev)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/aiagentsprp)

<div align="center">

### Persistent memory for AI agents that improves over time

**No fine-tuning. No model changes. Just smarter prompts built from relevant past experiences.**

[**Documentation**](https://alma-memory.pages.dev) | [**PyPI**](https://pypi.org/project/alma-memory/) | [**npm**](https://www.npmjs.com/package/@rbkunnela/alma-memory) | [**Support**](https://buymeacoffee.com/aiagentsprp)

</div>

---

## What ALMA Does

ALMA gives your AI agents **long-term memory**. Instead of starting every conversation from scratch, agents retrieve relevant past experiences, strategies, and knowledge — then learn from new outcomes to get better over time.

```
BEFORE TASK                        DURING TASK                      AFTER TASK
+----------------------------+     +-------------------------+     +---------------------------+
| Retrieve relevant memories |     | Agent executes with     |     | Learn from outcome        |
| - Past strategies          | --> | injected knowledge      | --> | - Success? New heuristic  |
| - Anti-patterns            |     | from memory             |     | - Failure? Anti-pattern   |
| - Domain knowledge         |     |                         |     | - Always: Knowledge grows |
+----------------------------+     +-------------------------+     +---------------------------+
```

**The core insight:** AI agents don't need weight modifications to "learn." They need **smart prompts** built from **relevant past experiences** — retrieved, scored, and injected at runtime.

---

## Key Capabilities

| Capability | What It Does |
|------------|-------------|
| **5 Memory Types** | Heuristics, outcomes, preferences, domain knowledge, anti-patterns |
| **Scoped Learning** | Agents only learn within defined domains (`can_learn` / `cannot_learn`) |
| **Anti-Pattern Tracking** | Explicit "what NOT to do" with `why_bad` + `better_alternative` |
| **Multi-Agent Sharing** | Hierarchical knowledge sharing via `inherit_from` + `share_with` |
| **7 Storage Backends** | SQLite+FAISS, PostgreSQL+pgvector, Qdrant, Pinecone, Chroma, Azure Cosmos DB, File |
| **4 Graph Backends** | Neo4j, Memgraph, Kuzu, In-memory — entity relationship tracking |
| **22 MCP Tools** | Native Claude Code integration via stdio/HTTP server |
| **RAG Integration** | Bridge any RAG system with memory signals, hybrid search, feedback loops |
| **Multi-Factor Scoring** | Similarity + Recency + Success rate + Confidence — not just vector distance |
| **Memory Lifecycle** | Decay, compression, consolidation, archival, verified retrieval |
| **Workflow Context** | Checkpoints, state merging, artifacts, scoped retrieval |
| **Event System** | Webhooks + in-process callbacks for real-time memory reactions |
| **Domain Factory** | 6 pre-built schemas: coding, research, sales, support, content, general |
| **TypeScript SDK** | Full-featured JavaScript/TypeScript client library |

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

# RAG integration (hybrid search, reranking)
pip install alma-memory[rag]

# Everything
pip install alma-memory[all]
```

**TypeScript/JavaScript:**
```bash
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

Agents only learn within their defined domains:

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

```yaml
agents:
  senior_dev:
    can_learn: [architecture, best_practices]
    share_with: [junior_dev, qa_agent]

  junior_dev:
    can_learn: [coding_patterns]
    inherit_from: [senior_dev]
```

```python
# Junior dev retrieves memories (includes senior's shared memories)
memories = alma.retrieve(
    task="Implement user authentication",
    agent="junior_dev",
    include_shared=True
)

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

### Deployment Platforms

ALMA is a library — you bring your own infrastructure:

| Platform | Storage Backend | Best For |
|----------|----------------|----------|
| **Azure** | Azure Cosmos DB (native backend) | Enterprise, compliance-heavy workloads |
| **AWS** | PostgreSQL (RDS + pgvector), Pinecone, Qdrant Cloud | Scalable production deployments |
| **Google Cloud** | PostgreSQL (Cloud SQL + pgvector), Pinecone | Serverless agents, Firebase apps |
| **Cloudflare** | Qdrant Cloud, Pinecone (external) | Edge-deployed agents |
| **Self-hosted** | SQLite+FAISS, PostgreSQL+pgvector | Full control, offline/air-gapped |

---

## RAG Integration (v0.8.0)

### RAG Bridge - Enhance RAG with Memory

Accept chunks from any RAG framework and enhance them with ALMA memory signals:

```python
from alma import ALMA, RAGBridge, RAGChunk

alma = ALMA.from_config(".alma/config.yaml")
bridge = RAGBridge(alma=alma)

# Your RAG system retrieves chunks (from LangChain, LlamaIndex, etc.)
chunks = [
    RAGChunk(id="1", text="Deploy with blue-green strategy", score=0.85),
    RAGChunk(id="2", text="Use rolling updates for zero downtime", score=0.78),
    RAGChunk(id="3", text="Always run smoke tests after deploy", score=0.72),
]

# ALMA enhances with memory signals
result = bridge.enhance(
    chunks=chunks,
    query="how to deploy auth service safely",
    agent="backend-agent",
)

for chunk in result.chunks:
    print(f"{chunk.text} (score: {chunk.final_score:.2f})")
    print(f"  Memory signals: {chunk.signals}")
```

### Hybrid Search - Vector + Keyword with RRF Fusion

```python
from alma.retrieval.hybrid import HybridSearchEngine, HybridSearchConfig
from alma.retrieval.text_search import SimpleTFIDFProvider

config = HybridSearchConfig(
    vector_weight=0.7,
    keyword_weight=0.3,
    rrf_k=60,
)

engine = HybridSearchEngine(
    config=config,
    text_provider=SimpleTFIDFProvider(),
)

engine.index_documents(["doc1 text...", "doc2 text..."])

results = engine.search(
    query="JWT authentication refresh tokens",
    vector_results=vector_hits,
    top_k=10,
)
```

**Optional BM25 support:**
```bash
pip install alma-memory[rag]  # Installs bm25s + rerankers
```

### Feedback Loop - Learn From Retrieval Outcomes

```python
from alma.rag import RetrievalFeedbackTracker

tracker = RetrievalFeedbackTracker(alma=alma)

retrieval_id = tracker.record_retrieval(
    query="deploy auth service",
    agent="backend-agent",
    chunk_ids=["1", "2", "3"],
    scores=[0.85, 0.78, 0.72],
)

tracker.record_feedback(
    retrieval_id=retrieval_id,
    helpful_chunks=["1", "3"],
    unhelpful_chunks=["2"],
)

adjustments = tracker.compute_weight_adjustments(agent="backend-agent")
# -> {'vector_weight': 0.65, 'keyword_weight': 0.35}  # Auto-tuned
```

### IR Metrics - Measure Retrieval Quality

```python
from alma.rag import RetrievalMetrics, RelevanceJudgment

metrics = RetrievalMetrics()

judgments = [
    RelevanceJudgment(query="deploy", doc_id="1", relevant=True),
    RelevanceJudgment(query="deploy", doc_id="2", relevant=False),
    RelevanceJudgment(query="deploy", doc_id="3", relevant=True),
]

result = metrics.compute(
    retrieved=["1", "2", "3"],
    judgments=judgments,
    k=10,
)

print(f"MRR: {result.mrr:.3f}")
print(f"NDCG@10: {result.ndcg:.3f}")
print(f"Recall@10: {result.recall:.3f}")
print(f"MAP: {result.map:.3f}")
```

---

## Memory Lifecycle

### Memory Consolidation

Automatically deduplicate and merge similar memories:

```python
from alma.consolidation import ConsolidationEngine

engine = ConsolidationEngine(
    storage=alma.storage,
    llm_client=my_llm_client
)

result = await engine.consolidate(
    agent="helena",
    project_id="my-project",
    memory_type="heuristics",
    similarity_threshold=0.85,
    use_llm=True,
    dry_run=False
)

print(f"Merged {result.merged_count} memories from {result.groups_found} groups")
```

### Memory Decay and Compression

- **Decay**: Time-based confidence decay for aging memories, automatic refresh on access
- **Compression**: LLM-powered summarization to reduce storage while preserving insights
- **Verified Retrieval**: Two-stage verification to filter stale or contradictory memories
- **Archive**: Soft-delete with recovery for compliance-friendly deletion

### Retrieval Modes

| Mode | Use Case |
|------|----------|
| `standard` | Default similarity-based retrieval |
| `progressive` | Start broad, narrow down |
| `verified` | Include relevance verification |
| `budget` | Token-aware retrieval with limits |

### Trust-Integrated Retrieval

Weight memories by agent trust scores. Configurable trust thresholds with context propagation across agents.

---

## Event System

React to memory changes with webhooks or in-process callbacks:

```python
from alma.events import get_emitter, MemoryEventType

def on_memory_created(event):
    print(f"Memory created: {event.memory_id} by {event.agent}")

emitter = get_emitter()
emitter.subscribe(MemoryEventType.CREATED, on_memory_created)
```

**Webhooks:**

```python
from alma.events import WebhookConfig, WebhookManager, get_emitter

manager = WebhookManager()
manager.add_webhook(WebhookConfig(
    url="https://your-app.com/alma-webhook",
    events=[MemoryEventType.CREATED, MemoryEventType.UPDATED],
    secret="your-webhook-secret",
    retry_count=3,
    retry_delay=5.0
))
manager.start(get_emitter())
```

**Event Types:** `CREATED`, `UPDATED`, `DELETED`, `ACCESSED`, `CONSOLIDATED`

---

## Graph Memory

Capture entity relationships for complex reasoning:

```python
from alma.graph import create_graph_backend, BackendGraphStore, EntityExtractor

# Neo4j (production)
backend = create_graph_backend(
    "neo4j",
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j",
    password="your-password"
)

# Also available: Memgraph, Kuzu (embedded), In-memory (testing)

graph = BackendGraphStore(backend)
extractor = EntityExtractor()

entities, relationships = extractor.extract(
    "Alice from Acme Corp reviewed the PR that Bob submitted."
)

for entity in entities:
    graph.add_entity(entity)
for rel in relationships:
    graph.add_relationship(rel)

alice_relations = graph.get_relationships("alice", relationship_type="WORKS_FOR")
```

---

## The Harness Pattern

Decouples agents from domain memory for reusable architecture:

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

harness = create_harness("coding", "helena", alma)

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

**22 MCP Tools:**

| Core Tools | Description |
|------------|-------------|
| `alma_retrieve` | Get memories for a task |
| `alma_learn` | Record task outcome |
| `alma_add_preference` | Add user preference |
| `alma_add_knowledge` | Add domain knowledge |
| `alma_forget` | Prune stale memories |
| `alma_stats` | Get memory statistics |
| `alma_health` | Health check |

| Workflow Tools | Description |
|----------------|-------------|
| `alma_consolidate` | Merge similar memories |
| `alma_checkpoint` | Save workflow state |
| `alma_resume` | Resume from checkpoint |
| `alma_merge_states` | Merge parallel agent states |
| `alma_workflow_learn` | Learn with workflow context |
| `alma_link_artifact` | Link output to workflow |
| `alma_get_artifacts` | Get workflow artifacts |
| `alma_cleanup_checkpoints` | Clean old checkpoints |
| `alma_retrieve_scoped` | Scoped memory retrieval |

| Memory Wall Tools | Description |
|-------------------|-------------|
| `alma_reinforce` | Strengthen a memory to prevent forgetting |
| `alma_get_weak_memories` | List memories at risk of decay |
| `alma_smart_forget` | Intelligent forgetting with archive |
| `alma_retrieve_verified` | Two-stage verified retrieval |
| `alma_compress_and_learn` | Compress and store content |
| `alma_extract_heuristic` | Extract patterns from experiences |

---

## Advanced Features

### Domain Memory Factory

Create ALMA instances for any domain:

```python
from alma.domains import DomainMemoryFactory, get_research_schema

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

```python
from alma.progress import ProgressTracker

tracker = ProgressTracker("my-project")

item = tracker.create_work_item(
    title="Fix authentication bug",
    description="Login fails on mobile devices",
    priority=80,
    agent="Victor",
)

tracker.update_status(item.id, "in_progress")
next_task = tracker.get_next_item(strategy="priority")
```

### Session Handoff

Maintain context across sessions:

```python
from alma.session import SessionManager

manager = SessionManager("my-project")

context = manager.start_session(agent="Helena", goal="Complete auth testing")

if context.previous_handoff:
    print(f"Last action: {context.previous_handoff.last_action}")
    print(f"Blockers: {context.previous_handoff.blockers}")

manager.create_handoff("Helena", context.session_id,
    last_action="completed_oauth_tests",
    last_outcome="success",
    next_steps=["Test refresh tokens", "Add error cases"],
)
```

### LLM-Powered Fact Extraction

```python
from alma.extraction import AutoLearner

auto_learner = AutoLearner(alma)

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

```python
from alma.confidence import ConfidenceEngine

engine = ConfidenceEngine(alma)

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
|                          ALMA v0.8.0                                    |
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
      share_with: [qa_lead]

    victor:
      domain: coding
      can_learn:
        - api_patterns
        - database_queries
      cannot_learn:
        - frontend_selectors
      inherit_from: [senior_architect]
```

### Storage Backend Configuration

<details>
<summary>PostgreSQL + pgvector</summary>

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
</details>

<details>
<summary>Qdrant</summary>

```yaml
storage: qdrant
qdrant:
  url: http://localhost:6333
  api_key: ${QDRANT_API_KEY}
  collection_prefix: alma
```
</details>

<details>
<summary>Pinecone</summary>

```yaml
storage: pinecone
pinecone:
  api_key: ${PINECONE_API_KEY}
  environment: us-east-1-aws
  index_name: alma-memories
```
</details>

<details>
<summary>Chroma</summary>

```yaml
storage: chroma
chroma:
  persist_directory: .alma/chroma
  # Or for client-server mode:
  # host: localhost
  # port: 8000
```
</details>

### Embedding Providers

| Provider | Model | Dimensions | Cost | Best For |
|----------|-------|------------|------|----------|
| **local** | all-MiniLM-L6-v2 | 384 | Free | Development, offline |
| **azure** | text-embedding-3-small | 1536 | ~$0.02/1M | Production |
| **mock** | (hash-based) | 384 | Free | Testing only |

---

## Comparisons

<details>
<summary>How ALMA compares to alternatives</summary>

If you've tried Mem0, LangChain Memory, or Graphiti, here's how ALMA differs:

| Feature | ALMA | Mem0 | LangChain | Graphiti |
|---------|------|------|-----------|----------|
| **Memory Scoping** | `can_learn` / `cannot_learn` per agent | Basic isolation | Session-based | None |
| **Anti-Pattern Learning** | `why_bad` + `better_alternative` | None | None | None |
| **Multi-Agent Sharing** | `inherit_from` + `share_with` | None | None | None |
| **Multi-Factor Scoring** | 4 factors (similarity + recency + success + confidence) | Similarity only | Similarity only | Similarity only |
| **MCP Integration** | 22 tools | None | None | None |
| **Workflow Checkpoints** | Full checkpoint/resume/merge | None | None | None |
| **TypeScript SDK** | Full-featured client | None | JavaScript wrappers | None |
| **Graph + Vector Hybrid** | 4 graph + 7 vector backends | Limited | Limited | Graph-focused |
| **Memory Consolidation** | LLM-powered deduplication | Basic | None | None |
| **Event System** | Webhooks + in-process callbacks | None | None | None |
| **Domain Factory** | 6 pre-built schemas | None | None | None |

**The key difference:** Most solutions treat memory as "store embeddings, retrieve similar." ALMA treats it as "teach agents to improve within safe boundaries."

See detailed comparisons: [ALMA vs Mem0](https://alma-memory.pages.dev/comparison/mem0-vs-alma.html) | [ALMA vs LangChain Memory](https://alma-memory.pages.dev/comparison/langchain-memory-vs-alma.html)

</details>

---

## Release History

<details>
<summary>v0.8.0 - RAG Integration Layer</summary>

- RAG Bridge: Accept chunks from any RAG framework and enhance with memory signals
- Hybrid Search: Vector + keyword with RRF fusion
- Feedback Loop: Track and auto-tune retrieval weights
- IR Metrics: MRR, NDCG, Recall, Precision, MAP
- Cross-Encoder Reranking: Pluggable reranking pipeline

</details>

<details>
<summary>v0.7.x - Memory Wall + Intelligence Layer</summary>

- Memory Decay: Time-based confidence decay
- Memory Compression: LLM + rule-based summarization
- Verified Retrieval: Two-stage verification pipeline
- Retrieval Modes: 7 cognitive task modes
- Trust-Integrated Scoring, Token Budget, Progressive Disclosure
- 6 new MCP tools for Memory Wall
- Archive System: Soft-delete with recovery
- Embedding Performance Boost: 2.6x faster via batched processing + LRU cache
- Storage Backend Factory, Consolidation Strategies, Standalone Dedup Engine

</details>

<details>
<summary>v0.6.0 - Workflow Context Layer</summary>

- Checkpoint & Resume workflow state
- State Reducers for parallel agent states
- Artifact Linking to workflows
- Scoped Retrieval by workflow/agent/project
- 8 MCP Workflow Tools
- TypeScript SDK v0.6.0 with full workflow API parity

</details>

<details>
<summary>v0.5.0 - Vector Database Backends</summary>

- Qdrant, Pinecone, Chroma backends
- Graph Database Abstraction (Neo4j, Memgraph, Kuzu, In-memory)
- Testing Module (MockStorage, MockEmbedder, factories)
- Memory Consolidation Engine
- Event System (Webhooks + callbacks)
- TypeScript SDK initial release
- Multi-Agent Memory Sharing

</details>

See [CHANGELOG.md](CHANGELOG.md) for the complete history.

---

## Roadmap

**Next:**
- Temporal reasoning (time-aware retrieval)
- Proactive memory suggestions
- Visual memory explorer dashboard
- Additional graph backends (Neptune, TigerGraph)

---

## Troubleshooting

<details>
<summary>Common Issues</summary>

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
docker run -p 7687:7687 memgraph/memgraph-mage
```

**Kuzu database locked**
- Ensure only one process accesses the database at a time
- Use `read_only=True` for concurrent read access

**Debug Logging:**
```python
import logging
logging.getLogger("alma").setLevel(logging.DEBUG)
```

</details>

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For questions, support, or contribution guidelines, email **dev@friendlyai.fi**.

**What we need most:**
- Documentation improvements
- Test coverage for edge cases
- Additional LLM provider integrations (Ollama, Groq)
- Frontend dashboard for memory visualization

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
- **Get help** - Email **dev@friendlyai.fi** for support and inquiries

---

## Links

- **Documentation:** [alma-memory.pages.dev](https://alma-memory.pages.dev)
- **PyPI:** [pypi.org/project/alma-memory](https://pypi.org/project/alma-memory/)
- **npm:** [@rbkunnela/alma-memory](https://www.npmjs.com/package/@rbkunnela/alma-memory)
- **Issues:** [GitHub Issues](https://github.com/RBKunnela/ALMA-memory/issues)
- **Technical Reference:** [Technical Documentation](docs/TECHNICAL.md)

---

**Built for AI agents that get better with every task.**

*Created by [@RBKunnela](https://github.com/RBKunnela)*
