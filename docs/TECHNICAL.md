# ALMA Technical Documentation

> Detailed technical reference for ALMA (Agent Learning Memory Architecture)

For a gentler introduction, see the [main README](../README.md).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Memory Types](#memory-types)
- [Storage Backends](#storage-backends)
- [Embedding Providers](#embedding-providers)
- [Graph Memory](#graph-memory)
- [API Reference](#api-reference)
- [MCP Protocol](#mcp-protocol)
- [Configuration Reference](#configuration-reference)
- [Performance Tuning](#performance-tuning)
- [Security Considerations](#security-considerations)

---

## Architecture Overview

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
|  +-------------+  +---------------+  +------------------+               |
|  | Event       |  | TypeScript    |  | Workflow         |               |
|  | System      |  | SDK           |  | Context          |               |
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
|  +---------------+  +------------------+  +---------------+             |
|  |    Qdrant     |  |    Pinecone      |  |    Chroma     |             |
|  +---------------+  +------------------+  +---------------+             |
+-------------------------------------------------------------------------+
|  GRAPH LAYER                                                            |
|  +---------------+  +------------------+  +---------------+             |
|  |    Neo4j      |  |    Memgraph      |  |     Kuzu      |             |
|  +---------------+  +------------------+  +---------------+             |
+-------------------------------------------------------------------------+
|  INTEGRATION LAYER                                                      |
|  +-------------------------------------------------------------------+  |
|  |                    MCP Server (stdio / HTTP)                      |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

---

## Core Components

### ALMA Class

The main entry point. Coordinates retrieval, learning, and storage.

```python
from alma import ALMA, MemoryScope
from alma.storage.sqlite_local import SQLiteStorage
from alma.retrieval.engine import RetrievalEngine
from alma.learning.protocols import LearningProtocol

# Create components
storage = SQLiteStorage(storage_dir=".alma", db_name="alma.db")
retrieval = RetrievalEngine(storage=storage, embedding_provider="local")
learning = LearningProtocol(storage=storage, scopes=scopes)

# Initialize ALMA
alma = ALMA(
    storage=storage,
    retrieval_engine=retrieval,
    learning_protocol=learning,
    project_id="my-project",
    scopes=scopes
)
```

### Memory Scope

Defines what an agent can and cannot learn:

```python
from alma import MemoryScope

scope = MemoryScope(
    agent_name="helena",
    can_learn=["testing_strategies", "selector_patterns", "form_validation"],
    cannot_learn=["backend_logic", "database_queries"],
    min_occurrences_for_heuristic=3,  # Require 3 successes before creating heuristic
    inherit_from=["senior_tester"],    # Read memories from these agents
    share_with=["junior_tester"]       # Allow these agents to read my memories
)
```

### Retrieval Engine

Handles semantic search and multi-factor scoring:

```python
from alma.retrieval.engine import RetrievalEngine
from alma.retrieval.scoring import ScoringConfig

retrieval = RetrievalEngine(
    storage=storage,
    embedding_provider="local",  # or "azure", "mock"
    scoring_config=ScoringConfig(
        similarity_weight=0.4,    # Vector similarity importance
        recency_weight=0.2,       # Recent memories score higher
        success_weight=0.25,      # Successful outcomes score higher
        confidence_weight=0.15    # High-confidence heuristics score higher
    )
)
```

### Learning Protocol

Manages outcome recording and heuristic formation:

```python
from alma.learning.protocols import LearningProtocol

learning = LearningProtocol(
    storage=storage,
    scopes={"helena": helena_scope, "victor": victor_scope},
    auto_consolidate=True,         # Auto-merge similar heuristics
    consolidation_threshold=0.85   # Similarity threshold for merging
)
```

---

## Memory Types

### Heuristic

Learned strategies with confidence scores:

```python
@dataclass
class Heuristic:
    id: str
    agent: str
    project_id: str
    category: str
    strategy: str
    supporting_outcomes: List[str]  # Outcome IDs that support this
    confidence: float               # 0.0 to 1.0
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
```

### Outcome

Raw task completion records:

```python
@dataclass
class Outcome:
    id: str
    agent: str
    project_id: str
    task: str
    task_type: Optional[str]
    outcome: Literal["success", "failure"]
    strategy_used: str
    duration_ms: Optional[int]
    error_message: Optional[str]
    feedback: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any]
```

### UserPreference

User constraints that persist across sessions:

```python
@dataclass
class UserPreference:
    id: str
    user_id: str
    project_id: str
    category: str  # code_style, communication, workflow, etc.
    preference: str
    source: Literal["explicit_instruction", "inferred_from_feedback"]
    created_at: datetime
    metadata: Dict[str, Any]
```

### DomainKnowledge

Factual information about the domain:

```python
@dataclass
class DomainKnowledge:
    id: str
    agent: str
    project_id: str
    domain: str
    fact: str
    source: Literal["documentation", "code_analysis", "user_stated"]
    created_at: datetime
    metadata: Dict[str, Any]
```

### AntiPattern

What NOT to do:

```python
@dataclass
class AntiPattern:
    id: str
    agent: str
    project_id: str
    pattern: str
    why_bad: str
    better_alternative: str
    created_at: datetime
    metadata: Dict[str, Any]
```

---

## Storage Backends

### SQLite + FAISS (Local Development)

```python
from alma.storage.sqlite_local import SQLiteStorage

storage = SQLiteStorage(
    storage_dir=".alma",
    db_name="alma.db",
    embedding_dim=384
)
```

**Requirements:** `pip install alma-memory[local]`

### PostgreSQL + pgvector

```python
from alma.storage.postgresql import PostgreSQLStorage

storage = PostgreSQLStorage(
    host="localhost",
    port=5432,
    database="alma",
    user="alma_user",
    password="secret",
    embedding_dim=1536,
    vector_index_type="hnsw"  # or "ivfflat"
)
```

**Requirements:**
- `pip install alma-memory[postgres]`
- PostgreSQL with pgvector extension: `CREATE EXTENSION vector;`

### Qdrant

```python
from alma.storage.qdrant import QdrantStorage

storage = QdrantStorage(
    url="http://localhost:6333",
    api_key=None,  # Optional for cloud
    collection_prefix="alma",
    embedding_dim=384
)
```

**Requirements:** `pip install alma-memory[qdrant]`

### Pinecone

```python
from alma.storage.pinecone import PineconeStorage

storage = PineconeStorage(
    api_key="your-api-key",
    environment="us-east-1-aws",
    index_name="alma-memories",
    embedding_dim=1536
)
```

**Requirements:** `pip install alma-memory[pinecone]`

### Chroma

```python
from alma.storage.chroma import ChromaStorage

# Persistent mode
storage = ChromaStorage(
    persist_directory=".alma/chroma",
    embedding_dim=384
)

# Client-server mode
storage = ChromaStorage(
    host="localhost",
    port=8000,
    embedding_dim=384
)
```

**Requirements:** `pip install alma-memory[chroma]`

### Azure Cosmos DB

```python
from alma.storage.azure_cosmos import AzureCosmosStorage

storage = AzureCosmosStorage(
    endpoint="https://your-account.documents.azure.com",
    key="your-key",
    database_name="alma",
    container_name="memories",
    embedding_dim=1536
)
```

**Requirements:** `pip install alma-memory[azure]`

---

## Embedding Providers

### Local (Sentence Transformers)

```python
from alma.retrieval.embeddings import LocalEmbedder

embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
# Output: 384 dimensions
```

### Azure OpenAI

```python
from alma.retrieval.embeddings import AzureEmbedder

embedder = AzureEmbedder(
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-key",
    deployment_name="text-embedding-3-small"
)
# Output: 1536 dimensions
```

### Mock (Testing)

```python
from alma.retrieval.embeddings import MockEmbedder

embedder = MockEmbedder(embedding_dim=384)
# Deterministic hash-based embeddings for testing
```

---

## Graph Memory

### Creating a Graph Backend

```python
from alma.graph import create_graph_backend, BackendGraphStore

# Neo4j
backend = create_graph_backend(
    "neo4j",
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j",
    password="password"
)

# Memgraph
backend = create_graph_backend("memgraph", uri="bolt://localhost:7687")

# Kuzu (embedded)
backend = create_graph_backend("kuzu", database_path="./graph_db")

# In-memory (testing)
backend = create_graph_backend("memory")

# Create store
graph = BackendGraphStore(backend)
```

### Entity Extraction

```python
from alma.graph.extraction import EntityExtractor

extractor = EntityExtractor()
entities, relationships = extractor.extract(
    "Alice from Acme Corp reviewed the PR that Bob submitted."
)

# entities: [Entity(name="Alice", type="PERSON"), Entity(name="Acme Corp", type="ORGANIZATION"), ...]
# relationships: [Relationship(source="Alice", target="Acme Corp", type="WORKS_FOR"), ...]
```

---

## API Reference

### Python SDK

See [API Documentation](api/README.md) for complete reference.

#### Core Methods

| Method | Description |
|--------|-------------|
| `retrieve(task, agent, top_k, user_id, include_shared)` | Get relevant memories |
| `learn(agent, task, outcome, strategy_used, ...)` | Record task outcome |
| `add_preference(user_id, category, preference, source)` | Add user preference |
| `add_knowledge(agent, domain, fact, source)` | Add domain knowledge |
| `add_anti_pattern(agent, pattern, why_bad, better_alternative)` | Add anti-pattern |
| `forget(agent, older_than_days, below_confidence)` | Prune memories |
| `stats(agent)` | Get statistics |

#### Workflow Methods (v0.6.0)

| Method | Description |
|--------|-------------|
| `checkpoint(workflow_id, state, metadata)` | Save workflow state |
| `resume(workflow_id)` | Resume from checkpoint |
| `merge_states(workflow_id, states, reducer)` | Merge parallel states |
| `link_artifact(workflow_id, artifact_type, ref, metadata)` | Link artifact |
| `get_artifacts(workflow_id, artifact_type)` | Get artifacts |
| `cleanup_checkpoints(older_than_days)` | Clean old checkpoints |
| `retrieve_scoped(query, scope, agent)` | Scoped retrieval |

### TypeScript SDK

```typescript
import { ALMA } from '@rbkunnela/alma-memory';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  timeout: 30000,
  retry: { maxRetries: 3, baseDelay: 1000, maxDelay: 10000 }
});

// All methods return Promises
await alma.retrieve({ query: 'task', agent: 'name', topK: 5 });
await alma.learn({ agent: 'name', task: 'task', outcome: 'success', strategyUsed: '...' });
await alma.checkpoint({ workflowId: 'wf-1', state: {...} });
```

---

## MCP Protocol

### Starting the Server

```bash
# Stdio mode (for Claude Code)
python -m alma.mcp --config .alma/config.yaml

# HTTP mode (for TypeScript SDK)
python -m alma.mcp --config .alma/config.yaml --http --port 8765
```

### Available Tools (16)

| Tool | Parameters |
|------|------------|
| `alma_retrieve` | task, agent, top_k?, user_id? |
| `alma_learn` | agent, task, outcome, strategy_used, task_type?, duration_ms?, error_message?, feedback? |
| `alma_add_preference` | user_id, category, preference, source? |
| `alma_add_knowledge` | agent, domain, fact, source? |
| `alma_forget` | agent?, older_than_days?, below_confidence? |
| `alma_stats` | agent? |
| `alma_health` | - |
| `alma_consolidate` | agent, memory_type, similarity_threshold?, use_llm?, dry_run? |
| `alma_checkpoint` | workflow_id, state, metadata? |
| `alma_resume` | workflow_id |
| `alma_merge_states` | workflow_id, states, reducer? |
| `alma_workflow_learn` | workflow_id, agent, task, outcome, strategy_used, ... |
| `alma_link_artifact` | workflow_id, artifact_type, ref, metadata? |
| `alma_get_artifacts` | workflow_id, artifact_type? |
| `alma_cleanup_checkpoints` | older_than_days? |
| `alma_retrieve_scoped` | query, scope, agent? |

---

## Configuration Reference

### Full Configuration Schema

```yaml
alma:
  # Required
  project_id: string
  storage: sqlite | postgres | qdrant | pinecone | chroma | azure | file

  # Embedding
  embedding_provider: local | azure | mock
  embedding_dim: 384 | 1536  # Must match provider

  # Storage paths (for sqlite)
  storage_dir: .alma
  db_name: alma.db

  # PostgreSQL
  postgres:
    host: localhost
    port: 5432
    database: alma
    user: string
    password: ${ENV_VAR}  # Environment variable expansion
    vector_index_type: hnsw | ivfflat

  # Qdrant
  qdrant:
    url: http://localhost:6333
    api_key: ${QDRANT_API_KEY}
    collection_prefix: alma

  # Pinecone
  pinecone:
    api_key: ${PINECONE_API_KEY}
    environment: us-east-1-aws
    index_name: alma-memories

  # Chroma
  chroma:
    persist_directory: .alma/chroma
    # OR client-server:
    host: localhost
    port: 8000

  # Azure
  azure:
    endpoint: ${COSMOS_ENDPOINT}
    key: ${COSMOS_KEY}
    database_name: alma
    container_name: memories

  # Azure OpenAI (if using azure embedding)
  azure_openai:
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_KEY}
    deployment_name: text-embedding-3-small

  # Agent definitions
  agents:
    agent_name:
      domain: coding | research | sales | general | custom
      can_learn:
        - category1
        - category2
      cannot_learn:
        - category3
      min_occurrences_for_heuristic: 3
      inherit_from:
        - other_agent
      share_with:
        - another_agent
```

---

## Performance Tuning

### Caching

```python
from alma.retrieval.cache import CacheConfig

cache_config = CacheConfig(
    enabled=True,
    max_size=1000,          # Max cached queries
    ttl_seconds=300,        # Cache TTL
    embedding_cache=True    # Cache embeddings
)
```

### Batch Operations

```python
# Batch learning (more efficient)
outcomes = [
    {"agent": "helena", "task": "...", "outcome": "success", "strategy_used": "..."},
    {"agent": "helena", "task": "...", "outcome": "failure", "strategy_used": "..."},
]
alma.learn_batch(outcomes)
```

### Index Optimization

For PostgreSQL with pgvector:
```sql
-- HNSW index (recommended for production)
CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat index (faster builds, slightly lower recall)
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

## Security Considerations

### API Key Management

- Never commit API keys to version control
- Use environment variables: `${ENV_VAR}` syntax in config
- Rotate keys regularly

### Data Privacy

- ALMA stores task descriptions and strategies
- Consider PII implications in your use case
- Use encryption at rest for sensitive data (database-level)

### MCP Server

- HTTP mode should be behind authentication in production
- Use HTTPS for remote deployments
- Consider rate limiting

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: sentence-transformers` | `pip install alma-memory[local]` |
| `pgvector extension not found` | `CREATE EXTENSION IF NOT EXISTS vector;` |
| `Embedding dimension mismatch` | Ensure `embedding_dim` matches provider |
| `Connection refused (Qdrant)` | `docker run -p 6333:6333 qdrant/qdrant` |
| `Kuzu database locked` | Only one process can write at a time |

### Debug Logging

```python
import logging
logging.getLogger("alma").setLevel(logging.DEBUG)
```

---

## Next Steps

- [Main README](../README.md) - Overview and quick start
- [Guides](guides/) - Feature-specific documentation
- [API Reference](api/) - Detailed API documentation
- [Examples](../examples/) - Working code examples
