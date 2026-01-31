# Storage Backends Guide

ALMA supports multiple storage backends to fit different deployment scenarios. This guide covers setup and configuration for each backend.

## Quick Comparison

| Backend | Best For | Vector Search | Setup Complexity | Cost |
|---------|----------|---------------|------------------|------|
| SQLite + FAISS | Local dev, prototyping | Yes | Low | Free |
| PostgreSQL + pgvector | Production, self-hosted | Yes (HNSW) | Medium | Self-hosted |
| Qdrant | Managed vector DB | Yes (HNSW) | Low | Free tier available |
| Pinecone | Serverless, no infra | Yes | Low | Pay-per-use |
| Chroma | Lightweight local | Yes | Low | Free |
| Azure Cosmos DB | Enterprise, Azure | Yes (DiskANN) | Medium | Azure pricing |

---

## SQLite + FAISS (Default)

Best for local development and prototyping. Zero external dependencies.

### Installation

```bash
pip install alma-memory[local]
```

### Configuration

```yaml
alma:
  storage: sqlite
  storage_dir: .alma        # Where to store database files
  db_name: alma.db          # Database filename
  embedding_dim: 384        # Must match embedding provider
```

### Features

- Automatic FAISS index management
- Lazy index rebuilding for performance
- Thread-safe operations
- Full-text search fallback when vectors unavailable

---

## PostgreSQL + pgvector

Production-ready with high availability support.

### Installation

```bash
pip install alma-memory[postgres]
```

### Prerequisites

1. PostgreSQL 14+ with pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Create database:
   ```sql
   CREATE DATABASE alma;
   ```

### Configuration

```yaml
alma:
  storage: postgres
  embedding_dim: 384

postgres:
  host: localhost
  port: 5432
  database: alma
  user: alma_user
  password: ${POSTGRES_PASSWORD}  # Environment variable

  # Vector index type (optional)
  vector_index_type: hnsw  # hnsw (recommended) or ivfflat

  # Connection pool (optional)
  pool_min_size: 5
  pool_max_size: 20
```

### Index Types

**HNSW (Recommended)**
- Better recall with similar performance
- Works on empty tables
- Slightly more memory usage

**IVFFlat**
- Requires data to build index
- Lower memory footprint
- May need retraining as data grows

### Schema

Tables are created automatically:
- `heuristics`
- `outcomes`
- `user_preferences`
- `domain_knowledge`
- `anti_patterns`

Each table includes:
- `embedding vector(384)` - Vector column
- `created_at`, `updated_at` - Timestamps with indexes

---

## Qdrant

Managed vector database with excellent scaling.

### Installation

```bash
pip install alma-memory[qdrant]
```

### Local Development

Start Qdrant with Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Configuration

```yaml
alma:
  storage: qdrant
  embedding_dim: 384

qdrant:
  url: http://localhost:6333
  api_key: ${QDRANT_API_KEY}      # Optional for cloud
  collection_prefix: alma          # Prefix for collection names
  prefer_grpc: false               # Use gRPC for better performance
```

### Cloud Setup (Qdrant Cloud)

1. Create account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Get your URL and API key:

```yaml
qdrant:
  url: https://your-cluster.qdrant.io
  api_key: ${QDRANT_API_KEY}
```

### Collections

ALMA creates these collections automatically:
- `{prefix}_heuristics`
- `{prefix}_outcomes`
- `{prefix}_user_preferences`
- `{prefix}_domain_knowledge`
- `{prefix}_anti_patterns`

---

## Pinecone

Serverless vector database - no infrastructure to manage.

### Installation

```bash
pip install alma-memory[pinecone]
```

### Setup

1. Create account at [pinecone.io](https://pinecone.io)
2. Create an index with:
   - Dimension: 384 (or match your embedding provider)
   - Metric: cosine
   - Serverless: recommended

### Configuration

```yaml
alma:
  storage: pinecone
  embedding_dim: 384

pinecone:
  api_key: ${PINECONE_API_KEY}
  index_name: alma-memories

  # For serverless indexes
  cloud: aws                # aws or gcp
  region: us-east-1
```

### Namespaces

ALMA uses namespaces to separate memory types:
- `heuristics`
- `outcomes`
- `user_preferences`
- `domain_knowledge`
- `anti_patterns`

All namespaces are within your single index.

### Best Practices

- Use serverless for automatic scaling
- Set appropriate quotas in Pinecone dashboard
- Monitor usage via Pinecone console

---

## Chroma

Lightweight, embedded vector database.

### Installation

```bash
pip install alma-memory[chroma]
```

### Configuration

**Persistent Mode (Recommended):**
```yaml
alma:
  storage: chroma
  embedding_dim: 384

chroma:
  persist_directory: .alma/chroma
```

**Client-Server Mode:**
```yaml
chroma:
  host: localhost
  port: 8000
```

**Ephemeral Mode (Testing):**
```yaml
chroma:
  ephemeral: true
```

### Starting Chroma Server

For client-server mode:
```bash
chroma run --path /path/to/data --host localhost --port 8000
```

### Collections

ALMA creates these collections:
- `alma_heuristics`
- `alma_outcomes`
- `alma_user_preferences`
- `alma_domain_knowledge`
- `alma_anti_patterns`

---

## Azure Cosmos DB

Enterprise-grade with Azure ecosystem integration.

### Installation

```bash
pip install alma-memory[azure]
```

### Prerequisites

1. Azure account with Cosmos DB access
2. Cosmos DB account with:
   - API: Core (SQL)
   - Vector search capability enabled

### Configuration

```yaml
alma:
  storage: azure
  embedding_dim: 1536  # Azure OpenAI default

azure:
  cosmos_endpoint: https://your-account.documents.azure.com:443/
  cosmos_key: ${AZURE_COSMOS_KEY}
  database_name: alma

  # Optional: Use managed identity instead of key
  use_managed_identity: false
```

### With Azure OpenAI Embeddings

```yaml
alma:
  embedding_provider: azure
  embedding_dim: 1536

azure:
  openai_endpoint: https://your-resource.openai.azure.com/
  openai_key: ${AZURE_OPENAI_KEY}
  openai_deployment: text-embedding-3-small
```

---

## Migration Between Backends

ALMA doesn't provide automatic migration. To migrate:

1. Export data using retrieval with high `top_k`:
   ```python
   all_heuristics = alma.storage.get_heuristics(
       project_id="my-project",
       top_k=10000
   )
   ```

2. Initialize new backend:
   ```python
   new_alma = ALMA.from_config("new_config.yaml")
   ```

3. Import data:
   ```python
   for h in all_heuristics:
       new_alma.storage.save_heuristic(h)
   ```

---

## Graph Database Backends

ALMA supports graph database backends for storing entity relationships. These are separate from the main storage backends and are used by the Graph Memory system.

### Quick Comparison (Graph Backends)

| Backend | Best For | Deployment | Setup Complexity | Cost |
|---------|----------|------------|------------------|------|
| In-Memory | Testing | None | Zero | Free |
| Neo4j | Production | Server/Cloud | Medium | Free tier available |
| Memgraph | High-performance | Server/Docker | Medium | Free |
| Kuzu | Embedded/Local | File-based | Low | Free |

---

## Memgraph

In-memory graph database compatible with Neo4j's Bolt protocol. Excellent for high-performance graph operations.

### Installation

```bash
pip install alma-memory[memgraph]
# Note: Uses the neo4j Python driver
pip install neo4j
```

### Starting Memgraph

Using Docker:
```bash
docker run -p 7687:7687 memgraph/memgraph-mage
```

### Configuration

```python
from alma.graph.backends.memgraph import MemgraphBackend

# Basic connection (no auth)
backend = MemgraphBackend(
    uri="bolt://localhost:7687",
)

# With authentication (if enabled)
backend = MemgraphBackend(
    uri="bolt://localhost:7687",
    username="memgraph",
    password="your-password",
)
```

### Features

- **In-Memory Performance**: All data kept in RAM for fast queries
- **Cypher Compatible**: Uses same query language as Neo4j
- **MAGE Extensions**: Optional analytics algorithms (PageRank, community detection)
- **Bolt Protocol**: Uses standard Neo4j driver

### Usage Example

```python
from alma.graph.backends.memgraph import MemgraphBackend
from alma.graph.store import Entity, Relationship

# Create backend
backend = MemgraphBackend(uri="bolt://localhost:7687")

# Add entities
entity = Entity(
    id="user-123",
    name="Alice",
    entity_type="Person",
    properties={"role": "developer", "project_id": "my-project"}
)
backend.add_entity(entity)

# Add relationships
rel = Relationship(
    id="rel-1",
    source_id="user-123",
    target_id="team-456",
    relation_type="MEMBER_OF",
    confidence=1.0
)
backend.add_relationship(rel)

# Query relationships
relationships = backend.get_relationships("user-123")

# Search entities by name
results = backend.search_entities("Alice", top_k=10)

# Get entities by type
people = backend.get_entities(entity_type="Person", limit=100)

# Clean up
backend.close()
```

### Notes

- Memgraph typically runs without authentication by default
- Use empty strings for username/password if auth is disabled
- Vector similarity search requires Memgraph MAGE extensions
- Falls back to text search when vector operations unavailable

---

## Kuzu

Embedded graph database - like SQLite but for graph data. No server required.

### Installation

```bash
pip install alma-memory[kuzu]
pip install kuzu
```

### Configuration

```python
from alma.graph.backends.kuzu import KuzuBackend

# Persistent mode (data saved to disk)
backend = KuzuBackend(database_path="./my_graph_db")

# In-memory mode (data lost when closed)
backend = KuzuBackend()  # No path = in-memory

# Read-only mode
backend = KuzuBackend(
    database_path="./my_graph_db",
    read_only=True
)
```

### Features

- **Embedded**: No server required, runs in-process
- **Persistent or In-Memory**: Choose based on your needs
- **Cypher-Compatible**: Familiar query syntax
- **Lightweight**: Minimal resource footprint
- **Thread-Safe**: Safe for concurrent access

### Usage Example

```python
from alma.graph.backends.kuzu import KuzuBackend
from alma.graph.store import Entity, Relationship

# Create persistent backend
backend = KuzuBackend(database_path="./graph_data")

# Add entities
alice = Entity(
    id="alice-1",
    name="Alice",
    entity_type="Person",
    properties={"department": "Engineering"}
)
backend.add_entity(alice)

project = Entity(
    id="proj-1",
    name="ALMA",
    entity_type="Project",
    properties={"status": "active"}
)
backend.add_entity(project)

# Add relationship
rel = Relationship(
    id="works-on-1",
    source_id="alice-1",
    target_id="proj-1",
    relation_type="WORKS_ON",
    confidence=1.0
)
backend.add_relationship(rel)

# Query relationships with direction
outgoing = backend.get_relationships_directional(
    entity_id="alice-1",
    direction="outgoing"
)

# Search entities
results = backend.search_entities("alice", top_k=5)

# Filter by type
projects = backend.get_entities(
    entity_type="Project",
    project_id="my-project",
    limit=50
)

# Clear all data
backend.clear()

# Close connection
backend.close()
```

### Schema

Kuzu automatically creates this schema on first use:

```
Entity Node Table:
- id (STRING, PRIMARY KEY)
- name (STRING)
- entity_type (STRING)
- properties (STRING, JSON)
- project_id (STRING)
- agent (STRING)
- created_at (STRING)

RELATES_TO Edge Table:
- id (STRING)
- relation_type (STRING)
- properties (STRING, JSON)
- confidence (DOUBLE)
- created_at (STRING)
```

### Best Practices

- Use persistent mode for production workloads
- In-memory mode is ideal for testing
- The database directory is created automatically
- Call `close()` to ensure data is properly flushed

---

## Performance Tips

### General

- Use appropriate `top_k` values (5-10 for most use cases)
- Enable caching for read-heavy workloads
- Use batch operations for bulk writes

### PostgreSQL

- Increase `shared_buffers` and `work_mem`
- Use HNSW index for better recall
- Regular `VACUUM ANALYZE`

### Qdrant

- Use gRPC (`prefer_grpc: true`) for better performance
- Configure appropriate shard count for large collections

### Pinecone

- Use serverless for automatic scaling
- Batch upserts (up to 100 vectors per request)

### Chroma

- Use persistent mode for production
- Client-server mode for multi-process access
