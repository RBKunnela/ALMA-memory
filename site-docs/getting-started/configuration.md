# Configuration

ALMA is configured via a YAML file, typically at `.alma/config.yaml`.

## Basic Configuration

```yaml
alma:
  project_id: "my-project"
  storage: sqlite           # Storage backend
  embedding_provider: local # Embedding provider
  storage_dir: .alma        # Where to store data
  db_name: alma.db          # Database filename
  embedding_dim: 384        # Embedding dimensions

  agents:
    helena:
      domain: coding
      can_learn:
        - testing_strategies
        - selector_patterns
      cannot_learn:
        - backend_logic
      min_occurrences_for_heuristic: 3
```

## Storage Backends

### SQLite (Local Development)

```yaml
storage: sqlite
storage_dir: .alma
db_name: alma.db
```

### PostgreSQL (Production)

```yaml
storage: postgres
postgres:
  host: localhost
  port: 5432
  database: alma
  user: alma_user
  password: ${POSTGRES_PASSWORD}  # Environment variable
  vector_index_type: hnsw         # hnsw or ivfflat
```

### Qdrant

```yaml
storage: qdrant
qdrant:
  url: http://localhost:6333
  api_key: ${QDRANT_API_KEY}  # Optional for cloud
  collection_prefix: alma
```

### Pinecone

```yaml
storage: pinecone
pinecone:
  api_key: ${PINECONE_API_KEY}
  environment: us-east-1-aws
  index_name: alma-memories
```

### Chroma

```yaml
storage: chroma
chroma:
  persist_directory: .alma/chroma
  # Or for client-server mode:
  # host: localhost
  # port: 8000
```

### Azure Cosmos DB

```yaml
storage: azure
azure:
  endpoint: ${COSMOS_ENDPOINT}
  key: ${COSMOS_KEY}
  database_name: alma
  container_name: memories
```

## Embedding Providers

| Provider | Model | Dimensions | Config |
|----------|-------|------------|--------|
| `local` | all-MiniLM-L6-v2 | 384 | Free, offline |
| `azure` | text-embedding-3-small | 1536 | Requires API key |
| `mock` | Hash-based | 384 | Testing only |

```yaml
embedding_provider: local  # or azure, mock
embedding_dim: 384         # Must match provider
```

## Agent Configuration

### Basic Agent

```yaml
agents:
  my-agent:
    domain: general
    can_learn:
      - strategies
      - patterns
    cannot_learn: []
    min_occurrences_for_heuristic: 3
```

### Scoped Learning

```yaml
agents:
  frontend-tester:
    domain: coding
    can_learn:
      - testing_strategies
      - selector_patterns
      - form_validation
    cannot_learn:
      - backend_logic
      - database_queries
      - api_design
```

### Multi-Agent Sharing

```yaml
agents:
  senior_dev:
    can_learn: [architecture, best_practices]
    share_with: [junior_dev, qa_agent]  # Others can read my memories

  junior_dev:
    can_learn: [coding_patterns]
    inherit_from: [senior_dev]  # I can read senior's memories

  qa_agent:
    can_learn: [testing_strategies]
    inherit_from: [senior_dev]
```

## Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
postgres:
  password: ${POSTGRES_PASSWORD}

pinecone:
  api_key: ${PINECONE_API_KEY}
```

## Complete Example

```yaml
alma:
  project_id: "production-app"
  storage: postgres
  embedding_provider: azure
  embedding_dim: 1536

  postgres:
    host: ${DB_HOST}
    port: 5432
    database: alma_prod
    user: alma
    password: ${DB_PASSWORD}
    vector_index_type: hnsw

  azure_openai:
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_key: ${AZURE_OPENAI_KEY}
    deployment_name: text-embedding-3-small

  agents:
    helena:
      domain: coding
      can_learn:
        - testing_strategies
        - selector_patterns
        - accessibility_checks
      cannot_learn:
        - backend_logic
      min_occurrences_for_heuristic: 3
      share_with: [qa_lead]

    victor:
      domain: coding
      can_learn:
        - api_patterns
        - database_queries
        - performance_optimization
      cannot_learn:
        - frontend_selectors
      inherit_from: [senior_architect]

    senior_architect:
      domain: coding
      can_learn:
        - architecture_decisions
        - best_practices
        - security_patterns
      share_with: [helena, victor]
```

## Next Steps

- [Storage Backends Guide](../guides/storage-backends.md) - Deep dive into each backend
- [Multi-Agent Sharing](../guides/multi-agent-sharing.md) - Advanced sharing patterns
