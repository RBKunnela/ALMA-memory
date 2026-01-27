# ALMA System Architecture Document

**Version:** 0.4.0
**Last Updated:** 2026-01-28
**Status:** Production-Ready with Active Development

---

## Executive Summary

ALMA (Agent Learning Memory Architecture) is a persistent memory framework designed to make AI agents appear to "learn" without requiring model weight updates or fine-tuning. The system stores outcomes, strategies, user preferences, and domain knowledge from past interactions, then retrieves and injects relevant memories into agent prompts before each new task.

### Key Design Principles

1. **No Fine-Tuning Required** - Learning through memory injection, not model modification
2. **Scoped Learning** - Agents only learn within their defined domains
3. **Multi-Backend Support** - Pluggable storage layer (SQLite, PostgreSQL, Azure Cosmos DB)
4. **Semantic Retrieval** - Vector embeddings for relevance-based memory recall
5. **Forgetting Mechanism** - Automatic pruning of stale/low-confidence memories

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Core | Python 3.10+ | Main implementation |
| Storage | SQLite/PostgreSQL/Cosmos DB | Persistence |
| Vectors | FAISS/pgvector | Similarity search |
| Embeddings | sentence-transformers/Azure OpenAI | Text vectorization |
| Integration | MCP Protocol | Claude Code integration |

---

## High-Level Architecture Overview

```
                            ALMA Memory Architecture v0.4.0

+-----------------------------------------------------------------------------------+
|                                   CLIENT LAYER                                     |
|  +-------------+  +----------------+  +-------------+  +-----------------------+  |
|  | Claude Code |  | MCP Server     |  | Python SDK  |  | REST API (HTTP mode)  |  |
|  | Integration |  | (stdio/http)   |  | Direct Use  |  |                       |  |
|  +------+------+  +-------+--------+  +------+------+  +-----------+-----------+  |
|         |                 |                  |                     |              |
+---------+-----------------+------------------+---------------------+--------------+
          |                 |                  |                     |
          +--------+--------+------------------+---------------------+
                   |
+------------------v--------------------------------------------------------------------+
|                                    CORE LAYER                                         |
|                                                                                       |
|  +------------------+     +---------------------+     +---------------------+         |
|  |    ALMA Core     |     |  Learning Protocol  |     |  Retrieval Engine   |         |
|  |  (alma/core.py)  |<--->|  (learning/         |<--->|  (retrieval/        |         |
|  |                  |     |   protocols.py)     |     |   engine.py)        |         |
|  +--------+---------+     +----------+----------+     +----------+----------+         |
|           |                          |                           |                    |
|           |     +--------------------+---------------------------+                    |
|           |     |                    |                           |                    |
|  +--------v-----v----+    +----------v----------+     +----------v----------+         |
|  |   Memory Scopes   |    |   Scoring Engine    |     |   Cache Layer       |         |
|  |   (types.py)      |    |   (scoring.py)      |     |   (cache.py)        |         |
|  +-------------------+    +---------------------+     +---------------------+         |
|                                                                                       |
+---------------------------------------------------------------------------------------+
          |                          |                           |
          +------------+-------------+---------------------------+
                       |
+----------------------v----------------------------------------------------------------+
|                                   STORAGE LAYER                                       |
|                                                                                       |
|  +------------------+     +---------------------+     +---------------------+         |
|  |  StorageBackend  |<----| Embedding Provider  |<----|  Config Loader      |         |
|  |   (base.py)      |     |  (embeddings.py)    |     |  (config/loader.py) |         |
|  +--------+---------+     +---------------------+     +---------------------+         |
|           |                                                                           |
|     +-----+-----+-----+-----+                                                         |
|     |           |           |                                                         |
|  +--v--+     +--v--+     +--v--+     +--------+                                        |
|  |SQLite|    |Postgres|  |Azure |    |File-   |                                       |
|  |+FAISS|    |+pgvector| |Cosmos|    |Based   |                                       |
|  +------+    +--------+  +------+    +--------+                                       |
|                                                                                       |
+---------------------------------------------------------------------------------------+
          |                          |
+----------------------v----------------------------------------------------------------+
|                                EXTENSION MODULES                                      |
|                                                                                       |
|  +------------------+     +---------------------+     +---------------------+         |
|  |  Graph Memory    |     |  Auto-Learning      |     |  Session Manager    |         |
|  |  (graph/)        |     |  (extraction/)      |     |  (session/)         |         |
|  +------------------+     +---------------------+     +---------------------+         |
|                                                                                       |
|  +------------------+     +---------------------+     +---------------------+         |
|  |  Progress Track  |     |  Confidence Engine  |     |  Domain Factory     |         |
|  |  (progress/)     |     |  (confidence/)      |     |  (domains/)         |         |
|  +------------------+     +---------------------+     +---------------------+         |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

---

## Core Components and Responsibilities

### 1. ALMA Core (`alma/core.py:27-326`)

The main entry point and facade for all memory operations.

**Responsibilities:**
- Initialize and coordinate all subsystems
- Expose high-level API methods (`retrieve`, `learn`, `forget`)
- Manage agent scopes and project context
- Handle cache invalidation after mutations

**Key Methods:**
```python
# alma/core.py:51-108
@classmethod
def from_config(cls, config_path: str) -> "ALMA":
    """Initialize ALMA from a configuration file."""

# alma/core.py:128-158
def retrieve(self, task: str, agent: str, user_id: Optional[str], top_k: int) -> MemorySlice:
    """Retrieve relevant memories for a task."""

# alma/core.py:160-206
def learn(self, agent: str, task: str, outcome: str, strategy_used: str, ...) -> bool:
    """Learn from a task outcome."""
```

**Dependencies:**
- `StorageBackend` - Persistence layer
- `RetrievalEngine` - Memory retrieval with scoring
- `LearningProtocol` - Learning validation and heuristic creation

### 2. Memory Types (`alma/types.py`)

Defines the five core memory structures:

| Type | Definition | Purpose | File:Line |
|------|------------|---------|-----------|
| `Heuristic` | Learned rule with condition/strategy | "When X, do Y" patterns | `types.py:44-68` |
| `Outcome` | Task execution record | Raw learning data | `types.py:71-91` |
| `UserPreference` | User constraint/preference | Personalization | `types.py:93-108` |
| `DomainKnowledge` | Domain-specific fact | Context injection | `types.py:110-127` |
| `AntiPattern` | What NOT to do | Mistake avoidance | `types.py:129-147` |

**MemoryScope** (`types.py:22-41`) - Controls what an agent can learn:
```python
@dataclass
class MemoryScope:
    agent_name: str
    can_learn: List[str]       # Allowed domains
    cannot_learn: List[str]    # Forbidden domains
    min_occurrences_for_heuristic: int = 3
```

### 3. Learning Protocol (`alma/learning/protocols.py:25-327`)

Manages how agents learn from task outcomes.

**Key Behaviors:**
1. Creates `Outcome` records for every task execution
2. Aggregates outcomes into `Heuristic` after `min_occurrences` similar results
3. Creates `AntiPattern` from repeated failures with same error
4. Validates learning against agent scope before committing

**Heuristic Creation Logic** (`protocols.py:127-182`):
- Requires `min_occurrences` similar outcomes (default: 3)
- Only creates heuristic if confidence > 50%
- Confidence = success_count / occurrence_count

**Anti-Pattern Detection** (`protocols.py:184-220`):
- Triggered on failure with error message
- Requires 2+ similar failures to create anti-pattern

### 4. Retrieval Engine (`alma/retrieval/engine.py:19-288`)

Handles semantic search and memory scoring.

**Pipeline:**
1. Check cache for query (if enabled)
2. Generate query embedding
3. Retrieve candidates from storage with vector search
4. Score and rank using `MemoryScorer`
5. Apply score threshold
6. Return top-k per memory type
7. Cache result

**Scoring Weights** (`scoring.py:16-35`):
```python
@dataclass
class ScoringWeights:
    similarity: float = 0.4      # Semantic relevance
    recency: float = 0.3         # How recent
    success_rate: float = 0.2    # Historical success
    confidence: float = 0.1      # Stored confidence
```

**Recency Decay** (`scoring.py:261-283`):
- Exponential decay: `score = 0.5 ^ (days_ago / half_life)`
- Default half-life: 30 days

### 5. Cache Layer (`alma/retrieval/cache.py:170-521`)

Multi-backend caching with TTL expiration.

**Backends:**
| Backend | Class | Use Case |
|---------|-------|----------|
| In-Memory | `RetrievalCache` | Single instance, development |
| Redis | `RedisCache` | Distributed, production |
| Null | `NullCache` | Testing, disabled caching |

**Features:**
- LRU eviction when max size reached
- Selective invalidation by agent/project
- Performance metrics (hit rate, latency percentiles)
- Thread-safe operations

---

## Storage Layer Architecture

### Abstract Interface (`alma/storage/base.py:21-373`)

All storage backends implement `StorageBackend` ABC with:

**Write Operations:**
- `save_heuristic()`, `save_outcome()`, `save_user_preference()`
- `save_domain_knowledge()`, `save_anti_pattern()`

**Read Operations:**
- `get_heuristics()`, `get_outcomes()`, `get_user_preferences()`
- `get_domain_knowledge()`, `get_anti_patterns()`
- All read methods support optional embedding-based vector search

**Update Operations:**
- `update_heuristic()`, `increment_heuristic_occurrence()`
- `update_heuristic_confidence()`, `update_knowledge_confidence()`

**Delete Operations:**
- Individual deletes by ID
- Bulk deletes: `delete_outcomes_older_than()`, `delete_low_confidence_heuristics()`

### SQLite + FAISS Backend (`alma/storage/sqlite_local.py`)

**Recommended for:** Local development, single-node deployments

**Architecture:**
- SQLite for structured data (6 tables)
- FAISS IndexFlatIP for vector search (4 indices)
- Falls back to numpy cosine similarity if FAISS unavailable

**Database Schema:**
```sql
-- alma/storage/sqlite_local.py:107-223
heuristics (id, agent, project_id, condition, strategy, confidence, ...)
outcomes (id, agent, project_id, task_type, task_description, success, ...)
preferences (id, user_id, category, preference, source, ...)
domain_knowledge (id, agent, project_id, domain, fact, ...)
anti_patterns (id, agent, project_id, pattern, why_bad, ...)
embeddings (id, memory_type, memory_id, embedding BLOB)
```

**Vector Search** (`sqlite_local.py:292-333`):
- Maintains separate FAISS index per memory type
- Embeddings stored as binary blobs in SQLite
- Indices rebuilt on startup from stored embeddings

### PostgreSQL + pgvector Backend (`alma/storage/postgresql.py`)

**Recommended for:** Production, high-availability, distributed

**Features:**
- Native `VECTOR` type with pgvector extension
- Connection pooling via `psycopg_pool`
- IVFFlat index for approximate nearest neighbor search
- Falls back to application-level similarity if pgvector unavailable

**Schema Notes:**
- Uses `alma_` prefix for all tables
- Embeddings stored directly in tables (not separate table)
- JSONB for metadata fields

**Vector Index** (`postgresql.py:280-291`):
```sql
CREATE INDEX idx_{table}_embedding
ON {schema}.{table}
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)
```

### Azure Cosmos DB Backend (`alma/storage/azure_cosmos.py`)

**Recommended for:** Azure-native deployments, enterprise scale

**Features:**
- NoSQL API with document storage
- DiskANN vector indexing (native)
- Partition key: `project_id` for efficient queries
- Built-in TTL support possible

**Container Structure:**
- `alma-heuristics` (vectors enabled)
- `alma-outcomes` (vectors enabled)
- `alma-preferences` (no vectors)
- `alma-knowledge` (vectors enabled)
- `alma-antipatterns` (vectors enabled)

### File-Based Backend (`alma/storage/file_based.py`)

**Use Case:** Testing, simple deployments, fallback

**Limitations:**
- No vector search support
- Returns all matching items, sorted by confidence/recency
- Not suitable for production workloads

---

## Data Flow Patterns

### Learning Flow

```
User Task Completion
        |
        v
+-------------------+
| alma.learn()      |
| (core.py:160)     |
+--------+----------+
         |
         v
+-------------------+
| LearningProtocol  |
| .learn()          |
| (protocols.py:50) |
+--------+----------+
         |
    +----+----+
    |         |
    v         v
+-------+  +---------------+
|Outcome|  |Check Similar  |
|Record |  |Outcomes       |
+-------+  +-------+-------+
                   |
         +--------+--------+
         |                 |
    >= min_occurrences?   < min_occurrences
         |                 |
         v                 v
+-------------------+    (wait)
|Create/Update      |
|Heuristic          |
|(protocols.py:127) |
+-------------------+
         |
         v
+-------------------+
|Invalidate Cache   |
|(core.py:203-204)  |
+-------------------+
```

### Retrieval Flow

```
Task Query
    |
    v
+-------------------+
| alma.retrieve()   |
| (core.py:128)     |
+--------+----------+
         |
         v
+-------------------+
| RetrievalEngine   |
| .retrieve()       |
| (engine.py:75)    |
+--------+----------+
         |
    +----+----+
    |         |
    v         |
+-------+     |
|Cache? |-----+ (cache hit: return)
+-------+     |
    | miss    |
    v         |
+-------------------+
|Generate Embedding |
|(engine.py:205-214)|
+--------+----------+
         |
         v
+-------------------+
|Storage.get_*()    |
|with embedding     |
+--------+----------+
         |
         v
+-------------------+
|MemoryScorer       |
|.score_*()         |
|(scoring.py:75-259)|
+--------+----------+
         |
         v
+-------------------+
|Apply Threshold    |
|Extract top_k      |
+--------+----------+
         |
         v
+-------------------+
|Build MemorySlice  |
|Cache Result       |
+--------+----------+
         |
         v
    MemorySlice
```

### Extraction Flow (Auto-Learning)

```
Conversation Messages
        |
        v
+----------------------+
| AutoLearner          |
| .learn_from_         |
|  conversation()      |
| (auto_learner.py:69) |
+----------+-----------+
           |
           v
+----------------------+
| FactExtractor        |
| .extract()           |
| (extractor.py)       |
+----------+-----------+
           |
           v
+----------------------+
| Validate against     |
| - Confidence threshold|
| - Agent scope        |
| - Existing memories  |
+----------+-----------+
           |
      +----+----+
      |         |
   valid    rejected
      |         |
      v         v
+----------+ (logged)
|Commit to |
|ALMA via  |
|learning  |
|protocol  |
+----------+
```

---

## Integration Points

### MCP Server Integration (`alma/mcp/server.py`)

Exposes ALMA to MCP-compatible clients (Claude Code).

**Protocol Support:**
- stdio mode (line 438-490): Primary mode for Claude Code
- HTTP mode (line 491-534): For remote access

**Available Tools:**
| Tool | Description | Required Args |
|------|-------------|---------------|
| `alma_retrieve` | Get memories for task | task, agent |
| `alma_learn` | Record task outcome | agent, task, outcome, strategy_used |
| `alma_add_preference` | Add user preference | user_id, category, preference |
| `alma_add_knowledge` | Add domain knowledge | agent, domain, fact |
| `alma_forget` | Prune stale memories | (optional) agent, older_than_days |
| `alma_stats` | Get memory statistics | (optional) agent |
| `alma_health` | Health check | none |

**Request Handling** (`server.py:240-276`):
```python
async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
    method = request.get("method", "")
    if method == "tools/call":
        return await self._handle_tool_call(request_id, params)
    # ... other handlers
```

### Embedding Providers (`alma/retrieval/embeddings.py`)

**LocalEmbedder** (line 34-86):
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Dependency: `sentence-transformers`
- Latency: ~10ms per embedding

**AzureEmbedder** (line 88-167):
- Model: `text-embedding-3-small` (1536 dimensions)
- Dependency: `openai` package
- Requires: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`

**MockEmbedder** (line 170-202):
- Deterministic hash-based embeddings
- For testing only

### Graph Memory (`alma/graph/`)

**Entities and Relationships:**
```python
# graph/store.py:17-36
@dataclass
class Entity:
    id: str
    name: str
    entity_type: str  # person, organization, tool, concept
    properties: Dict[str, Any]

@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation_type: str  # WORKS_AT, USES, KNOWS, etc.
    confidence: float
```

**Backends:**
- `Neo4jGraphStore` (line 116-402): Production graph database
- `InMemoryGraphStore` (line 405-569): Testing/development

**Entity Extraction** (`graph/extraction.py:24-194`):
- LLM-powered extraction via OpenAI/Anthropic
- Extracts entities and relationships from text
- JSON-structured output parsing

---

## Configuration and Deployment

### Configuration File Structure

```yaml
# .alma/config.yaml
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

  # Azure configuration (if storage: azure)
  azure:
    endpoint: ${AZURE_COSMOS_ENDPOINT}
    key: ${KEYVAULT:cosmos-db-key}
    database: alma-memory
    openai_endpoint: ${AZURE_OPENAI_ENDPOINT}
    openai_key: ${AZURE_OPENAI_KEY}

  # PostgreSQL configuration (if storage: postgres)
  postgres:
    host: ${PGHOST}
    port: 5432
    database: alma_memory
    user: ${PGUSER}
    password: ${PGPASSWORD}
    ssl_mode: require
```

### Environment Variables

| Variable | Purpose | Required For |
|----------|---------|--------------|
| `AZURE_COSMOS_ENDPOINT` | Cosmos DB endpoint | Azure storage |
| `AZURE_COSMOS_KEY` | Cosmos DB key | Azure storage |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Azure embeddings |
| `AZURE_OPENAI_KEY` | Azure OpenAI key | Azure embeddings |
| `PGHOST`, `PGUSER`, `PGPASSWORD` | PostgreSQL credentials | Postgres storage |

### Deployment Options

**1. Local Development:**
```bash
pip install alma-memory[local]
# Uses SQLite + FAISS, local embeddings
```

**2. Production PostgreSQL:**
```bash
pip install alma-memory[postgres]
# Requires PostgreSQL 14+ with pgvector extension
```

**3. Azure Production:**
```bash
pip install alma-memory[azure]
# Uses Cosmos DB + Azure OpenAI
```

**4. MCP Server:**
```bash
pip install alma-memory[mcp]
python -m alma.mcp --config .alma/config.yaml
```

---

## Technical Debt Inventory

### Critical Issues

1. **Missing Update Method in Azure Backend** (`azure_cosmos.py`)
   - `update_heuristic_confidence()` and `update_knowledge_confidence()` are defined in base but not implemented
   - Would cause `AttributeError` if called
   - **Impact:** High - breaks confidence updates for Azure deployments

2. **eval() Usage in Neo4j Store** (`graph/store.py:224, 262`)
   ```python
   properties=eval(r["properties"]) if r["properties"] else {}
   ```
   - Security vulnerability - arbitrary code execution
   - Should use `json.loads()` or `ast.literal_eval()`
   - **Impact:** Critical - security risk

3. **Inconsistent Memory Type Names in Delete** (`sqlite_local.py:942-1011`)
   - Delete methods use singular form (`heuristic`, `outcome`) in embeddings table
   - Add/get methods use plural form (`heuristics`, `outcomes`)
   - May cause orphaned embeddings or failed deletions
   - **Impact:** Medium - data integrity

### Moderate Issues

4. **Missing Neptune Support** (`graph/store.py:590`)
   ```python
   elif provider == "neptune":
       raise NotImplementedError("Neptune support coming soon")
   ```
   - Advertised but not implemented
   - **Impact:** Low - documented as TODO

5. **Strategy Similarity Detection is Basic** (`learning/protocols.py:313-319`)
   ```python
   def _strategies_similar(self, s1: str, s2: str) -> bool:
       words1 = set(s1.lower().split())
       words2 = set(s2.lower().split())
       overlap = len(words1 & words2)
       return overlap >= min(3, len(words1) // 2)
   ```
   - Simple word overlap, no semantic similarity
   - Could group unrelated strategies or miss similar ones
   - **Impact:** Medium - learning quality

6. **No Embedding Dimension Validation**
   - Embedding dimension in config not validated against provider's actual dimension
   - Mismatch would cause vector search failures
   - **Impact:** Medium - runtime errors

7. **Cache Key Collision Possibility** (`cache.py:238-255`)
   - SHA256 truncated to 32 chars
   - Low probability but possible collision
   - **Impact:** Low - rare data integrity issue

### Minor Issues

8. **Hardcoded Token Estimation** (`types.py:202`)
   ```python
   # Basic token estimation (rough: 1 token ~ 4 chars)
   if len(result) > max_tokens * 4:
   ```
   - Inaccurate for non-ASCII text
   - **Impact:** Low - prompt truncation issues

9. **Missing datetime.utcnow() Deprecation Fix** (`types.py:88, 106`)
   - Uses deprecated `datetime.utcnow()` (removed in Python 3.12+)
   - Should use `datetime.now(timezone.utc)`
   - **Impact:** Low - future compatibility

10. **Inconsistent Logging Levels**
    - Mix of `logger.info()` and `logger.debug()` for similar operations
    - No structured logging (JSON) support
    - **Impact:** Low - observability

---

## Architectural Recommendations

### Short-Term (1-2 Sprints)

1. **Fix Critical Security Issue**
   - Replace `eval()` in Neo4j store with `json.loads()` or `ast.literal_eval()`
   - Add input validation for graph properties

2. **Implement Missing Azure Methods**
   - Add `update_heuristic_confidence()` and `update_knowledge_confidence()`
   - Add unit tests for Azure backend

3. **Fix Embedding Type Names**
   - Standardize on plural names across all operations
   - Add migration script for existing data

### Medium-Term (1-2 Quarters)

4. **Improve Strategy Similarity**
   - Use embedding similarity instead of word overlap
   - Configure similarity threshold

5. **Add Embedding Dimension Validation**
   - Validate on startup
   - Provide clear error messages

6. **Implement Neptune Support**
   - Or remove from documentation if not planned

7. **Add Structured Logging**
   - JSON format for production
   - Correlation IDs for request tracing

### Long-Term (Strategic)

8. **Consider Event Sourcing**
   - Store all operations as events
   - Enable time-travel debugging
   - Support event replay for migrations

9. **Add Multi-Tenancy**
   - Proper tenant isolation
   - Tenant-specific configuration
   - Resource quotas

10. **Implement Conflict Resolution**
    - Handle concurrent writes to same memory
    - Optimistic locking for updates

---

## Appendix: File Structure

```
alma/
├── __init__.py                    # Package exports (150 lines)
├── core.py                        # Main ALMA class (326 lines)
├── types.py                       # Memory type definitions (217 lines)
│
├── storage/
│   ├── __init__.py
│   ├── base.py                    # Abstract StorageBackend (373 lines)
│   ├── sqlite_local.py            # SQLite + FAISS (1014 lines)
│   ├── postgresql.py              # PostgreSQL + pgvector (1078 lines)
│   ├── azure_cosmos.py            # Azure Cosmos DB (979 lines)
│   └── file_based.py              # JSON file storage (584 lines)
│
├── retrieval/
│   ├── __init__.py
│   ├── engine.py                  # RetrievalEngine (288 lines)
│   ├── embeddings.py              # Embedding providers (203 lines)
│   ├── scoring.py                 # Memory scoring (335 lines)
│   └── cache.py                   # Cache backends (1063 lines)
│
├── learning/
│   ├── __init__.py
│   ├── protocols.py               # Learning protocol (327 lines)
│   ├── forgetting.py              # Memory pruning
│   ├── heuristic_extractor.py     # Heuristic extraction
│   └── validation.py              # Validation logic
│
├── graph/
│   ├── __init__.py
│   ├── store.py                   # Graph storage (593 lines)
│   └── extraction.py              # Entity extraction (195 lines)
│
├── extraction/
│   ├── __init__.py
│   ├── extractor.py               # Fact extraction
│   └── auto_learner.py            # Auto-learning (260 lines)
│
├── mcp/
│   ├── __init__.py
│   ├── __main__.py                # CLI entry point
│   ├── server.py                  # MCP server (534 lines)
│   ├── tools.py                   # MCP tool implementations
│   └── resources.py               # MCP resources
│
├── harness/
│   ├── __init__.py
│   ├── base.py                    # Harness pattern
│   └── domains.py                 # Domain configurations
│
├── session/
│   ├── __init__.py
│   ├── manager.py                 # Session management
│   └── types.py                   # Session types
│
├── progress/
│   ├── __init__.py
│   ├── tracker.py                 # Progress tracking
│   └── types.py                   # Work item types
│
├── domains/
│   ├── __init__.py
│   ├── factory.py                 # Domain memory factory
│   ├── schemas.py                 # Pre-built schemas
│   └── types.py                   # Domain types
│
├── confidence/
│   ├── __init__.py
│   ├── engine.py                  # Confidence engine
│   └── types.py                   # Signal types
│
├── initializer/
│   ├── __init__.py
│   ├── initializer.py             # Session initializer
│   └── types.py                   # Initialization types
│
├── config/
│   ├── __init__.py
│   └── loader.py                  # Config loading
│
└── integration/
    ├── __init__.py
    ├── claude_agents.py           # Claude integration
    ├── helena.py                  # Helena agent config
    └── victor.py                  # Victor agent config

tests/
├── test_core.py
├── test_storage/
├── test_retrieval/
├── test_learning/
└── ...

docs/
├── architecture/
│   └── system-architecture.md     # This document
└── ...
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.4.0 | 2026-01-28 | Aria (Architect Agent) | Initial brownfield documentation |

---

*This document was generated through analysis of the ALMA-memory codebase at version 0.4.0. It reflects the actual implementation, not aspirational design.*
