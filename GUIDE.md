# ALMA Setup Guide

Complete setup instructions for ALMA — from zero to persistent AI memory. Written for all experience levels.

**Already installed?** Jump to your database:
- [Local SQLite (zero infrastructure)](#path-1-local-sqlite--faiss-zero-infrastructure)
- [PostgreSQL + pgvector](#path-2-postgresql--pgvector-production)
- [Supabase (free cloud)](#path-3-supabase-free-cloud-database)
- [Qdrant](#path-4-qdrant)
- [Pinecone](#path-5-pinecone)
- [Chroma](#path-6-chroma)
- [Azure Cosmos DB](#path-7-azure-cosmos-db)

---

## Prerequisites

You need **Python 3.10 or newer**. Check your version:

```bash
python --version
# or
python3 --version
```

If you don't have Python 3.10+, download it from [python.org](https://www.python.org/downloads/).

**That's the only hard requirement.** Everything else depends on which database path you choose below.

---

## Step 1: Install ALMA

```bash
pip install alma-memory
```

If you're using a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows

pip install alma-memory
```

---

## Step 2: Choose Your Database

ALMA stores memories in a database. You control which one. Here are your options, from simplest to most powerful:

| Path | What You Need | Cost | Best For |
|---|---|---|---|
| **SQLite + FAISS** | Nothing (runs on your machine) | $0.00 | Getting started, local development, offline use |
| **Supabase** | Free account at supabase.com | $0.00 (free tier) | Cloud-hosted, personal brain, sharing across machines |
| **PostgreSQL** | A PostgreSQL server with pgvector | Varies | Production, teams, high availability |
| **Qdrant** | Qdrant server (local Docker or cloud) | Free (local) / Varies (cloud) | Dedicated vector search |
| **Pinecone** | Pinecone account | Free tier available | Serverless, fully managed |
| **Chroma** | Nothing (runs locally) | $0.00 | Lightweight local alternative |
| **Azure Cosmos DB** | Azure account | Varies | Enterprise, Azure-native |

---

## Path 1: Local SQLite + FAISS (Zero Infrastructure)

**Best for:** Getting started, development, offline work.
**You need:** Nothing beyond Python. No servers, no accounts, no configuration.

### Install

```bash
pip install alma-memory[local]
```

This installs SQLite support (built into Python), FAISS for vector search, and a local embedding model.

### Configure

Create a folder and config file:

```bash
mkdir -p .alma
```

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite
  embedding_provider: local
  storage_dir: .alma
  db_name: alma.db
  embedding_dim: 384
```

### That's it

Tables are created automatically on first run. No SQL, no setup, no external services. Your memory database is a single file at `.alma/alma.db`.

### Verify it works

```python
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")

# Store a test memory
alma.learn(
    agent="test-agent",
    task="Setup verification",
    outcome="success",
    strategy_used="Followed the guide",
)

# Retrieve it
memories = alma.retrieve(task="setup", agent="test-agent", top_k=1)
print(f"Memories found: {len(memories.outcomes)}")
# -> Memories found: 1
```

---

## Path 2: PostgreSQL + pgvector (Production)

**Best for:** Production deployments, teams, high availability.
**You need:** A PostgreSQL 14+ server with the pgvector extension.

Works with any PostgreSQL provider: Supabase, Neon, AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL, or self-hosted.

### Install

```bash
pip install alma-memory[postgres]
```

### Set up your database

**Step 1:** Connect to your PostgreSQL server and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

> **Note:** Most managed PostgreSQL providers (Supabase, Neon, AWS RDS) have pgvector pre-installed. You just need to enable it.

**Step 2 (optional):** Create the ALMA tables manually. ALMA creates tables automatically on first connection, but for production you may want explicit control:

<details>
<summary>Click to expand: Core memory tables SQL</summary>

```sql
CREATE TABLE alma_heuristics (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    occurrence_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_validated TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);
CREATE INDEX idx_heuristics_project_agent ON alma_heuristics(project_id, agent);
CREATE INDEX idx_heuristics_confidence ON alma_heuristics(project_id, confidence DESC);

CREATE TABLE alma_outcomes (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_type TEXT,
    task_description TEXT NOT NULL,
    success BOOLEAN DEFAULT FALSE,
    strategy_used TEXT,
    duration_ms INTEGER,
    error_message TEXT,
    user_feedback TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);
CREATE INDEX idx_outcomes_project_agent ON alma_outcomes(project_id, agent);
CREATE INDEX idx_outcomes_task_type ON alma_outcomes(project_id, agent, task_type);
CREATE INDEX idx_outcomes_timestamp ON alma_outcomes(project_id, timestamp DESC);

CREATE TABLE alma_preferences (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT,
    preference TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);
CREATE INDEX idx_preferences_user ON alma_preferences(user_id);

CREATE TABLE alma_domain_knowledge (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    domain TEXT,
    fact TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    last_verified TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);
CREATE INDEX idx_domain_knowledge_project_agent ON alma_domain_knowledge(project_id, agent);
CREATE INDEX idx_domain_knowledge_confidence ON alma_domain_knowledge(project_id, confidence DESC);

CREATE TABLE alma_anti_patterns (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    project_id TEXT NOT NULL,
    pattern TEXT NOT NULL,
    why_bad TEXT,
    better_alternative TEXT,
    occurrence_count INTEGER DEFAULT 1,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(384)
);
CREATE INDEX idx_anti_patterns_project_agent ON alma_anti_patterns(project_id, agent);
```

</details>

<details>
<summary>Click to expand: Workflow tables SQL (optional — for checkpoint/resume features)</summary>

```sql
CREATE TABLE alma_checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    state_json JSONB NOT NULL,
    state_hash TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    branch_id TEXT,
    parent_checkpoint_id TEXT REFERENCES alma_checkpoints(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT uk_checkpoint_run_seq UNIQUE (run_id, sequence_number)
);
CREATE INDEX idx_checkpoints_run_seq ON alma_checkpoints(run_id, sequence_number DESC);
CREATE INDEX idx_checkpoints_run_branch ON alma_checkpoints(run_id, branch_id) WHERE branch_id IS NOT NULL;

CREATE TABLE alma_workflow_outcomes (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workflow_id TEXT NOT NULL,
    workflow_version TEXT DEFAULT '1.0',
    run_id TEXT NOT NULL UNIQUE,
    success BOOLEAN NOT NULL,
    duration_ms INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    nodes_succeeded INTEGER NOT NULL DEFAULT 0,
    nodes_failed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    artifacts_json JSONB,
    learnings_extracted INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding VECTOR(384),
    metadata JSONB,
    CONSTRAINT chk_nodes_count CHECK (nodes_succeeded + nodes_failed <= node_count)
);
CREATE INDEX idx_wo_tenant ON alma_workflow_outcomes(tenant_id);
CREATE INDEX idx_wo_workflow ON alma_workflow_outcomes(tenant_id, workflow_id);
CREATE INDEX idx_wo_timestamp ON alma_workflow_outcomes(timestamp DESC);

CREATE TABLE alma_artifact_links (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT chk_size_positive CHECK (size_bytes > 0)
);
CREATE INDEX idx_artifact_memory ON alma_artifact_links(memory_id, memory_type);
```

</details>

### Configure

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: postgres
  embedding_dim: 384

  postgres:
    host: localhost
    port: 5432
    database: alma
    user: alma_user
    password: ${POSTGRES_PASSWORD}
    vector_index_type: hnsw
```

Set your password as an environment variable:

```bash
export POSTGRES_PASSWORD="your-password-here"
```

> **Security:** Never put passwords directly in config files. Use `${ENV_VAR}` syntax — ALMA expands environment variables automatically.

---

## Path 3: Supabase (Free Cloud Database)

**Best for:** Getting started with cloud-hosted memory. Free tier is generous.
**You need:** A free Supabase account.

### Step 1: Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and create a free account
2. Click "New Project"
3. Choose a name and set a database password
4. **Write down your password** — you'll need it in Step 4

### Step 2: Find your connection details

In your Supabase dashboard, go to **Settings > Database**. You'll need:
- **Host:** `db.<your-project-ref>.supabase.co`
- **Port:** `6543` (connection pooler — recommended) or `5432` (direct)
- **Database:** `postgres`
- **User:** `postgres`

### Step 3: Enable pgvector

Go to the **SQL Editor** in your Supabase dashboard and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

ALMA creates tables automatically on first connection. You don't need to run any other SQL.

### Step 4: Install and configure

```bash
pip install alma-memory[postgres]
```

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: postgres
  embedding_dim: 384

  postgres:
    host: db.<your-project-ref>.supabase.co
    port: 6543
    database: postgres
    user: postgres
    password: ${SUPABASE_DB_PASSWORD}
    vector_index_type: hnsw
```

Set your password:

```bash
export SUPABASE_DB_PASSWORD="your-password-from-step-1"
```

### Step 5: Verify

```python
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")
alma.learn(agent="test", task="verify setup", outcome="success", strategy_used="guide")
memories = alma.retrieve(task="verify", agent="test", top_k=1)
print(f"Cloud memory works: {len(memories.outcomes)} memories found")
```

Your AI now has persistent memory in the cloud. Free.

---

## Path 4: Qdrant

**Best for:** Dedicated vector search at scale.
**You need:** Qdrant running locally (Docker) or a Qdrant Cloud account.

### Install

```bash
pip install alma-memory[qdrant]
```

### Option A: Local Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Option B: Qdrant Cloud

1. Create an account at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Note your URL and API key

### Configure

```yaml
alma:
  project_id: "my-project"
  storage: qdrant
  embedding_dim: 384

  qdrant:
    url: http://localhost:6333          # Local
    # url: https://xxx.cloud.qdrant.io  # Cloud
    # api_key: ${QDRANT_API_KEY}        # Cloud only
    collection_prefix: alma
```

---

## Path 5: Pinecone

**Best for:** Fully managed, serverless vector search.
**You need:** A Pinecone account (free tier available).

### Install

```bash
pip install alma-memory[pinecone]
```

### Setup

1. Create an account at [pinecone.io](https://www.pinecone.io)
2. Create an index with dimension 384 (or matching your embedding model)
3. Note your API key and environment

### Configure

```yaml
alma:
  project_id: "my-project"
  storage: pinecone
  embedding_dim: 384

  pinecone:
    api_key: ${PINECONE_API_KEY}
    environment: us-east-1-aws
    index_name: alma-memories
```

```bash
export PINECONE_API_KEY="your-api-key"
```

---

## Path 6: Chroma

**Best for:** Lightweight local vector database, alternative to SQLite+FAISS.
**You need:** Nothing extra.

### Install

```bash
pip install alma-memory[chroma]
```

### Configure

```yaml
alma:
  project_id: "my-project"
  storage: chroma
  embedding_dim: 384

  chroma:
    persist_directory: .alma/chroma
    # Or for client-server mode:
    # host: localhost
    # port: 8000
```

---

## Path 7: Azure Cosmos DB

**Best for:** Enterprise, Azure-native deployments, compliance requirements.
**You need:** An Azure account with Cosmos DB.

### Install

```bash
pip install alma-memory[azure]
```

### Setup

1. In the Azure Portal, create a Cosmos DB account (NoSQL API)
2. Create a database named `alma` (or your preferred name)
3. Note your endpoint and key from the "Keys" section

### Configure

```yaml
alma:
  project_id: "my-project"
  storage: azure
  embedding_dim: 384

  azure:
    cosmos_endpoint: ${AZURE_COSMOS_ENDPOINT}
    cosmos_key: ${AZURE_COSMOS_KEY}
    cosmos_database: alma
    embedding_deployment: text-embedding-3-small   # If using Azure OpenAI
    azure_openai_endpoint: ${AZURE_OPENAI_ENDPOINT}
    azure_openai_key: ${AZURE_OPENAI_KEY}
```

```bash
export AZURE_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export AZURE_COSMOS_KEY="your-cosmos-key"
```

---

## Step 3: Configure Your Agents

Once your database is set up, define your agents in the same config file:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite       # or postgres, qdrant, etc.
  embedding_provider: local
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
        - database_queries
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

**Key concepts:**
- **`can_learn`** — Topics this agent is allowed to form memories about
- **`cannot_learn`** — Topics this agent must ignore (prevents knowledge contamination)
- **`share_with`** — Other agents that can access this agent's memories
- **`inherit_from`** — This agent inherits memories from the listed agents
- **`min_occurrences_for_heuristic`** — How many times a pattern must appear before becoming a heuristic

---

## Embedding Providers

ALMA needs an embedding model to convert text into vectors for similarity search. Three options:

| Provider | Model | Dimensions | Cost | Best For |
|---|---|---|---|---|
| **local** | all-MiniLM-L6-v2 | 384 | Free | Development, offline, getting started |
| **azure** | text-embedding-3-small | 1536 | ~$0.02/1M tokens | Production, higher accuracy |
| **mock** | (hash-based) | 384 | Free | Testing only (no real semantics) |

### Local embeddings (default)

```bash
pip install alma-memory[local]
```

```yaml
embedding_provider: local
embedding_dim: 384
```

Downloads a small model (~90 MB) on first run. Works offline after that.

### Azure OpenAI embeddings

```yaml
embedding_provider: azure
embedding_dim: 1536

azure:
  azure_openai_endpoint: ${AZURE_OPENAI_ENDPOINT}
  azure_openai_key: ${AZURE_OPENAI_KEY}
  embedding_deployment: text-embedding-3-small
```

---

## MCP Server Setup

To use ALMA directly from Claude Code or Claude Desktop:

### Claude Code

Create `.mcp.json` in your project root:

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

Claude Code will automatically discover and use the 22 ALMA MCP tools.

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "alma-memory": {
      "command": "python",
      "args": ["-m", "alma.mcp", "--config", "/full/path/to/.alma/config.yaml"]
    }
  }
}
```

### HTTP Server (for remote access)

```bash
python -m alma.mcp --config .alma/config.yaml --transport http --port 8765
```

---

## Troubleshooting

### ImportError: sentence-transformers is required

You're using `embedding_provider: local` but didn't install the local extras:

```bash
pip install alma-memory[local]
```

### pgvector extension not found

Connect to your PostgreSQL database and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

On managed providers (Supabase, Neon), this is usually the only step needed. On self-hosted PostgreSQL, you may need to install pgvector first: [pgvector installation guide](https://github.com/pgvector/pgvector#installation).

### Connection refused (PostgreSQL)

1. Check that your PostgreSQL server is running
2. Verify the host, port, user, and password in your config
3. Make sure your IP is allowed in the server's firewall/security rules
4. For Supabase: use port `6543` (connection pooler) instead of `5432`

### Qdrant connection refused

Start Qdrant locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or check your Qdrant Cloud URL and API key.

### Pinecone index not found

1. Verify your index exists in the [Pinecone console](https://app.pinecone.io)
2. Check that `index_name` in your config matches exactly
3. Ensure `embedding_dim` matches the index dimension

### Embeddings dimension mismatch

Your `embedding_dim` config must match your embedding provider:
- **local** (all-MiniLM-L6-v2): `384`
- **azure** (text-embedding-3-small): `1536`

If you change embedding providers, you need to recreate your database tables (the vector column dimension must match).

### Memgraph connection refused

```bash
docker run -p 7687:7687 memgraph/memgraph-mage
```

### Kuzu database locked

Kuzu only supports one writer process at a time. Options:
- Use `read_only=True` for concurrent read access
- Ensure only one process writes at a time

### Debug logging

Enable detailed logging to diagnose issues:

```python
import logging
logging.getLogger("alma").setLevel(logging.DEBUG)
```

This shows every storage operation, embedding call, and retrieval step.

---

## Environment Variables

ALMA config files support `${ENV_VAR}` syntax for secrets. Common variables:

```bash
# PostgreSQL
export POSTGRES_PASSWORD="your-password"

# Supabase
export SUPABASE_DB_PASSWORD="your-password"

# Qdrant Cloud
export QDRANT_API_KEY="your-api-key"

# Pinecone
export PINECONE_API_KEY="your-api-key"

# Azure
export AZURE_COSMOS_ENDPOINT="https://your-account.documents.azure.com:443/"
export AZURE_COSMOS_KEY="your-cosmos-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your-openai-key"
```

You can also use Azure Key Vault references: `${KEYVAULT:secret-name}`.

**Never commit secrets to version control.** Add your `.env` file to `.gitignore`.

---

## Bootstrap From Existing Knowledge (v0.9.0+)

Already have chat exports or project files? ALMA can ingest and classify them into the 5 memory types automatically:

```python
from alma.ingestion import ingest_directory, ingest_conversations

# Ingest project files — auto-classifies into memory types
result = ingest_directory("/path/to/project", agent="dev", project_id="myapp")

# Ingest chat exports (6 formats: Claude Code, ChatGPT, Slack, Codex, Claude.ai, plain text)
result = ingest_conversations("/path/to/chats", agent="dev", project_id="myapp")

print(f"Domain knowledge: {len(result.domain_knowledge)}")
print(f"User preferences: {len(result.user_preferences)}")
print(f"Anti-patterns: {len(result.anti_patterns)}")
print(f"Outcomes: {len(result.outcomes)}")
```

This is not RAG — ALMA reads, classifies, and structures content into typed memories that improve over time.

---

## Token-Efficient Context Loading (v0.9.0+)

Use the 4-layer MemoryStack to load only what you need:

```python
from alma.context import MemoryStack

stack = MemoryStack(alma)

# Session start: load identity + essential memories (~900 tokens)
context = stack.wake_up()

# During conversation: recall specific topics
result = stack.recall("authentication patterns")

# Format for prompt injection with token budget
prompt = stack.to_prompt(max_tokens=2000)
```

---

## Retrieval Feedback Loop (v1.0+)

ALMA tracks which retrieved memories your agents actually use versus ignore, then adjusts future retrieval scores so useful memories rank higher over time. This is a closed-loop system: retrieve, observe, learn, improve.

### How it works

Every time an agent retrieves memories, some get used and some get ignored. The feedback loop records these signals and blends them into future retrieval scores. Memories that agents consistently use get a score boost; memories that get ignored drift down.

### Setup

Create a `FeedbackTracker` and `FeedbackAwareScorer`:

```python
from alma.retrieval.feedback import FeedbackTracker, FeedbackAwareScorer

tracker = FeedbackTracker(storage=alma.storage)
scorer = FeedbackAwareScorer(feedback_tracker=tracker, feedback_weight=0.15)
```

### Recording usage

After retrieval, record which memories the agent actually used. This is the primary feedback signal:

```python
from alma.types import MemoryType

# Agent retrieved 3 memories, used 2 of them
alma.record_usage(
    retrieved_memory_ids=["m1", "m2", "m3"],
    used_memory_ids=["m1", "m3"],
    memory_type=MemoryType.HEURISTIC,
    agent="dev-agent",
)
```

Each retrieved memory is automatically marked as `USED` or `IGNORED` based on whether its ID appears in `used_memory_ids`.

### Explicit feedback

For direct thumbs up/down signals (e.g., from user ratings or agent self-evaluation):

```python
from alma.types import FeedbackSignal

alma.record_feedback(
    memory_id="m1",
    memory_type=MemoryType.HEURISTIC,
    signal=FeedbackSignal.THUMBS_UP,
    agent="dev-agent",
)
```

Available signals: `USED`, `IGNORED`, `THUMBS_UP`, `THUMBS_DOWN`.

### How scoring works

The `FeedbackAwareScorer` blends feedback into retrieval scores using this formula:

```
final_score = (1 - weight) * base_score + weight * normalized_feedback
```

Where `normalized_feedback` maps the accumulated feedback score from `[-1, 1]` to `[0, 1]`. A memory with all positive feedback gets `normalized_feedback = 1.0`; all negative gets `0.0`; no feedback leaves the score unchanged.

The default weight is `0.15` (15% influence). The remaining 85% comes from the base retrieval score (similarity, recency, confidence, success rate).

### Integration with RetrievalEngine

Pass the scorer to `RetrievalEngine` to enable automatic feedback-based re-ranking on every retrieval:

```python
from alma.retrieval.engine import RetrievalEngine

engine = RetrievalEngine(
    storage=storage,
    embedder=embedder,
    feedback_scorer=scorer,  # Optional — enables feedback re-ranking
)
```

When `feedback_scorer` is provided, the engine applies feedback re-ranking to all memory types (heuristics, outcomes, domain knowledge, and anti-patterns) after the base scoring pass.

### Tuning `feedback_weight`

The weight controls how aggressively feedback influences retrieval:

| Weight | Behavior | When to use |
|---|---|---|
| `0.05` | Minimal influence, very stable | Early stages, small feedback datasets |
| `0.15` | Balanced (default) | Most use cases |
| `0.30` | Aggressive learning | Agents frequently retrieve irrelevant memories |
| `0.50+` | Feedback-dominant | Only when you have high-volume, high-quality feedback |

Start with the default `0.15`. Increase if agents consistently retrieve memories they never use. Decrease if you see useful memories dropping out of results too quickly.

---

## Next Steps

Once ALMA is running:

1. **Read the [README](README.md)** for feature overview and code examples
2. **Explore the [Documentation](https://alma-memory.pages.dev)** for the full API reference
3. **Run the benchmark** — verify ALMA works on your machine: `python -m benchmarks.longmemeval.runner --limit 20`
4. **Ingest existing knowledge** — feed ALMA from chat exports or project files
5. **Try the MCP tools** — connect ALMA to Claude Code and let Claude manage memories automatically
6. **Define your agents** — scope learning domains to keep memory clean and relevant
7. **Set up webhooks** — react to memory changes in real-time
8. **Read the [Benchmark Report](docs/benchmarks/BENCHMARK-REPORT.md)** — understand how ALMA achieves R@5=0.964

---

**Questions?** Email **dev@friendlyai.fi** or open an issue on [GitHub](https://github.com/RBKunnela/ALMA-memory/issues).
