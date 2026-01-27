# ALMA - Agent Learning Memory Architecture

[![PyPI version](https://badge.fury.io/py/alma-memory.svg)](https://pypi.org/project/alma-memory/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Persistent memory for AI agents that learn and improve over time - without model weight updates.

## What is ALMA?

ALMA is a **memory framework** that makes AI agents appear to "learn" by:
1. **Storing** outcomes, strategies, and knowledge from past tasks
2. **Retrieving** relevant memories before each new task
3. **Injecting** that knowledge into prompts
4. **Learning** from new outcomes to improve future performance

**No fine-tuning. No model changes. Just smarter prompts.**

```
┌─────────────────────────────────────────────────────────────────┐
│  BEFORE TASK: Retrieve relevant memories                        │
│  ├── "Last time you tested forms, incremental validation worked"│
│  ├── "User prefers verbose output"                              │
│  └── "Don't use sleep() - causes flaky tests"                   │
├─────────────────────────────────────────────────────────────────┤
│  DURING TASK: Agent executes with injected knowledge            │
├─────────────────────────────────────────────────────────────────┤
│  AFTER TASK: Learn from outcome                                 │
│  └── Success? → New heuristic. Failure? → Anti-pattern.         │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install alma-memory
```

## Quick Start

### 1. Set Up Configuration

Copy the example config to your project:

```bash
# Copy from project root
cp config.yaml.example .alma/config.yaml

# Or copy from templates directory
cp .alma/templates/config.yaml.template .alma/config.yaml
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

## Features

### Core Memory System

| Feature | Description |
|---------|-------------|
| **5 Memory Types** | Heuristics, Outcomes, Preferences, Domain Knowledge, Anti-patterns |
| **Semantic Search** | Vector similarity for relevant memory retrieval |
| **Scoped Learning** | Agents only learn from their domain (Helena can't learn backend) |
| **LLM Fact Extraction** | Automatic learning from conversations (NEW) |
| **Graph Memory** | Entity relationships via Neo4j (NEW) |
| **Confidence Decay** | Old memories fade, recent ones stay strong |
| **Forgetting** | Automatic cleanup of low-value memories |

### Storage Backends

| Backend | Use Case | Vector Search |
|---------|----------|---------------|
| **SQLite + FAISS** | Local development | Yes |
| **Azure Cosmos DB** | Production | Yes (native) |
| **File-based** | Testing/Simple use | No |

### Domain Memory Factory (NEW in v0.4.0)

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

### Progress Tracking (NEW in v0.4.0)

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

# Get progress summary
summary = tracker.get_progress_summary()
print(f"Done: {summary.done}/{summary.total} ({summary.completion_percentage}%)")
```

### Session Handoff (NEW in v0.4.0)

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

# Track decisions and blockers during work
manager.update_session("Helena", context.session_id,
    decision="Using OAuth mock for testing",
    blocker="Staging API is down",
)

# End session with handoff for next time
manager.create_handoff("Helena", context.session_id,
    last_action="completed_oauth_tests",
    last_outcome="success",
    next_steps=["Test refresh tokens", "Add error cases"],
)
```

### LLM-Powered Fact Extraction (NEW in v0.5.0)

Automatically extract and learn from conversations - no manual `learn()` calls needed:

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
print(f"Committed {results['committed_count']} to memory")
```

Supports OpenAI, Anthropic, or rule-based extraction (free, offline).

### Graph Memory (NEW in v0.5.0)

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

# Query relationships
result = graph.traverse("alice-id", max_hops=2)
print(f"Found {len(result.entities)} related entities")
```

Supports Neo4j and in-memory (for testing).

### MCP Server Integration

Expose ALMA to Claude Code or any MCP-compatible client:

```bash
# Start MCP server
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
- `alma_retrieve` - Get memories for a task
- `alma_learn` - Record task outcome
- `alma_add_preference` - Add user preference
- `alma_add_knowledge` - Add domain knowledge
- `alma_forget` - Prune stale memories
- `alma_stats` - Get memory statistics
- `alma_health` - Health check

## Memory Types

| Type | What It Stores | Example |
|------|----------------|---------|
| **Heuristic** | Learned strategies | "For forms with >5 fields, test validation incrementally" |
| **Outcome** | Task results | "Login test succeeded using JWT token strategy" |
| **Preference** | User constraints | "User prefers verbose test output" |
| **Domain Knowledge** | Accumulated facts | "Login uses OAuth 2.0 with 24h token expiry" |
| **Anti-pattern** | What NOT to do | "Don't use sleep() for async waits - causes flaky tests" |

## The Harness Pattern

ALMA implements a generalized harness pattern for any tool-using agent:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SETTING        Fixed environment: tools, constraints        │
├─────────────────────────────────────────────────────────────────┤
│  2. CONTEXT        Ephemeral per-run inputs: task, user         │
├─────────────────────────────────────────────────────────────────┤
│  3. AGENT          The executor with scoped intelligence        │
├─────────────────────────────────────────────────────────────────┤
│  4. MEMORY SCHEMA  Domain-specific learning structure           │
└─────────────────────────────────────────────────────────────────┘
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

## Configuration

Create `.alma/config.yaml`:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite  # or "azure" for production
  embedding_provider: local  # or "azure" for production

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
      min_occurrences_for_heuristic: 3
```

### Embedding Model Configuration

ALMA uses embeddings for semantic memory retrieval. You can choose between local (free, offline) or cloud-based providers.

#### Provider Comparison

| Provider | Model | Dimensions | Cost | Latency | Best For |
|----------|-------|------------|------|---------|----------|
| **local** | all-MiniLM-L6-v2 | 384 | Free | ~10ms | Development, offline use, cost-sensitive |
| **local** | all-mpnet-base-v2 | 768 | Free | ~25ms | Better quality, still offline |
| **azure** | text-embedding-3-small | 1536 | ~$0.02/1M tokens | ~50ms | Production, high accuracy |
| **azure** | text-embedding-3-large | 3072 | ~$0.13/1M tokens | ~80ms | Maximum quality, enterprise |
| **mock** | (hash-based) | 384 | Free | <1ms | Testing only |

#### Local Embeddings (Default)

No API keys required. Uses [sentence-transformers](https://www.sbert.net/).

```yaml
# .alma/config.yaml
alma:
  embedding_provider: local
  # Default model: all-MiniLM-L6-v2
  # To use a different model, set in code:
  # LocalEmbedder(model_name="all-mpnet-base-v2")
```

Install dependencies:
```bash
pip install sentence-transformers
```

#### Azure OpenAI Embeddings (Production)

For production deployments with higher accuracy.

```yaml
# .alma/config.yaml
alma:
  embedding_provider: azure
  azure:
    openai_endpoint: ${AZURE_OPENAI_ENDPOINT}
    openai_key: ${AZURE_OPENAI_KEY}
    openai_deployment: text-embedding-3-small
```

Set environment variables:
```bash
# .env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

Or use Azure Key Vault for secrets:
```yaml
alma:
  azure:
    openai_key: ${KEYVAULT:alma-openai-key}
```

#### Mock Embeddings (Testing)

Deterministic hash-based embeddings for unit tests.

```yaml
alma:
  embedding_provider: mock
```

### Storage Backend Configuration

| Backend | Config Value | Vector Search | Use Case |
|---------|--------------|---------------|----------|
| SQLite + FAISS | `sqlite` | Yes (FAISS) | Local development |
| Azure Cosmos DB | `azure` | Yes (native) | Production |
| File-based | `file` | No | Simple testing |

```yaml
alma:
  storage: sqlite  # Options: sqlite, azure, file
  storage_dir: .alma  # Where to store local databases
  db_name: alma.db  # Database filename
  embedding_dim: 384  # Must match your embedding model
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ALMA v0.4.0                              │
├─────────────────────────────────────────────────────────────────┤
│  HARNESS LAYER                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Setting  │  │ Context  │  │  Agent   │  │MemorySchema  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  NEW IN v0.4.0                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────┐   │
│  │ Progress     │  │ Session        │  │ Domain Memory     │   │
│  │ Tracking     │  │ Handoff        │  │ Factory           │   │
│  └──────────────┘  └────────────────┘  └───────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  CORE LAYER                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Retrieval  │  │  Learning  │  │  Caching   │  │Forgetting│  │
│  │  Engine    │  │  Protocol  │  │   Layer    │  │Mechanism │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  STORAGE LAYER                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ SQLite + FAISS │  │  Azure Cosmos  │  │   File-based   │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  INTEGRATION LAYER                                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    MCP Server                           │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Abstractions | ✅ Done |
| 2 | Local Storage (SQLite + FAISS) | ✅ Done |
| 3 | Retrieval Engine | ✅ Done |
| 4 | Learning Protocols | ✅ Done |
| 5 | Agent Integration | ✅ Done |
| 6 | Azure Cosmos DB | ✅ Done |
| 7 | Cache Layer | ✅ Done |
| 8 | Forgetting Mechanism | ✅ Done |
| 9 | MCP Server + Testing Suite | ✅ Done |
| 10 | Progress, Sessions, Domains | ✅ Done |
| 11 | Initializer Agent Pattern | ✅ Done |
| 12 | Forward-Looking Confidence | ✅ Done |

## API Reference

### Core

```python
# Initialize
alma = ALMA.from_config(".alma/config.yaml")

# Retrieve memories
memories = alma.retrieve(task, agent, top_k=5)

# Learn from outcome
alma.learn(agent, task, outcome, strategy_used, feedback=None)

# Add knowledge directly
alma.add_preference(user_id, category, preference)
alma.add_domain_knowledge(agent, domain, fact)

# Memory hygiene
alma.forget(agent, older_than_days=90, min_confidence=0.3)
```

### Progress Tracking

```python
from alma.progress import ProgressTracker, WorkItem

tracker = ProgressTracker("project-id")
item = tracker.create_work_item(title, description, priority=50)
tracker.update_status(item.id, "in_progress")  # or "done", "blocked"
next_item = tracker.get_next_item(strategy="priority")
summary = tracker.get_progress_summary(agent="helena")
```

### Session Management

```python
from alma.session import SessionManager

manager = SessionManager("project-id")
context = manager.start_session(agent, goal)
manager.update_session(agent, session_id, decision=..., blocker=...)
manager.create_handoff(agent, session_id, last_action, outcome, next_steps)
reload_str = manager.get_quick_reload(agent)
```

### Domain Factory

```python
from alma.domains import DomainMemoryFactory, get_coding_schema

factory = DomainMemoryFactory()
schema = get_coding_schema()  # or research, sales, general
alma = factory.create_alma(schema, "project-id")
```

### Session Initializer (NEW in v0.4.0)

```python
from alma.initializer import SessionInitializer

initializer = SessionInitializer(alma)

# Full initialization before starting work
result = initializer.initialize(
    project_id="my-project",
    agent="Helena",
    user_prompt="Test the login form validation",
    project_path="/path/to/project",
)

# Inject into agent prompt
prompt = f"""
{result.to_prompt()}

Now proceed with the first work item.
"""
```

### Confidence Engine (NEW in v0.4.0)

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
# → Confidence: 78%
# → Recommendation: yes

# Rank multiple strategies
rankings = engine.rank_strategies(
    strategies=["Strategy A", "Strategy B", "Strategy C"],
    context="Current task",
    agent="Helena",
)
```

## Troubleshooting

### Common Issues

**ImportError: sentence-transformers is required**
```bash
pip install alma-memory[local]
# or
pip install sentence-transformers
```

**pgvector extension not found**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Azure Cosmos DB connection failed**
- Verify `AZURE_COSMOS_ENDPOINT` and `AZURE_COSMOS_KEY` are set
- Check that vector search is enabled on your Cosmos DB account

**Embeddings dimension mismatch**
- Ensure `embedding_dim` in config matches your embedding provider
- Local embeddings: 384 dimensions
- Azure OpenAI text-embedding-3-small: 1536 dimensions

**Cache not invalidating**
- Call `alma.forget()` to clear stale memories
- Check that cache TTL hasn't been set too high

### Debug Logging

Enable debug logging to troubleshoot issues:
```python
import logging
logging.getLogger("alma").setLevel(logging.DEBUG)
```

## License

MIT

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built for AI agents that get better with every task.**
