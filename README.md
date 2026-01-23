# ALMA - Agent Learning Memory Architecture

> Persistent memory system for AI agents that learn and improve over time through structured memory layers - without model weight updates.

## Overview

ALMA enables AI agents to accumulate intelligence across runs by structuring memory into scoped layers:

- **Heuristics**: "When X happens, strategy Y works 80% of the time"
- **Outcomes**: Records of task success/failure with context
- **User Preferences**: Remembered constraints and communication styles
- **Domain Knowledge**: Accumulated facts within agent's scope
- **Anti-patterns**: What NOT to do - learned from failures

## Key Principle

```
Code exists ≠ Knowledge retained
Knowledge retained ≠ Knowledge scoped
Knowledge scoped ≠ Knowledge retrieved efficiently
```

ALMA solves all three:
1. **Retention**: Memories persist across sessions in Azure Cosmos DB
2. **Scoping**: Each agent only learns within defined boundaries
3. **Efficiency**: Semantic retrieval injects only relevant slices per-call

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PER-CALL CONTEXT                      │
│         (minimal - only relevant slices)                │
└─────────────────────────────────────────────────────────┘
                         ↑ retrieval (< 200ms)
┌─────────────────────────────────────────────────────────┐
│              ALMA RETRIEVAL ENGINE                      │
│    Semantic search + recency + success-weighting        │
└─────────────────────────────────────────────────────────┘
                         ↑
┌─────────────────────────────────────────────────────────┐
│              AZURE COSMOS DB (Vector Search)            │
│    Partitioned by project_id / agent_name               │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install alma-memory
# or from source
pip install git+https://github.com/RBKunnela/ALMA-memory.git
```

### Basic Usage

```python
from alma import ALMA, MemoryScope

# Initialize with project config
alma = ALMA.from_config(".alma/config.yaml")

# Retrieve relevant memories for a task
memories = alma.retrieve(
    task="Test the login form validation",
    agent="helena",
    top_k=5
)

# Learn from an outcome
alma.learn(
    agent="helena",
    task="Test login form",
    outcome="success",
    strategy_used="Tested empty fields, invalid email, valid submission",
    feedback="User confirmed tests were thorough"
)
```

### Project Configuration

Create `.alma/config.yaml` in your project:

```yaml
alma:
  project_id: "my-project"
  storage: azure  # or "local" for development

  agents:
    helena:
      can_learn:
        - testing_strategies
        - selector_patterns
        - ui_patterns
      cannot_learn:
        - backend_logic
      min_occurrences_for_heuristic: 3

    victor:
      can_learn:
        - api_testing
        - database_validation
      cannot_learn:
        - frontend_logic
      min_occurrences_for_heuristic: 3
```

## Storage Backends

| Backend | Use Case | Setup |
|---------|----------|-------|
| `azure` | Production | Cosmos DB + Key Vault |
| `sqlite` | Local dev | SQLite + FAISS vectors |
| `file` | Testing | JSON files |

## Documentation

- [PRD](docs/architecture/PRD.md) - Full product requirements
- [API Reference](docs/api/) - Coming soon
- [Integration Guide](docs/guides/) - Coming soon

## Status

**Current Phase**: 1 - Core Abstractions

See [PRD](docs/architecture/PRD.md) for full implementation roadmap.

## License

MIT
