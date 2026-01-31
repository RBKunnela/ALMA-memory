# ALMA vs Mem0: Complete Comparison Guide

> Looking for a Mem0 alternative? This guide compares ALMA and Mem0 feature-by-feature.

## Quick Comparison

| Feature | ALMA | Mem0 |
|---------|------|------|
| **Memory Scoping** | `can_learn` / `cannot_learn` per agent | Basic user/session isolation |
| **Anti-Pattern Learning** | Yes - with `why_bad` + `better_alternative` | No |
| **Multi-Agent Sharing** | `inherit_from` + `share_with` | No |
| **Memory Consolidation** | LLM-powered deduplication | Basic |
| **Event System** | Webhooks + callbacks | No |
| **MCP Integration** | Native stdio/HTTP server | No |
| **TypeScript SDK** | Full-featured | No |
| **Graph Memory** | Neo4j, Memgraph, Kuzu | Limited |
| **Vector Backends** | 6 (PostgreSQL, Qdrant, Pinecone, Chroma, SQLite, Azure) | Limited |
| **Workflow Context** | Checkpoints, state merging, artifacts | No |
| **Open Source** | MIT License | Partially open |
| **Self-Hosted** | Yes, fully | Limited |

## Why Choose ALMA Over Mem0?

### 1. Scoped Learning Prevents Domain Confusion

With Mem0, all memories are accessible to all agents. ALMA lets you define exactly what each agent can and cannot learn:

```yaml
agents:
  frontend_tester:
    can_learn:
      - testing_strategies
      - selector_patterns
    cannot_learn:
      - backend_logic
      - database_queries
```

### 2. Anti-Pattern Learning

ALMA explicitly tracks what NOT to do:

```python
alma.add_anti_pattern(
    agent="helena",
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests and slow execution",
    better_alternative="Use explicit waits with conditions"
)
```

Mem0 has no equivalent feature.

### 3. Multi-Agent Memory Sharing

ALMA enables hierarchical knowledge sharing:

```yaml
agents:
  senior_dev:
    share_with: [junior_dev, qa_agent]

  junior_dev:
    inherit_from: [senior_dev]
```

### 4. Native MCP Integration

ALMA runs as an MCP server for direct Claude Code integration:

```json
{
  "mcpServers": {
    "alma-memory": {
      "command": "python",
      "args": ["-m", "alma.mcp"]
    }
  }
}
```

16 MCP tools available out of the box.

### 5. Workflow Context Layer

ALMA v0.6.0 adds checkpoints, state merging, and artifact linking for complex multi-agent workflows:

```python
# Save workflow state
alma.checkpoint(workflow_id="deploy-v2", state=current_state)

# Resume after failure
alma.resume(workflow_id="deploy-v2")

# Merge parallel agent outputs
alma.merge_states(workflow_id, states, reducer="latest_wins")
```

## Migration from Mem0 to ALMA

```python
# Before (Mem0)
from mem0 import Memory
m = Memory()
m.add("User prefers TypeScript", user_id="user-1")

# After (ALMA)
from alma import ALMA
alma = ALMA.from_config(".alma/config.yaml")
alma.add_preference(
    user_id="user-1",
    category="language",
    preference="User prefers TypeScript"
)
```

## Installation

```bash
pip install alma-memory
```

## Links

- [GitHub Repository](https://github.com/RBKunnela/ALMA-memory)
- [PyPI Package](https://pypi.org/project/alma-memory/)
- [Documentation](https://github.com/RBKunnela/ALMA-memory#readme)

---

*ALMA - Agent Learning Memory Architecture. MIT Licensed.*
