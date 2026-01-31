# MCP Integration

ALMA runs as an MCP (Model Context Protocol) server, enabling direct integration with Claude Code and other MCP-compatible clients.

## Setup

### 1. Configure MCP Server

Add to your `.mcp.json` or MCP configuration:

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

### 2. Start the Server

The server starts automatically when Claude Code loads, or manually:

```bash
python -m alma.mcp --config .alma/config.yaml
```

For HTTP mode:

```bash
python -m alma.mcp --config .alma/config.yaml --http --port 8765
```

## Available Tools (16 Total)

### Core Tools

| Tool | Description |
|------|-------------|
| `alma_retrieve` | Get memories relevant to a task |
| `alma_learn` | Record task outcome |
| `alma_add_preference` | Add user preference |
| `alma_add_knowledge` | Add domain knowledge |
| `alma_forget` | Prune old/low-confidence memories |
| `alma_stats` | Get memory statistics |
| `alma_health` | Health check |

### Workflow Tools (v0.6.0)

| Tool | Description |
|------|-------------|
| `alma_consolidate` | Merge similar memories |
| `alma_checkpoint` | Save workflow state |
| `alma_resume` | Resume from checkpoint |
| `alma_merge_states` | Merge parallel agent states |
| `alma_workflow_learn` | Learn with workflow context |
| `alma_link_artifact` | Link output to workflow |
| `alma_get_artifacts` | Get workflow artifacts |
| `alma_cleanup_checkpoints` | Clean old checkpoints |
| `alma_retrieve_scoped` | Scoped memory retrieval |

## Tool Examples

### Retrieve Memories

```json
{
  "name": "alma_retrieve",
  "arguments": {
    "task": "Test the login form validation",
    "agent": "helena",
    "top_k": 5
  }
}
```

### Learn from Outcome

```json
{
  "name": "alma_learn",
  "arguments": {
    "agent": "helena",
    "task": "Test login form",
    "outcome": "success",
    "strategy_used": "Tested empty fields, invalid email, valid submission"
  }
}
```

### Checkpoint Workflow

```json
{
  "name": "alma_checkpoint",
  "arguments": {
    "workflow_id": "deploy-v2",
    "state": {
      "current_step": "run_tests",
      "tests_passed": 42,
      "tests_failed": 0
    },
    "metadata": {
      "agent": "victor",
      "started_at": "2024-01-15T10:00:00Z"
    }
  }
}
```

### Resume Workflow

```json
{
  "name": "alma_resume",
  "arguments": {
    "workflow_id": "deploy-v2"
  }
}
```

## Resources

The MCP server also exposes resources:

| Resource | Description |
|----------|-------------|
| `alma://config` | Current configuration |
| `alma://agents` | Registered agents |
| `alma://stats` | Memory statistics |

## Using with Claude Code

Once configured, Claude Code can:

1. **Retrieve context** before tasks:
   ```
   Claude: Let me check what I know about testing forms...
   [Calls alma_retrieve]
   ```

2. **Learn from outcomes**:
   ```
   Claude: That worked! Let me remember this strategy.
   [Calls alma_learn]
   ```

3. **Manage complex workflows**:
   ```
   Claude: Saving checkpoint before deployment...
   [Calls alma_checkpoint]
   ```

## TypeScript SDK Connection

The TypeScript SDK connects to the HTTP mode:

```typescript
import { ALMA } from '@rbkunnela/alma-memory';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',  // MCP HTTP server
  projectId: 'my-project'
});
```
