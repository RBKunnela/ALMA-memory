# MCP Tools Reference

16 tools available via the ALMA MCP server.

## Core Tools

| Tool | Parameters | Returns |
|------|------------|---------|
| `alma_retrieve` | task, agent, top_k?, user_id? | MemorySlice |
| `alma_learn` | agent, task, outcome, strategy_used, ... | LearnResult |
| `alma_add_preference` | user_id, category, preference, source? | Success |
| `alma_add_knowledge` | agent, domain, fact, source? | Success |
| `alma_forget` | agent?, older_than_days?, below_confidence? | PruneResult |
| `alma_stats` | agent? | MemoryStats |
| `alma_health` | - | HealthStatus |

## Workflow Tools

| Tool | Parameters | Returns |
|------|------------|---------|
| `alma_consolidate` | agent, memory_type, threshold? | ConsolidateResult |
| `alma_checkpoint` | workflow_id, state, metadata? | CheckpointResult |
| `alma_resume` | workflow_id | Checkpoint |
| `alma_merge_states` | workflow_id, states, reducer? | MergedState |
| `alma_workflow_learn` | workflow_id, agent, task, outcome, ... | LearnResult |
| `alma_link_artifact` | workflow_id, artifact_type, ref, metadata? | Success |
| `alma_get_artifacts` | workflow_id, artifact_type? | Artifact[] |
| `alma_cleanup_checkpoints` | older_than_days? | CleanupResult |
| `alma_retrieve_scoped` | query, scope, agent? | MemorySlice |

See [MCP Integration Guide](../guides/mcp-integration.md) for usage examples.
