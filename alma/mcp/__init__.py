"""
ALMA MCP Server Module.

Exposes ALMA functionality to any Claude Code instance via the
Model Context Protocol (MCP).

Usage:
    # stdio mode (for Claude Code integration)
    python -m alma.mcp --config .alma/config.yaml

    # HTTP mode (for remote access)
    python -m alma.mcp --http --port 8765

Integration with Claude Code (.mcp.json):
    {
        "mcpServers": {
            "alma-memory": {
                "command": "python",
                "args": ["-m", "alma.mcp", "--config", ".alma/config.yaml"]
            }
        }
    }

Async API:
    Both sync and async variants of tools are available:
    - Sync: alma_retrieve, alma_learn, etc.
    - Async: async_alma_retrieve, async_alma_learn, etc.

Workflow Tools (v0.6.0):
    Tools for AGtestari workflow integration:
    - alma_checkpoint: Create crash recovery checkpoints
    - alma_resume: Get checkpoint to resume from
    - alma_merge_states: Merge parallel branch states
    - alma_workflow_learn: Record workflow outcomes
    - alma_link_artifact: Link artifacts to memories
    - alma_get_artifacts: Get artifacts for a memory
    - alma_cleanup_checkpoints: Clean up old checkpoints
    - alma_retrieve_scoped: Scoped memory retrieval
"""

from alma.mcp.server import ALMAMCPServer
from alma.mcp.tools import (
    # Sync tools
    alma_add_knowledge,
    alma_add_preference,
    # Workflow tools (v0.6.0)
    alma_checkpoint,
    alma_cleanup_checkpoints,
    alma_consolidate,
    alma_forget,
    alma_get_artifacts,
    alma_health,
    alma_learn,
    alma_link_artifact,
    alma_merge_states,
    alma_resume,
    alma_retrieve,
    alma_retrieve_scoped,
    alma_stats,
    alma_workflow_learn,
    # Async tools
    async_alma_add_knowledge,
    async_alma_add_preference,
    # Async workflow tools
    async_alma_checkpoint,
    async_alma_forget,
    async_alma_health,
    async_alma_learn,
    async_alma_link_artifact,
    async_alma_resume,
    async_alma_retrieve,
    async_alma_retrieve_scoped,
    async_alma_stats,
    async_alma_workflow_learn,
)

__all__ = [
    "ALMAMCPServer",
    # Sync tools
    "alma_retrieve",
    "alma_learn",
    "alma_add_preference",
    "alma_add_knowledge",
    "alma_forget",
    "alma_stats",
    "alma_health",
    "alma_consolidate",
    # Workflow tools (v0.6.0)
    "alma_checkpoint",
    "alma_resume",
    "alma_merge_states",
    "alma_workflow_learn",
    "alma_link_artifact",
    "alma_get_artifacts",
    "alma_cleanup_checkpoints",
    "alma_retrieve_scoped",
    # Async tools
    "async_alma_retrieve",
    "async_alma_learn",
    "async_alma_add_preference",
    "async_alma_add_knowledge",
    "async_alma_forget",
    "async_alma_stats",
    "async_alma_health",
    # Async workflow tools
    "async_alma_checkpoint",
    "async_alma_resume",
    "async_alma_workflow_learn",
    "async_alma_link_artifact",
    "async_alma_retrieve_scoped",
]
