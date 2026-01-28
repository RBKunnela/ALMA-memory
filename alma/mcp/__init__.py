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
"""

from alma.mcp.server import ALMAMCPServer
from alma.mcp.tools import (
    # Sync tools
    alma_add_knowledge,
    alma_add_preference,
    alma_consolidate,
    alma_forget,
    alma_health,
    alma_learn,
    alma_retrieve,
    alma_stats,
    # Async tools
    async_alma_add_knowledge,
    async_alma_add_preference,
    async_alma_forget,
    async_alma_health,
    async_alma_learn,
    async_alma_retrieve,
    async_alma_stats,
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
    # Async tools
    "async_alma_retrieve",
    "async_alma_learn",
    "async_alma_add_preference",
    "async_alma_add_knowledge",
    "async_alma_forget",
    "async_alma_stats",
    "async_alma_health",
]
