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
"""

from alma.mcp.server import ALMAMCPServer
from alma.mcp.tools import (
    alma_retrieve,
    alma_learn,
    alma_add_preference,
    alma_add_knowledge,
    alma_forget,
    alma_stats,
    alma_health,
)

__all__ = [
    "ALMAMCPServer",
    "alma_retrieve",
    "alma_learn",
    "alma_add_preference",
    "alma_add_knowledge",
    "alma_forget",
    "alma_stats",
    "alma_health",
]
