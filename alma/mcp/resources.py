"""
ALMA MCP Resource Definitions.

Provides read-only resources that can be accessed via MCP protocol.
Resources represent configuration and metadata about the ALMA instance.
"""

import logging
from typing import Dict, Any, List

from alma import ALMA
from alma.types import MemoryScope

logger = logging.getLogger(__name__)


def get_config_resource(alma: ALMA) -> Dict[str, Any]:
    """
    Get current ALMA configuration as a resource.

    This exposes non-sensitive configuration information about
    the ALMA instance.

    Args:
        alma: ALMA instance

    Returns:
        Dict with configuration details
    """
    return {
        "uri": "alma://config",
        "name": "ALMA Configuration",
        "description": "Current configuration of the ALMA memory system",
        "mimeType": "application/json",
        "content": {
            "project_id": alma.project_id,
            "storage_type": type(alma.storage).__name__,
            "registered_agents": list(alma.scopes.keys()),
            "scopes": {
                name: {
                    "can_learn": scope.can_learn,
                    "cannot_learn": scope.cannot_learn,
                    "min_occurrences_for_heuristic": scope.min_occurrences_for_heuristic,
                }
                for name, scope in alma.scopes.items()
            },
        },
    }


def get_agents_resource(alma: ALMA) -> Dict[str, Any]:
    """
    Get registered agents and their scopes as a resource.

    This provides detailed information about each registered agent
    and their learning permissions.

    Args:
        alma: ALMA instance

    Returns:
        Dict with agent details
    """
    agents = []

    for name, scope in alma.scopes.items():
        # Get stats for this agent
        try:
            stats = alma.get_stats(agent=name)
        except Exception:
            stats = {}

        agents.append({
            "name": name,
            "scope": {
                "can_learn": scope.can_learn,
                "cannot_learn": scope.cannot_learn,
                "min_occurrences_for_heuristic": scope.min_occurrences_for_heuristic,
            },
            "stats": {
                "heuristics_count": stats.get("heuristics_count", 0),
                "outcomes_count": stats.get("outcomes_count", 0),
                "domain_knowledge_count": stats.get("domain_knowledge_count", 0),
                "anti_patterns_count": stats.get("anti_patterns_count", 0),
            },
        })

    return {
        "uri": "alma://agents",
        "name": "ALMA Registered Agents",
        "description": "All registered agents and their memory scopes",
        "mimeType": "application/json",
        "content": {
            "project_id": alma.project_id,
            "agent_count": len(agents),
            "agents": agents,
        },
    }


def list_resources() -> List[Dict[str, Any]]:
    """
    List all available resources.

    Returns:
        List of resource descriptors
    """
    return [
        {
            "uri": "alma://config",
            "name": "ALMA Configuration",
            "description": "Current configuration of the ALMA memory system",
            "mimeType": "application/json",
        },
        {
            "uri": "alma://agents",
            "name": "ALMA Registered Agents",
            "description": "All registered agents and their memory scopes",
            "mimeType": "application/json",
        },
    ]
