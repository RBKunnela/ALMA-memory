"""
ALMA MCP Admin Tools.

Provides sync and async admin tool functions for the MCP protocol.
Includes stats, health checks, trust pattern storage, and trust warnings.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from alma import ALMA

logger = logging.getLogger(__name__)


def alma_stats(
    alma: ALMA,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get memory statistics.

    Args:
        alma: ALMA instance
        agent: Specific agent or None for all

    Returns:
        Dict with memory statistics
    """
    try:
        stats = alma.get_stats(agent=agent)

        return {
            "success": True,
            "stats": stats,
        }

    except Exception as e:
        logger.exception(f"Error in alma_stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_health(alma: ALMA) -> Dict[str, Any]:
    """
    Health check for ALMA server.

    Args:
        alma: ALMA instance

    Returns:
        Dict with health status
    """
    try:
        # Basic health checks
        stats = alma.get_stats()

        return {
            "success": True,
            "status": "healthy",
            "project_id": alma.project_id,
            "total_memories": stats.get("total_count", 0),
            "registered_agents": list(alma.scopes.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error in alma_health: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
        }


def alma_store_trust_pattern(
    alma: ALMA,
    agent_id: str,
    pattern_type: str,
    task_type: str,
    description: str,
    evidence: Optional[str] = None,
    severity: str = "medium",
) -> Dict[str, Any]:
    """
    Store a trust-related pattern (violation or verification).

    Use this to record:
    - Trust violations (claims without evidence, silent failures)
    - Verification patterns (successful verification approaches)

    These patterns are retrieved as warnings for similar future tasks.

    Args:
        alma: ALMA instance
        agent_id: Agent the pattern is about
        pattern_type: Type (violation, verification)
        task_type: Category of task where pattern occurred
        description: Description of the pattern
        evidence: Optional evidence or context
        severity: Severity level (low, medium, high, critical)

    Returns:
        Dict with stored pattern info
    """
    if not agent_id or not agent_id.strip():
        return {"success": False, "error": "agent_id cannot be empty"}
    if not pattern_type or pattern_type not in ["violation", "verification"]:
        return {
            "success": False,
            "error": "pattern_type must be 'violation' or 'verification'",
        }
    if not task_type or not task_type.strip():
        return {"success": False, "error": "task_type cannot be empty"}
    if not description or not description.strip():
        return {"success": False, "error": "description cannot be empty"}

    valid_severities = ["low", "medium", "high", "critical"]
    if severity not in valid_severities:
        return {
            "success": False,
            "error": f"severity must be one of: {', '.join(valid_severities)}",
        }

    try:
        from alma.retrieval.trust_scoring import TrustPatternStore

        pattern_store = TrustPatternStore(alma.storage)

        if pattern_type == "violation":
            pattern_id = pattern_store.store_trust_violation(
                agent_id=agent_id,
                project_id=alma.project_id,
                violation_type=task_type,
                description=description,
                severity=severity,
                evidence=evidence,
            )
            message = "Trust violation recorded"
        else:
            pattern_id = pattern_store.store_verification_pattern(
                agent_id=agent_id,
                project_id=alma.project_id,
                task_type=task_type,
                verification_approach=description,
                evidence=evidence,
            )
            message = "Verification pattern recorded"

        return {
            "success": True,
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "agent_id": agent_id,
            "message": message,
        }

    except Exception as e:
        logger.exception(f"Error in alma_store_trust_pattern: {e}")
        return {"success": False, "error": str(e)}


def alma_get_trust_warnings(
    alma: ALMA,
    task_description: str,
    agent_id: str,
) -> Dict[str, Any]:
    """
    Get trust warnings relevant to a task.

    Retrieves past violations and warnings that are semantically
    similar to the current task. Use this before starting a task
    to understand trust-related pitfalls.

    Args:
        alma: ALMA instance
        task_description: Description of the task to check
        agent_id: Agent to get warnings for

    Returns:
        Dict with list of relevant warnings
    """
    if not task_description or not task_description.strip():
        return {"success": False, "error": "task_description cannot be empty"}
    if not agent_id or not agent_id.strip():
        return {"success": False, "error": "agent_id cannot be empty"}

    try:
        from alma.retrieval.trust_scoring import TrustPatternStore

        pattern_store = TrustPatternStore(alma.storage)

        warnings = pattern_store.retrieve_trust_warnings(
            task_description=task_description,
            agent_id=agent_id,
            project_id=alma.project_id,
        )

        return {
            "success": True,
            "warnings": warnings,
            "count": len(warnings),
            "has_warnings": len(warnings) > 0,
        }

    except Exception as e:
        logger.exception(f"Error in alma_get_trust_warnings: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ASYNC ADMIN TOOLS
# =============================================================================


async def async_alma_stats(
    alma: ALMA,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async version of alma_stats. Get memory statistics.

    Args:
        alma: ALMA instance
        agent: Specific agent or None for all

    Returns:
        Dict with memory statistics
    """
    try:
        stats = await alma.async_get_stats(agent=agent)

        return {
            "success": True,
            "stats": stats,
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_health(alma: ALMA) -> Dict[str, Any]:
    """
    Async version of alma_health. Health check for ALMA server.

    Args:
        alma: ALMA instance

    Returns:
        Dict with health status
    """
    try:
        # Basic health checks
        stats = await alma.async_get_stats()

        return {
            "success": True,
            "status": "healthy",
            "project_id": alma.project_id,
            "total_memories": stats.get("total_count", 0),
            "registered_agents": list(alma.scopes.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_health: {e}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
        }


async def async_alma_store_trust_pattern(
    alma: ALMA,
    agent_id: str,
    pattern_type: str,
    task_type: str,
    description: str,
    evidence: Optional[str] = None,
    severity: str = "medium",
) -> Dict[str, Any]:
    """Async version of alma_store_trust_pattern."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_store_trust_pattern(
            alma, agent_id, pattern_type, task_type, description, evidence, severity
        ),
    )


async def async_alma_get_trust_warnings(
    alma: ALMA,
    task_description: str,
    agent_id: str,
) -> Dict[str, Any]:
    """Async version of alma_get_trust_warnings."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_get_trust_warnings(alma, task_description, agent_id),
    )
