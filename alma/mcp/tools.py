"""
ALMA MCP Tool Definitions.

Provides the tool functions that can be called via MCP protocol.
Each tool corresponds to an ALMA operation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import asdict

from alma import ALMA
from alma.types import MemorySlice

logger = logging.getLogger(__name__)


def _serialize_memory_slice(memory_slice: MemorySlice) -> Dict[str, Any]:
    """Convert MemorySlice to JSON-serializable dict."""
    result = {
        "heuristics": [],
        "outcomes": [],
        "domain_knowledge": [],
        "anti_patterns": [],
        "preferences": [],
        "query": memory_slice.query,
        "agent": memory_slice.agent,
        "retrieval_time_ms": memory_slice.retrieval_time_ms,
        "total_items": memory_slice.total_items,
    }

    for h in memory_slice.heuristics:
        result["heuristics"].append({
            "id": h.id,
            "condition": h.condition,
            "strategy": h.strategy,
            "confidence": h.confidence,
            "occurrence_count": h.occurrence_count,
            "success_rate": h.success_rate,
        })

    for o in memory_slice.outcomes:
        result["outcomes"].append({
            "id": o.id,
            "task_type": o.task_type,
            "task_description": o.task_description,
            "success": o.success,
            "strategy_used": o.strategy_used,
            "duration_ms": o.duration_ms,
        })

    for dk in memory_slice.domain_knowledge:
        result["domain_knowledge"].append({
            "id": dk.id,
            "domain": dk.domain,
            "fact": dk.fact,
            "confidence": dk.confidence,
        })

    for ap in memory_slice.anti_patterns:
        result["anti_patterns"].append({
            "id": ap.id,
            "pattern": ap.pattern,
            "why_bad": ap.why_bad,
            "better_alternative": ap.better_alternative,
        })

    for p in memory_slice.preferences:
        result["preferences"].append({
            "id": p.id,
            "category": p.category,
            "preference": p.preference,
        })

    return result


def alma_retrieve(
    alma: ALMA,
    task: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve relevant memories for a task.

    Args:
        alma: ALMA instance
        task: Description of the task to perform
        agent: Name of the agent requesting memories
        user_id: Optional user ID for preference retrieval
        top_k: Maximum items per memory type

    Returns:
        Dict containing the memory slice with relevant memories
    """
    # Input validation
    if not task or not task.strip():
        return {"success": False, "error": "task cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    try:
        memories = alma.retrieve(
            task=task,
            agent=agent,
            user_id=user_id,
            top_k=top_k,
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_learn(
    alma: ALMA,
    agent: str,
    task: str,
    outcome: str,
    strategy_used: str,
    task_type: Optional[str] = None,
    duration_ms: Optional[int] = None,
    error_message: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a task outcome for learning.

    Args:
        alma: ALMA instance
        agent: Name of the agent that executed the task
        task: Description of the task
        outcome: "success" or "failure"
        strategy_used: What approach was taken
        task_type: Category of task (for grouping)
        duration_ms: How long the task took
        error_message: Error details if failed
        feedback: User feedback if provided

    Returns:
        Dict with learning result
    """
    # Input validation
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not task or not task.strip():
        return {"success": False, "error": "task cannot be empty"}
    if not outcome or not outcome.strip():
        return {"success": False, "error": "outcome cannot be empty"}
    if not strategy_used or not strategy_used.strip():
        return {"success": False, "error": "strategy_used cannot be empty"}

    try:
        result = alma.learn(
            agent=agent,
            task=task,
            outcome=outcome,
            strategy_used=strategy_used,
            task_type=task_type,
            duration_ms=duration_ms,
            error_message=error_message,
            feedback=feedback,
        )

        return {
            "success": True,
            "learned": result,
            "message": "Outcome recorded" if result else "Learning rejected (scope violation)",
        }

    except Exception as e:
        logger.exception(f"Error in alma_learn: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_add_preference(
    alma: ALMA,
    user_id: str,
    category: str,
    preference: str,
    source: str = "explicit_instruction",
) -> Dict[str, Any]:
    """
    Add a user preference to memory.

    Args:
        alma: ALMA instance
        user_id: User identifier
        category: Category (communication, code_style, workflow)
        preference: The preference text
        source: How this was learned

    Returns:
        Dict with the created preference
    """
    # Input validation
    if not user_id or not user_id.strip():
        return {"success": False, "error": "user_id cannot be empty"}
    if not category or not category.strip():
        return {"success": False, "error": "category cannot be empty"}
    if not preference or not preference.strip():
        return {"success": False, "error": "preference cannot be empty"}

    try:
        pref = alma.add_user_preference(
            user_id=user_id,
            category=category,
            preference=preference,
            source=source,
        )

        return {
            "success": True,
            "preference": {
                "id": pref.id,
                "user_id": pref.user_id,
                "category": pref.category,
                "preference": pref.preference,
                "source": pref.source,
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_add_preference: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_add_knowledge(
    alma: ALMA,
    agent: str,
    domain: str,
    fact: str,
    source: str = "user_stated",
) -> Dict[str, Any]:
    """
    Add domain knowledge within agent's scope.

    Args:
        alma: ALMA instance
        agent: Agent this knowledge belongs to
        domain: Knowledge domain
        fact: The fact to remember
        source: How this was learned

    Returns:
        Dict with the created knowledge or rejection reason
    """
    # Input validation
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not domain or not domain.strip():
        return {"success": False, "error": "domain cannot be empty"}
    if not fact or not fact.strip():
        return {"success": False, "error": "fact cannot be empty"}

    try:
        knowledge = alma.add_domain_knowledge(
            agent=agent,
            domain=domain,
            fact=fact,
            source=source,
        )

        if knowledge is None:
            return {
                "success": False,
                "error": f"Agent '{agent}' not allowed to learn in domain '{domain}'",
            }

        return {
            "success": True,
            "knowledge": {
                "id": knowledge.id,
                "agent": knowledge.agent,
                "domain": knowledge.domain,
                "fact": knowledge.fact,
                "source": knowledge.source,
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_add_knowledge: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_forget(
    alma: ALMA,
    agent: Optional[str] = None,
    older_than_days: int = 90,
    below_confidence: float = 0.3,
) -> Dict[str, Any]:
    """
    Prune stale or low-confidence memories.

    Args:
        alma: ALMA instance
        agent: Specific agent to prune, or None for all
        older_than_days: Remove outcomes older than this
        below_confidence: Remove heuristics below this confidence

    Returns:
        Dict with number of items pruned
    """
    try:
        count = alma.forget(
            agent=agent,
            older_than_days=older_than_days,
            below_confidence=below_confidence,
        )

        return {
            "success": True,
            "pruned_count": count,
            "message": f"Pruned {count} stale or low-confidence memories",
        }

    except Exception as e:
        logger.exception(f"Error in alma_forget: {e}")
        return {
            "success": False,
            "error": str(e),
        }


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
