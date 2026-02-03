"""
ALMA MCP Tool Definitions.

Provides the tool functions that can be called via MCP protocol.
Each tool corresponds to an ALMA operation.

Both sync and async versions are provided:
- Sync tools: alma_retrieve, alma_learn, etc.
- Async tools: async_alma_retrieve, async_alma_learn, etc.

The async tools use ALMA's async API for better concurrency in
async MCP server implementations.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from alma import ALMA
from alma.retrieval.modes import RetrievalMode
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
        result["heuristics"].append(
            {
                "id": h.id,
                "condition": h.condition,
                "strategy": h.strategy,
                "confidence": h.confidence,
                "occurrence_count": h.occurrence_count,
                "success_rate": h.success_rate,
            }
        )

    for o in memory_slice.outcomes:
        result["outcomes"].append(
            {
                "id": o.id,
                "task_type": o.task_type,
                "task_description": o.task_description,
                "success": o.success,
                "strategy_used": o.strategy_used,
                "duration_ms": o.duration_ms,
            }
        )

    for dk in memory_slice.domain_knowledge:
        result["domain_knowledge"].append(
            {
                "id": dk.id,
                "domain": dk.domain,
                "fact": dk.fact,
                "confidence": dk.confidence,
            }
        )

    for ap in memory_slice.anti_patterns:
        result["anti_patterns"].append(
            {
                "id": ap.id,
                "pattern": ap.pattern,
                "why_bad": ap.why_bad,
                "better_alternative": ap.better_alternative,
            }
        )

    for p in memory_slice.preferences:
        result["preferences"].append(
            {
                "id": p.id,
                "category": p.category,
                "preference": p.preference,
            }
        )

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


def alma_retrieve_for_mode(
    alma: ALMA,
    query: str,
    mode: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieve memories using a specific retrieval mode.

    Mode-aware retrieval adapts strategy based on task type:
    - BROAD: For planning, brainstorming - diverse, exploratory results
    - PRECISE: For execution, implementation - high-confidence matches
    - DIAGNOSTIC: For debugging, troubleshooting - prioritizes failures
    - LEARNING: For pattern finding - similar memories for consolidation
    - RECALL: For exact lookup - prioritizes exact matches

    Args:
        alma: ALMA instance
        query: Description of the task to perform
        mode: Retrieval mode (broad, precise, diagnostic, learning, recall)
        agent: Name of the agent requesting memories
        user_id: Optional user ID for preference retrieval
        top_k: Override mode's default top_k

    Returns:
        Dict containing memories, mode used, and reason for mode selection
    """
    # Input validation
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not mode or not mode.strip():
        return {"success": False, "error": "mode cannot be empty"}

    # Validate mode
    valid_modes = ["broad", "precise", "diagnostic", "learning", "recall"]
    mode_lower = mode.lower()
    if mode_lower not in valid_modes:
        return {
            "success": False,
            "error": f"mode must be one of: {', '.join(valid_modes)}",
        }

    try:
        # Convert string to RetrievalMode enum
        retrieval_mode = RetrievalMode(mode_lower)

        # Call retrieve_with_mode on the retrieval engine
        memories, used_mode, mode_reason = alma.retrieval.retrieve_with_mode(
            query=query,
            agent=agent,
            project_id=alma.project_id,
            mode=retrieval_mode,
            user_id=user_id,
            top_k=top_k,
            scope=alma.scopes.get(agent),
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "mode": used_mode.value,
            "mode_reason": mode_reason,
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_for_mode: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def alma_retrieve_smart(
    alma: ALMA,
    query: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Retrieve memories with auto-inferred retrieval mode.

    Automatically detects the best retrieval mode from the query text:
    - Queries with "error", "bug", "fix", "debug" → DIAGNOSTIC mode
    - Queries with "how should", "options", "plan" → BROAD mode
    - Queries with "what was", "remember when" → RECALL mode
    - Queries with "pattern", "similar" → LEARNING mode
    - Default → PRECISE mode for implementation tasks

    Args:
        alma: ALMA instance
        query: Description of the task to perform
        agent: Name of the agent requesting memories
        user_id: Optional user ID for preference retrieval
        top_k: Override mode's default top_k

    Returns:
        Dict containing memories, inferred mode, and reason for mode selection
    """
    # Input validation
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    try:
        # Call retrieve_with_mode with mode=None to auto-infer
        memories, inferred_mode, mode_reason = alma.retrieval.retrieve_with_mode(
            query=query,
            agent=agent,
            project_id=alma.project_id,
            mode=None,  # Auto-infer
            user_id=user_id,
            top_k=top_k,
            scope=alma.scopes.get(agent),
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "mode": inferred_mode.value,
            "mode_reason": mode_reason,
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_smart: {e}")
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
        outcome_record = alma.learn(
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
            "learned": True,
            "outcome": {
                "id": outcome_record.id,
                "agent": outcome_record.agent,
                "task_type": outcome_record.task_type,
                "success": outcome_record.success,
            },
            "message": "Outcome recorded successfully",
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


async def alma_consolidate(
    alma: ALMA,
    agent: str,
    memory_type: str = "heuristics",
    similarity_threshold: float = 0.85,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Consolidate similar memories to reduce redundancy.

    This is ALMA's implementation of Mem0's core innovation - LLM-powered
    deduplication that merges similar memories intelligently.

    Args:
        alma: ALMA instance
        agent: Agent whose memories to consolidate
        memory_type: Type of memory to consolidate
                    ("heuristics", "outcomes", "domain_knowledge", "anti_patterns")
        similarity_threshold: Minimum cosine similarity to group (0.0 to 1.0)
                             Higher values are more conservative (fewer merges)
        dry_run: If True, report what would be merged without actually modifying storage
                 Recommended for first run to preview changes

    Returns:
        Dict with consolidation results including:
        - merged_count: Number of memories merged
        - groups_found: Number of similar memory groups identified
        - memories_processed: Total memories analyzed
        - merge_details: List of merge operations (or planned operations if dry_run)
        - errors: Any errors encountered
    """
    # Input validation
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    valid_types = ["heuristics", "outcomes", "domain_knowledge", "anti_patterns"]
    if memory_type not in valid_types:
        return {
            "success": False,
            "error": f"memory_type must be one of: {', '.join(valid_types)}",
        }

    if not 0.0 <= similarity_threshold <= 1.0:
        return {
            "success": False,
            "error": "similarity_threshold must be between 0.0 and 1.0",
        }

    try:
        from alma.consolidation import ConsolidationEngine

        # Create consolidation engine
        engine = ConsolidationEngine(
            storage=alma.storage,
            embedder=None,  # Will use default LocalEmbedder
            llm_client=None,  # LLM merging disabled by default
        )

        # Run consolidation
        result = await engine.consolidate(
            agent=agent,
            project_id=alma.project_id,
            memory_type=memory_type,
            similarity_threshold=similarity_threshold,
            use_llm=False,  # LLM disabled - uses highest confidence merge
            dry_run=dry_run,
        )

        # Invalidate cache after consolidation (if not dry run)
        if not dry_run and result.merged_count > 0:
            alma.retrieval.invalidate_cache(agent=agent, project_id=alma.project_id)

        return {
            "success": result.success,
            "dry_run": dry_run,
            "merged_count": result.merged_count,
            "groups_found": result.groups_found,
            "memories_processed": result.memories_processed,
            "merge_details": result.merge_details,
            "errors": result.errors,
            "message": (
                f"{'Would merge' if dry_run else 'Merged'} {result.merged_count} memories "
                f"from {result.groups_found} similar groups "
                f"(processed {result.memories_processed} total)"
            ),
        }

    except Exception as e:
        logger.exception(f"Error in alma_consolidate: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# ASYNC MCP TOOLS
# =============================================================================
#
# Async versions of MCP tools for use in async MCP server implementations.
# These use ALMA's async_* methods to avoid blocking the event loop.


async def async_alma_retrieve(
    alma: ALMA,
    task: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Async version of alma_retrieve. Retrieve relevant memories for a task.

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
        memories = await alma.async_retrieve(
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
        logger.exception(f"Error in async_alma_retrieve: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_retrieve_for_mode(
    alma: ALMA,
    query: str,
    mode: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Async version of alma_retrieve_for_mode.

    Retrieve memories using a specific retrieval mode asynchronously.

    Args:
        alma: ALMA instance
        query: Description of the task to perform
        mode: Retrieval mode (broad, precise, diagnostic, learning, recall)
        agent: Name of the agent requesting memories
        user_id: Optional user ID for preference retrieval
        top_k: Override mode's default top_k

    Returns:
        Dict containing memories, mode used, and reason for mode selection
    """
    # Input validation
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not mode or not mode.strip():
        return {"success": False, "error": "mode cannot be empty"}

    # Validate mode
    valid_modes = ["broad", "precise", "diagnostic", "learning", "recall"]
    mode_lower = mode.lower()
    if mode_lower not in valid_modes:
        return {
            "success": False,
            "error": f"mode must be one of: {', '.join(valid_modes)}",
        }

    try:
        # Convert string to RetrievalMode enum
        retrieval_mode = RetrievalMode(mode_lower)

        # Call retrieve_with_mode on the retrieval engine
        # Note: retrieve_with_mode is synchronous, run in executor for async
        import asyncio

        loop = asyncio.get_event_loop()
        memories, used_mode, mode_reason = await loop.run_in_executor(
            None,
            lambda: alma.retrieval.retrieve_with_mode(
                query=query,
                agent=agent,
                project_id=alma.project_id,
                mode=retrieval_mode,
                user_id=user_id,
                top_k=top_k,
                scope=alma.scopes.get(agent),
            ),
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "mode": used_mode.value,
            "mode_reason": mode_reason,
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_retrieve_for_mode: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_retrieve_smart(
    alma: ALMA,
    query: str,
    agent: str,
    user_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Async version of alma_retrieve_smart.

    Retrieve memories with auto-inferred retrieval mode asynchronously.

    Args:
        alma: ALMA instance
        query: Description of the task to perform
        agent: Name of the agent requesting memories
        user_id: Optional user ID for preference retrieval
        top_k: Override mode's default top_k

    Returns:
        Dict containing memories, inferred mode, and reason for mode selection
    """
    # Input validation
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    try:
        # Call retrieve_with_mode with mode=None to auto-infer
        # Note: retrieve_with_mode is synchronous, run in executor for async
        import asyncio

        loop = asyncio.get_event_loop()
        memories, inferred_mode, mode_reason = await loop.run_in_executor(
            None,
            lambda: alma.retrieval.retrieve_with_mode(
                query=query,
                agent=agent,
                project_id=alma.project_id,
                mode=None,  # Auto-infer
                user_id=user_id,
                top_k=top_k,
                scope=alma.scopes.get(agent),
            ),
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "mode": inferred_mode.value,
            "mode_reason": mode_reason,
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_retrieve_smart: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_learn(
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
    Async version of alma_learn. Record a task outcome for learning.

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
        outcome_record = await alma.async_learn(
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
            "learned": True,
            "outcome": {
                "id": outcome_record.id,
                "agent": outcome_record.agent,
                "task_type": outcome_record.task_type,
                "success": outcome_record.success,
            },
            "message": "Outcome recorded successfully",
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_learn: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_add_preference(
    alma: ALMA,
    user_id: str,
    category: str,
    preference: str,
    source: str = "explicit_instruction",
) -> Dict[str, Any]:
    """
    Async version of alma_add_preference. Add a user preference to memory.

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
        pref = await alma.async_add_user_preference(
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
        logger.exception(f"Error in async_alma_add_preference: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_add_knowledge(
    alma: ALMA,
    agent: str,
    domain: str,
    fact: str,
    source: str = "user_stated",
) -> Dict[str, Any]:
    """
    Async version of alma_add_knowledge. Add domain knowledge within agent's scope.

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
        knowledge = await alma.async_add_domain_knowledge(
            agent=agent,
            domain=domain,
            fact=fact,
            source=source,
        )

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
        logger.exception(f"Error in async_alma_add_knowledge: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def async_alma_forget(
    alma: ALMA,
    agent: Optional[str] = None,
    older_than_days: int = 90,
    below_confidence: float = 0.3,
) -> Dict[str, Any]:
    """
    Async version of alma_forget. Prune stale or low-confidence memories.

    Args:
        alma: ALMA instance
        agent: Specific agent to prune, or None for all
        older_than_days: Remove outcomes older than this
        below_confidence: Remove heuristics below this confidence

    Returns:
        Dict with number of items pruned
    """
    try:
        count = await alma.async_forget(
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
        logger.exception(f"Error in async_alma_forget: {e}")
        return {
            "success": False,
            "error": str(e),
        }


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


# =============================================================================
# WORKFLOW MCP TOOLS (v0.6.0)
# =============================================================================
#
# Tools for workflow integration: checkpointing, scoped retrieval,
# learning from workflows, and artifact linking.


def alma_checkpoint(
    alma: ALMA,
    run_id: str,
    node_id: str,
    state: Dict[str, Any],
    branch_id: Optional[str] = None,
    parent_checkpoint_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_if_unchanged: bool = True,
) -> Dict[str, Any]:
    """
    Create a checkpoint for crash recovery.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        node_id: The node creating this checkpoint
        state: The state to persist
        branch_id: Optional branch identifier for parallel execution
        parent_checkpoint_id: Previous checkpoint in the chain
        metadata: Additional checkpoint metadata
        skip_if_unchanged: If True, skip if state hasn't changed

    Returns:
        Dict with checkpoint info or skip notification
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not node_id or not node_id.strip():
        return {"success": False, "error": "node_id cannot be empty"}

    try:
        checkpoint = alma.checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

        if checkpoint is None:
            return {
                "success": True,
                "skipped": True,
                "message": "Checkpoint skipped - state unchanged",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "state_hash": checkpoint.state_hash,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_checkpoint: {e}")
        return {"success": False, "error": str(e)}


def alma_resume(
    alma: ALMA,
    run_id: str,
    branch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the checkpoint to resume from after a crash.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        branch_id: Optional branch to filter by

    Returns:
        Dict with checkpoint info or None if no checkpoints
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        checkpoint = alma.get_resume_point(run_id, branch_id)

        if checkpoint is None:
            return {
                "success": True,
                "checkpoint": None,
                "message": "No checkpoint found for this run",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_resume: {e}")
        return {"success": False, "error": str(e)}


def alma_merge_states(
    alma: ALMA,
    states: list,
    reducer_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Merge multiple branch states after parallel execution.

    Args:
        alma: ALMA instance
        states: List of state dicts from parallel branches
        reducer_config: Optional mapping of key -> reducer name.
                       Available: append, merge_dict, last_value,
                       first_value, sum, max, min, union

    Returns:
        Dict with merged state
    """
    if not states:
        return {"success": True, "merged_state": {}}

    try:
        merged = alma.merge_states(states, reducer_config)

        return {
            "success": True,
            "merged_state": merged,
            "input_count": len(states),
        }

    except Exception as e:
        logger.exception(f"Error in alma_merge_states: {e}")
        return {"success": False, "error": str(e)}


def alma_workflow_learn(
    alma: ALMA,
    agent: str,
    workflow_id: str,
    run_id: str,
    result: str,
    summary: str,
    strategies_used: Optional[list] = None,
    successful_patterns: Optional[list] = None,
    failed_patterns: Optional[list] = None,
    duration_seconds: Optional[float] = None,
    node_count: Optional[int] = None,
    error_message: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record learnings from a completed workflow execution.

    Args:
        alma: ALMA instance
        agent: The agent that executed the workflow
        workflow_id: The workflow definition that was executed
        run_id: The specific run this outcome is from
        result: Result status (success, failure, partial, cancelled, timeout)
        summary: Human-readable summary of what happened
        strategies_used: List of strategies/approaches attempted
        successful_patterns: Patterns that worked well
        failed_patterns: Patterns that didn't work
        duration_seconds: How long the workflow took
        node_count: Number of nodes executed
        error_message: Error details if failed
        tenant_id: Multi-tenant isolation identifier
        metadata: Additional outcome metadata

    Returns:
        Dict with workflow outcome info
    """
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not workflow_id or not workflow_id.strip():
        return {"success": False, "error": "workflow_id cannot be empty"}
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not result or not result.strip():
        return {"success": False, "error": "result cannot be empty"}
    if not summary or not summary.strip():
        return {"success": False, "error": "summary cannot be empty"}

    valid_results = ["success", "failure", "partial", "cancelled", "timeout"]
    if result not in valid_results:
        return {
            "success": False,
            "error": f"result must be one of: {', '.join(valid_results)}",
        }

    try:
        outcome = alma.learn_from_workflow(
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            summary=summary,
            strategies_used=strategies_used,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return {
            "success": True,
            "outcome": {
                "id": outcome.id,
                "workflow_id": outcome.workflow_id,
                "run_id": outcome.run_id,
                "result": outcome.result.value,
                "agent": outcome.agent,
                "created_at": outcome.created_at.isoformat(),
            },
            "message": "Workflow outcome recorded successfully",
        }

    except Exception as e:
        logger.exception(f"Error in alma_workflow_learn: {e}")
        return {"success": False, "error": str(e)}


def alma_link_artifact(
    alma: ALMA,
    memory_id: str,
    artifact_type: str,
    storage_url: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Link an external artifact to a memory.

    Args:
        alma: ALMA instance
        memory_id: The memory to link the artifact to
        artifact_type: Type (screenshot, log, report, file, etc.)
        storage_url: URL or path to the artifact in storage
        filename: Original filename
        mime_type: MIME type
        size_bytes: Size in bytes
        checksum: SHA256 checksum for integrity
        metadata: Additional artifact metadata

    Returns:
        Dict with artifact reference info
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}
    if not artifact_type or not artifact_type.strip():
        return {"success": False, "error": "artifact_type cannot be empty"}
    if not storage_url or not storage_url.strip():
        return {"success": False, "error": "storage_url cannot be empty"}

    try:
        artifact = alma.link_artifact(
            memory_id=memory_id,
            artifact_type=artifact_type,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata,
        )

        return {
            "success": True,
            "artifact": {
                "id": artifact.id,
                "memory_id": artifact.memory_id,
                "artifact_type": artifact.artifact_type.value,
                "storage_url": artifact.storage_url,
                "filename": artifact.filename,
                "created_at": artifact.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_link_artifact: {e}")
        return {"success": False, "error": str(e)}


def alma_get_artifacts(
    alma: ALMA,
    memory_id: str,
) -> Dict[str, Any]:
    """
    Get all artifacts linked to a memory.

    Args:
        alma: ALMA instance
        memory_id: The memory to get artifacts for

    Returns:
        Dict with list of artifact references
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}

    try:
        artifacts = alma.get_artifacts(memory_id)

        return {
            "success": True,
            "artifacts": [
                {
                    "id": a.id,
                    "artifact_type": a.artifact_type.value,
                    "storage_url": a.storage_url,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size_bytes": a.size_bytes,
                    "created_at": a.created_at.isoformat(),
                }
                for a in artifacts
            ],
            "count": len(artifacts),
        }

    except Exception as e:
        logger.exception(f"Error in alma_get_artifacts: {e}")
        return {"success": False, "error": str(e)}


def alma_cleanup_checkpoints(
    alma: ALMA,
    run_id: str,
    keep_latest: int = 1,
) -> Dict[str, Any]:
    """
    Clean up old checkpoints for a completed run.

    Args:
        alma: ALMA instance
        run_id: The workflow run identifier
        keep_latest: Number of latest checkpoints to keep

    Returns:
        Dict with cleanup results
    """
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        count = alma.cleanup_checkpoints(run_id, keep_latest)

        return {
            "success": True,
            "deleted_count": count,
            "kept": keep_latest,
            "message": f"Deleted {count} checkpoints, kept {keep_latest}",
        }

    except Exception as e:
        logger.exception(f"Error in alma_cleanup_checkpoints: {e}")
        return {"success": False, "error": str(e)}


def alma_retrieve_scoped(
    alma: ALMA,
    task: str,
    agent: str,
    scope: str = "agent",
    tenant_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    run_id: Optional[str] = None,
    node_id: Optional[str] = None,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve memories with workflow scope filtering.

    Args:
        alma: ALMA instance
        task: Description of the task to perform
        agent: Name of the agent requesting memories
        scope: Scope level (node, run, workflow, agent, tenant, global)
        tenant_id: Tenant identifier for multi-tenant
        workflow_id: Workflow definition identifier
        run_id: Specific run identifier
        node_id: Current node identifier
        user_id: Optional user ID for preferences
        top_k: Maximum items per memory type

    Returns:
        Dict with scoped memory slice
    """
    if not task or not task.strip():
        return {"success": False, "error": "task cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    valid_scopes = ["node", "run", "workflow", "agent", "tenant", "global"]
    if scope not in valid_scopes:
        return {
            "success": False,
            "error": f"scope must be one of: {', '.join(valid_scopes)}",
        }

    try:
        from alma.workflow import RetrievalScope, WorkflowContext

        context = WorkflowContext(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            run_id=run_id,
            node_id=node_id,
        )

        memories = alma.retrieve_with_scope(
            task=task,
            agent=agent,
            context=context,
            scope=RetrievalScope(scope),
            user_id=user_id,
            top_k=top_k,
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "scope": scope,
            "scope_filter": memories.metadata.get("scope_filter", {}),
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_scoped: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ASYNC WORKFLOW MCP TOOLS
# =============================================================================


async def async_alma_checkpoint(
    alma: ALMA,
    run_id: str,
    node_id: str,
    state: Dict[str, Any],
    branch_id: Optional[str] = None,
    parent_checkpoint_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    skip_if_unchanged: bool = True,
) -> Dict[str, Any]:
    """Async version of alma_checkpoint."""
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not node_id or not node_id.strip():
        return {"success": False, "error": "node_id cannot be empty"}

    try:
        checkpoint = await alma.async_checkpoint(
            run_id=run_id,
            node_id=node_id,
            state=state,
            branch_id=branch_id,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
        )

        if checkpoint is None:
            return {
                "success": True,
                "skipped": True,
                "message": "Checkpoint skipped - state unchanged",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "state_hash": checkpoint.state_hash,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_checkpoint: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_resume(
    alma: ALMA,
    run_id: str,
    branch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of alma_resume."""
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}

    try:
        checkpoint = await alma.async_get_resume_point(run_id, branch_id)

        if checkpoint is None:
            return {
                "success": True,
                "checkpoint": None,
                "message": "No checkpoint found for this run",
            }

        return {
            "success": True,
            "checkpoint": {
                "id": checkpoint.id,
                "run_id": checkpoint.run_id,
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "sequence_number": checkpoint.sequence_number,
                "branch_id": checkpoint.branch_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "created_at": checkpoint.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_resume: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_workflow_learn(
    alma: ALMA,
    agent: str,
    workflow_id: str,
    run_id: str,
    result: str,
    summary: str,
    strategies_used: Optional[list] = None,
    successful_patterns: Optional[list] = None,
    failed_patterns: Optional[list] = None,
    duration_seconds: Optional[float] = None,
    node_count: Optional[int] = None,
    error_message: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of alma_workflow_learn."""
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not workflow_id or not workflow_id.strip():
        return {"success": False, "error": "workflow_id cannot be empty"}
    if not run_id or not run_id.strip():
        return {"success": False, "error": "run_id cannot be empty"}
    if not result or not result.strip():
        return {"success": False, "error": "result cannot be empty"}
    if not summary or not summary.strip():
        return {"success": False, "error": "summary cannot be empty"}

    valid_results = ["success", "failure", "partial", "cancelled", "timeout"]
    if result not in valid_results:
        return {
            "success": False,
            "error": f"result must be one of: {', '.join(valid_results)}",
        }

    try:
        outcome = await alma.async_learn_from_workflow(
            agent=agent,
            workflow_id=workflow_id,
            run_id=run_id,
            result=result,
            summary=summary,
            strategies_used=strategies_used,
            successful_patterns=successful_patterns,
            failed_patterns=failed_patterns,
            duration_seconds=duration_seconds,
            node_count=node_count,
            error_message=error_message,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return {
            "success": True,
            "outcome": {
                "id": outcome.id,
                "workflow_id": outcome.workflow_id,
                "run_id": outcome.run_id,
                "result": outcome.result.value,
                "agent": outcome.agent,
                "created_at": outcome.created_at.isoformat(),
            },
            "message": "Workflow outcome recorded successfully",
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_workflow_learn: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_link_artifact(
    alma: ALMA,
    memory_id: str,
    artifact_type: str,
    storage_url: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version of alma_link_artifact."""
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}
    if not artifact_type or not artifact_type.strip():
        return {"success": False, "error": "artifact_type cannot be empty"}
    if not storage_url or not storage_url.strip():
        return {"success": False, "error": "storage_url cannot be empty"}

    try:
        artifact = await alma.async_link_artifact(
            memory_id=memory_id,
            artifact_type=artifact_type,
            storage_url=storage_url,
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata,
        )

        return {
            "success": True,
            "artifact": {
                "id": artifact.id,
                "memory_id": artifact.memory_id,
                "artifact_type": artifact.artifact_type.value,
                "storage_url": artifact.storage_url,
                "filename": artifact.filename,
                "created_at": artifact.created_at.isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_link_artifact: {e}")
        return {"success": False, "error": str(e)}


async def async_alma_retrieve_scoped(
    alma: ALMA,
    task: str,
    agent: str,
    scope: str = "agent",
    tenant_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    run_id: Optional[str] = None,
    node_id: Optional[str] = None,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Async version of alma_retrieve_scoped."""
    if not task or not task.strip():
        return {"success": False, "error": "task cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    valid_scopes = ["node", "run", "workflow", "agent", "tenant", "global"]
    if scope not in valid_scopes:
        return {
            "success": False,
            "error": f"scope must be one of: {', '.join(valid_scopes)}",
        }

    try:
        from alma.workflow import RetrievalScope, WorkflowContext

        context = WorkflowContext(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            run_id=run_id,
            node_id=node_id,
        )

        memories = await alma.async_retrieve_with_scope(
            task=task,
            agent=agent,
            context=context,
            scope=RetrievalScope(scope),
            user_id=user_id,
            top_k=top_k,
        )

        return {
            "success": True,
            "memories": _serialize_memory_slice(memories),
            "prompt_injection": memories.to_prompt(),
            "scope": scope,
            "scope_filter": memories.metadata.get("scope_filter", {}),
        }

    except Exception as e:
        logger.exception(f"Error in async_alma_retrieve_scoped: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MEMORY WALL ENHANCEMENT TOOLS (v0.7.0)
# =============================================================================
#
# Tools for Memory Wall enhancements: decay management, verified retrieval,
# and compression pipeline.


def alma_reinforce(
    alma: ALMA,
    memory_id: str,
    memory_type: str = "unknown",
) -> Dict[str, Any]:
    """
    Reinforce a memory to prevent it from being forgotten.

    Use this when:
    - A memory proved valuable and should be preserved
    - You want to strengthen a weak memory
    - You're reviewing memories and want to keep one active

    Args:
        alma: ALMA instance
        memory_id: ID of the memory to reinforce
        memory_type: Type of memory (heuristic, outcome, knowledge, etc.)

    Returns:
        Dict with new strength score (0.0-1.0). Higher is stronger.
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}

    try:
        from alma.learning.decay import DecayManager

        decay_manager = DecayManager(alma.storage)
        new_strength = decay_manager.reinforce_memory(memory_id, memory_type)
        strength_obj = decay_manager.get_strength(memory_id, memory_type)

        return {
            "success": True,
            "memory_id": memory_id,
            "new_strength": round(new_strength, 3),
            "access_count": strength_obj.access_count,
            "reinforcement_count": len(strength_obj.reinforcement_events),
            "status": "reinforced",
        }

    except Exception as e:
        logger.exception(f"Error in alma_reinforce: {e}")
        return {"success": False, "error": str(e)}


def alma_get_weak_memories(
    alma: ALMA,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    include_forgettable: bool = False,
) -> Dict[str, Any]:
    """
    List memories that are weak and may be forgotten soon.

    Returns memories in "recoverable" state (strength 0.1-0.3).
    These can be reinforced to save them, or left to naturally decay.

    Use this for periodic memory health checks.

    Args:
        alma: ALMA instance
        project_id: Project to check (defaults to ALMA's project)
        agent: Specific agent to check
        include_forgettable: Also include memories ready to forget (strength < 0.1)

    Returns:
        Dict with list of weak memories and counts
    """
    try:
        from alma.learning.decay import DecayManager

        decay_manager = DecayManager(alma.storage)
        pid = project_id or alma.project_id

        weak = decay_manager.get_weak_memories(project_id=pid, agent=agent)

        result: Dict[str, Any] = {
            "success": True,
            "weak_memories": [
                {"memory_id": mid, "memory_type": mtype, "strength": round(strength, 3)}
                for mid, mtype, strength in weak
            ],
            "count": len(weak),
        }

        if include_forgettable:
            forgettable = decay_manager.get_forgettable_memories(
                project_id=pid, agent=agent
            )
            result["forgettable"] = [
                {"memory_id": mid, "memory_type": mtype, "strength": round(strength, 3)}
                for mid, mtype, strength in forgettable
            ]
            result["forgettable_count"] = len(forgettable)

        return result

    except Exception as e:
        logger.exception(f"Error in alma_get_weak_memories: {e}")
        return {"success": False, "error": str(e)}


def alma_smart_forget(
    alma: ALMA,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    threshold: float = 0.1,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Trigger intelligent forgetting of weak memories.

    Memories below the strength threshold are:
    1. Archived (preserved for recovery if needed)
    2. Removed from active memory

    Use this for periodic cleanup or when memory gets noisy.
    Archives are kept for compliance/recovery.

    Args:
        alma: ALMA instance
        project_id: Project to clean (defaults to ALMA's project)
        agent: Specific agent to clean
        threshold: Strength threshold below which to forget (default 0.1)
        dry_run: If True, show what would be forgotten without doing it

    Returns:
        Dict with forgotten memories or preview
    """
    if not 0.0 <= threshold <= 1.0:
        return {
            "success": False,
            "error": "threshold must be between 0.0 and 1.0",
        }

    try:
        from alma.learning.decay import DecayManager

        decay_manager = DecayManager(alma.storage)
        pid = project_id or alma.project_id

        result = decay_manager.smart_forget(
            project_id=pid,
            agent=agent,
            threshold=threshold,
            archive=True,
            dry_run=dry_run,
        )

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "would_forget": result.get("would_forget", []),
                "count": result.get("count", 0),
                "message": f"Would forget {result.get('count', 0)} memories. "
                "Run with dry_run=False to execute.",
            }

        return {
            "success": True,
            "dry_run": False,
            "forgotten": result.get("forgotten", []),
            "archived": result.get("archived", []),
            "count": result.get("count", 0),
            "message": f"Forgot {result.get('count', 0)} memories (archived for recovery)",
        }

    except Exception as e:
        logger.exception(f"Error in alma_smart_forget: {e}")
        return {"success": False, "error": str(e)}


def alma_retrieve_verified(
    alma: ALMA,
    query: str,
    agent: str,
    project_id: Optional[str] = None,
    ground_truth: Optional[list] = None,
    cross_verify: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve memories with verification status.

    Two-stage retrieval:
    1. Semantic search for candidates
    2. Verify each candidate

    Results are categorized:
    - verified: Safe to use
    - uncertain: Use with caution
    - contradicted: Needs review (may be stale)

    Optionally provide ground_truth sources for fact-checking.

    Args:
        alma: ALMA instance
        query: Search query
        agent: Agent requesting retrieval
        project_id: Project context (defaults to ALMA's project)
        ground_truth: Optional list of authoritative source texts for verification
        cross_verify: Whether to cross-verify against other memories
        top_k: Number of results to return

    Returns:
        Dict with categorized verification results
    """
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    try:
        from alma.retrieval.verification import VerifiedRetriever

        pid = project_id or alma.project_id

        retriever = VerifiedRetriever(
            retrieval_engine=alma.retrieval,
            llm_client=getattr(alma, "llm", None),
        )

        results = retriever.retrieve_verified(
            query=query,
            agent=agent,
            project_id=pid,
            ground_truth_sources=ground_truth,
            cross_verify=cross_verify,
            top_k=top_k,
        )

        def serialize_verified_memory(vm):
            """Serialize a VerifiedMemory object."""
            memory_dict = {}
            memory = vm.memory
            # Handle different memory types
            if hasattr(memory, "to_dict"):
                memory_dict = memory.to_dict()
            elif hasattr(memory, "__dict__"):
                memory_dict = {
                    k: v for k, v in memory.__dict__.items() if not k.startswith("_")
                }
            else:
                memory_dict = {"content": str(memory)}

            return {
                "memory": memory_dict,
                "confidence": round(vm.verification.confidence, 3),
                "reason": vm.verification.reason,
                "retrieval_score": round(vm.retrieval_score, 3),
            }

        return {
            "success": True,
            "verified": [serialize_verified_memory(vm) for vm in results.verified],
            "uncertain": [serialize_verified_memory(vm) for vm in results.uncertain],
            "contradicted": [
                {
                    **serialize_verified_memory(vm),
                    "contradiction": vm.verification.contradicting_source,
                }
                for vm in results.contradicted
            ],
            "unverifiable": [
                serialize_verified_memory(vm) for vm in results.unverifiable
            ],
            "summary": results.summary(),
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_verified: {e}")
        return {"success": False, "error": str(e)}


def alma_compress_and_learn(
    alma: ALMA,
    content: str,
    agent: str,
    memory_type: str = "outcome",
    compression_level: str = "medium",
    project_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compress verbose content and store as memory.

    Extracts key facts, constraints, and patterns from lengthy content.
    Achieves 3-5x compression while preserving essential information.

    Use this instead of alma_learn when storing verbose task outcomes
    or long documents.

    Args:
        alma: ALMA instance
        content: Verbose content to compress and store
        agent: Agent storing the memory
        memory_type: Type of memory (outcome, heuristic, knowledge)
        compression_level: Compression level (light, medium, aggressive)
        project_id: Project context
        task_type: Optional task type categorization

    Returns:
        Dict with memory ID and compression stats
    """
    if not content or not content.strip():
        return {"success": False, "error": "content cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    valid_levels = ["light", "medium", "aggressive"]
    if compression_level not in valid_levels:
        return {
            "success": False,
            "error": f"compression_level must be one of: {', '.join(valid_levels)}",
        }

    valid_types = ["outcome", "heuristic", "knowledge"]
    if memory_type not in valid_types:
        return {
            "success": False,
            "error": f"memory_type must be one of: {', '.join(valid_types)}",
        }

    try:
        from alma.compression.pipeline import CompressionLevel, MemoryCompressor

        pid = project_id or alma.project_id
        level = CompressionLevel(compression_level)

        # Create compressor with optional LLM
        compressor = MemoryCompressor(llm_client=getattr(alma, "llm", None))

        # Compress the content
        compressed = compressor.compress_outcome(content, level)

        # Store based on memory type
        now = datetime.now(timezone.utc)
        metadata = compressed.to_metadata()

        if memory_type == "outcome":
            from alma.types import Outcome

            outcome = Outcome(
                id=f"out-compressed-{now.strftime('%Y%m%d%H%M%S')}",
                agent=agent,
                project_id=pid,
                task_type=task_type or "compressed",
                task_description=compressed.summary,
                success=True,
                strategy_used="compressed learning",
                timestamp=now,
                metadata=metadata,
            )
            alma.storage.save_outcome(outcome)
            memory_id = outcome.id

        elif memory_type == "heuristic":
            from alma.types import Heuristic

            # Extract pattern from key facts if available
            condition = compressed.key_facts[0] if compressed.key_facts else "general"
            strategy = compressed.summary

            heuristic = Heuristic(
                id=f"heur-compressed-{now.strftime('%Y%m%d%H%M%S')}",
                agent=agent,
                project_id=pid,
                condition=condition,
                strategy=strategy,
                confidence=0.7,
                occurrence_count=1,
                success_count=1,
                last_validated=now,
                created_at=now,
                metadata=metadata,
            )
            alma.storage.save_heuristic(heuristic)
            memory_id = heuristic.id

        else:  # knowledge
            from alma.types import DomainKnowledge

            knowledge = DomainKnowledge(
                id=f"dk-compressed-{now.strftime('%Y%m%d%H%M%S')}",
                agent=agent,
                project_id=pid,
                domain=task_type or "general",
                fact=compressed.summary,
                source="compressed_learning",
                confidence=0.8,
                last_verified=now,
                metadata=metadata,
            )
            alma.storage.save_domain_knowledge(knowledge)
            memory_id = knowledge.id

        return {
            "success": True,
            "memory_id": memory_id,
            "memory_type": memory_type,
            "compression_ratio": round(compressed.compression_ratio, 2),
            "original_length": compressed.original_length,
            "compressed_length": compressed.compressed_length,
            "key_facts": compressed.key_facts,
            "constraints": compressed.constraints,
            "summary_preview": (
                compressed.summary[:200] + "..."
                if len(compressed.summary) > 200
                else compressed.summary
            ),
        }

    except Exception as e:
        logger.exception(f"Error in alma_compress_and_learn: {e}")
        return {"success": False, "error": str(e)}


def alma_extract_heuristic(
    alma: ALMA,
    experiences: list,
    agent: str,
    project_id: Optional[str] = None,
    auto_save: bool = True,
) -> Dict[str, Any]:
    """
    Extract a general rule from multiple similar experiences.

    Provide 3+ similar experiences and this tool will identify patterns
    and create a reusable heuristic rule.

    Example: Pass 3 debugging experiences and get:
    "When tests fail with timeout errors, check for async operations
    that aren't being awaited."

    Returns null if no clear pattern found.

    Args:
        alma: ALMA instance
        experiences: List of 3+ similar experience descriptions
        agent: Agent to attribute the heuristic to
        project_id: Project context
        auto_save: If True, automatically save extracted heuristic

    Returns:
        Dict with extracted heuristic or null if no pattern found
    """
    if not experiences or len(experiences) < 3:
        return {
            "success": False,
            "error": "Need at least 3 experiences to extract a pattern",
            "provided": len(experiences) if experiences else 0,
        }
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    try:
        from alma.compression.pipeline import MemoryCompressor

        pid = project_id or alma.project_id

        compressor = MemoryCompressor(llm_client=getattr(alma, "llm", None))
        heuristic_text = compressor.extract_heuristic(experiences)

        if not heuristic_text:
            return {
                "success": True,
                "heuristic": None,
                "message": "No clear pattern found in these experiences",
            }

        result: Dict[str, Any] = {
            "success": True,
            "heuristic": heuristic_text,
            "source_count": len(experiences),
        }

        if auto_save:
            from alma.types import Heuristic

            now = datetime.now(timezone.utc)
            heuristic = Heuristic(
                id=f"heur-extracted-{now.strftime('%Y%m%d%H%M%S')}",
                agent=agent,
                project_id=pid,
                condition="extracted from experiences",
                strategy=heuristic_text,
                confidence=0.7,
                occurrence_count=len(experiences),
                success_count=len(experiences),
                last_validated=now,
                created_at=now,
                metadata={"extracted_from_count": len(experiences)},
            )
            alma.storage.save_heuristic(heuristic)
            result["saved"] = True
            result["memory_id"] = heuristic.id
        else:
            result["saved"] = False

        return result

    except Exception as e:
        logger.exception(f"Error in alma_extract_heuristic: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ASYNC MEMORY WALL ENHANCEMENT TOOLS
# =============================================================================


async def async_alma_reinforce(
    alma: ALMA,
    memory_id: str,
    memory_type: str = "unknown",
) -> Dict[str, Any]:
    """Async version of alma_reinforce."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_reinforce(alma, memory_id, memory_type),
    )


async def async_alma_get_weak_memories(
    alma: ALMA,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    include_forgettable: bool = False,
) -> Dict[str, Any]:
    """Async version of alma_get_weak_memories."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_get_weak_memories(alma, project_id, agent, include_forgettable),
    )


async def async_alma_smart_forget(
    alma: ALMA,
    project_id: Optional[str] = None,
    agent: Optional[str] = None,
    threshold: float = 0.1,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Async version of alma_smart_forget."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_smart_forget(alma, project_id, agent, threshold, dry_run),
    )


async def async_alma_retrieve_verified(
    alma: ALMA,
    query: str,
    agent: str,
    project_id: Optional[str] = None,
    ground_truth: Optional[list] = None,
    cross_verify: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Async version of alma_retrieve_verified."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_retrieve_verified(
            alma, query, agent, project_id, ground_truth, cross_verify, top_k
        ),
    )


async def async_alma_compress_and_learn(
    alma: ALMA,
    content: str,
    agent: str,
    memory_type: str = "outcome",
    compression_level: str = "medium",
    project_id: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of alma_compress_and_learn."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_compress_and_learn(
            alma, content, agent, memory_type, compression_level, project_id, task_type
        ),
    )


async def async_alma_extract_heuristic(
    alma: ALMA,
    experiences: list,
    agent: str,
    project_id: Optional[str] = None,
    auto_save: bool = True,
) -> Dict[str, Any]:
    """Async version of alma_extract_heuristic."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_extract_heuristic(alma, experiences, agent, project_id, auto_save),
    )


# =============================================================================
# TRUST-INTEGRATED RETRIEVAL TOOLS (v0.8.0)
# =============================================================================
#
# Tools for trust-aware scoring, token budget management, and progressive
# disclosure for context engineering.


def alma_retrieve_with_trust(
    alma: ALMA,
    query: str,
    agent: str,
    requesting_agent_id: str,
    requesting_agent_trust: float = 0.5,
    trust_behaviors: Optional[Dict[str, float]] = None,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve memories with trust-integrated scoring.

    Adjusts retrieval based on the requesting agent's trust profile.
    Higher trust agents get access to more sensitive memories and
    higher scores for high-confidence heuristics.

    Anti-patterns and warnings are prioritized for lower-trust agents.

    Args:
        alma: ALMA instance
        query: Search query
        agent: Agent whose memories to search
        requesting_agent_id: ID of agent making the request
        requesting_agent_trust: Trust score (0.0-1.0) of requesting agent
        trust_behaviors: Optional per-behavior trust scores
        user_id: Optional user ID for preferences
        top_k: Number of results per type

    Returns:
        Dict with trust-scored memories and trust context
    """
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if not 0.0 <= requesting_agent_trust <= 1.0:
        return {"success": False, "error": "requesting_agent_trust must be 0.0-1.0"}

    try:
        from alma.retrieval.trust_scoring import AgentTrustContext, TrustAwareScorer

        # Build trust context
        trust_context = AgentTrustContext(
            agent_id=requesting_agent_id,
            trust_score=requesting_agent_trust,
            trust_behaviors=trust_behaviors or {},
        )

        # Create trust-aware scorer
        scorer = TrustAwareScorer(embedder=alma.retrieval.embedder)

        # Get base memories
        memories = alma.retrieve(
            task=query,
            agent=agent,
            user_id=user_id,
            top_k=top_k * 2,  # Get more, then filter by trust
        )

        # Apply trust scoring to heuristics
        if memories.heuristics:
            query_embedding = scorer.embedder.embed(query)
            heuristic_embeddings = [
                scorer.embedder.embed(h.condition + " " + h.strategy)
                for h in memories.heuristics
            ]
            similarities = [
                scorer._cosine_similarity(query_embedding, e)
                for e in heuristic_embeddings
            ]

            trust_scored = scorer.score_heuristics_with_trust(
                memories.heuristics, similarities, trust_context
            )

            # Sort by trust-adjusted score and limit
            trust_scored.sort(key=lambda x: x.trust_adjusted_score, reverse=True)
            top_heuristics = trust_scored[:top_k]

            # Extract trust info
            heuristic_results = [
                {
                    "id": ts.item.id,
                    "condition": ts.item.condition,
                    "strategy": ts.item.strategy,
                    "confidence": ts.item.confidence,
                    "base_score": round(ts.base_score, 3),
                    "trust_adjusted_score": round(ts.trust_adjusted_score, 3),
                    "trust_factor": round(ts.trust_factor, 3),
                    "trust_explanation": ts.trust_explanation,
                }
                for ts in top_heuristics
            ]
        else:
            heuristic_results = []

        # Apply trust scoring to anti-patterns (prioritize for low-trust)
        if memories.anti_patterns:
            ap_embeddings = [
                scorer.embedder.embed(ap.pattern + " " + ap.why_bad)
                for ap in memories.anti_patterns
            ]
            ap_similarities = [
                scorer._cosine_similarity(query_embedding, e) for e in ap_embeddings
            ]

            trust_scored_ap = scorer.score_anti_patterns_with_trust(
                memories.anti_patterns, ap_similarities
            )
            trust_scored_ap.sort(key=lambda x: x.trust_adjusted_score, reverse=True)
            top_anti_patterns = trust_scored_ap[:top_k]

            anti_pattern_results = [
                {
                    "id": ts.item.id,
                    "pattern": ts.item.pattern,
                    "why_bad": ts.item.why_bad,
                    "better_alternative": ts.item.better_alternative,
                    "base_score": round(ts.base_score, 3),
                    "trust_adjusted_score": round(ts.trust_adjusted_score, 3),
                }
                for ts in top_anti_patterns
            ]
        else:
            anti_pattern_results = []

        # Trust-based retrieval summary
        trust_level = (
            "HIGH"
            if requesting_agent_trust >= 0.7
            else "MODERATE"
            if requesting_agent_trust >= 0.5
            else "LOW"
        )

        return {
            "success": True,
            "heuristics": heuristic_results,
            "anti_patterns": anti_pattern_results,
            "outcomes": _serialize_memory_slice(memories).get("outcomes", [])[:top_k],
            "trust_context": {
                "requesting_agent": requesting_agent_id,
                "trust_score": requesting_agent_trust,
                "trust_level": trust_level,
                "anti_patterns_prioritized": requesting_agent_trust < 0.5,
            },
            "retrieval_time_ms": memories.retrieval_time_ms,
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_with_trust: {e}")
        return {"success": False, "error": str(e)}


def alma_retrieve_with_budget(
    alma: ALMA,
    query: str,
    agent: str,
    max_tokens: int = 4000,
    must_see_types: Optional[list] = None,
    should_see_types: Optional[list] = None,
    user_id: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve memories within a token budget.

    Prioritizes memories based on token budget constraints:
    - MUST_SEE: Always included (anti-patterns, critical warnings)
    - SHOULD_SEE: Included if budget allows (heuristics, outcomes)
    - FETCH_ON_DEMAND: Summaries only, full content on request
    - EXCLUDE: Not included in this retrieval

    Returns a budget report showing what was included/excluded.

    Args:
        alma: ALMA instance
        query: Search query
        agent: Agent whose memories to search
        max_tokens: Maximum token budget for context
        must_see_types: Memory types that must be included
        should_see_types: Memory types to include if space allows
        user_id: Optional user ID for preferences
        top_k: Maximum items per type before budget filtering

    Returns:
        Dict with budgeted memories and allocation report
    """
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}
    if max_tokens < 100:
        return {"success": False, "error": "max_tokens must be at least 100"}

    try:
        from alma.retrieval.budget import PriorityTier, RetrievalBudget

        # Get all memories
        memories = alma.retrieve(
            task=query,
            agent=agent,
            user_id=user_id,
            top_k=top_k,
        )

        # Create budget manager
        budget = RetrievalBudget(max_tokens=max_tokens)

        # Default priority mapping
        default_must_see = must_see_types or ["anti_patterns"]
        default_should_see = should_see_types or ["heuristics", "outcomes"]

        type_priorities = {}
        for t in default_must_see:
            type_priorities[t] = PriorityTier.MUST_SEE
        for t in default_should_see:
            type_priorities[t] = PriorityTier.SHOULD_SEE
        # Domain knowledge and preferences default to FETCH_ON_DEMAND
        type_priorities.setdefault("domain_knowledge", PriorityTier.FETCH_ON_DEMAND)
        type_priorities.setdefault("preferences", PriorityTier.FETCH_ON_DEMAND)

        # Apply budget
        budgeted_slice, report = budget.apply_budget(memories, type_priorities)

        # Serialize budgeted results
        result = _serialize_memory_slice(budgeted_slice)

        return {
            "success": True,
            "memories": result,
            "prompt_injection": budgeted_slice.to_prompt(),
            "budget_report": {
                "max_tokens": max_tokens,
                "tokens_used": report.tokens_used,
                "tokens_remaining": report.tokens_remaining,
                "utilization": round(report.utilization, 3),
                "included_count": report.included_count,
                "excluded_count": report.excluded_count,
                "by_priority": {
                    tier.value: {
                        "included": count,
                        "tokens": report.tokens_by_priority.get(tier, 0),
                    }
                    for tier, count in report.included_by_priority.items()
                },
                "excluded_items": [
                    {
                        "type": item.memory_type,
                        "id": item.item.id,
                        "priority": item.priority.value,
                    }
                    for item in report.excluded_items[:5]  # Show first 5 excluded
                ],
            },
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_with_budget: {e}")
        return {"success": False, "error": str(e)}


def alma_retrieve_progressive(
    alma: ALMA,
    query: str,
    agent: str,
    disclosure_level: str = "summary",
    max_summaries: int = 10,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve memory summaries with lazy-loading of full content.

    Returns compact summaries first, with IDs to fetch full details
    on demand. Implements progressive disclosure for context efficiency.

    Disclosure levels:
    - REFERENCE: Just IDs and one-line descriptions
    - SUMMARY: Brief summaries with key info
    - KEY_DETAILS: Important details without full content
    - FULL: Complete memory content

    Args:
        alma: ALMA instance
        query: Search query
        agent: Agent whose memories to search
        disclosure_level: Level of detail (reference, summary, key_details, full)
        max_summaries: Maximum number of summaries to return
        user_id: Optional user ID for preferences

    Returns:
        Dict with memory summaries and fetch instructions
    """
    if not query or not query.strip():
        return {"success": False, "error": "query cannot be empty"}
    if not agent or not agent.strip():
        return {"success": False, "error": "agent cannot be empty"}

    valid_levels = ["reference", "summary", "key_details", "full"]
    if disclosure_level not in valid_levels:
        return {
            "success": False,
            "error": f"disclosure_level must be one of: {', '.join(valid_levels)}",
        }

    try:
        from alma.retrieval.progressive import DisclosureLevel, ProgressiveRetrieval

        level = DisclosureLevel(disclosure_level)

        # Create progressive retriever
        progressive = ProgressiveRetrieval(
            retrieval_engine=alma.retrieval,
            storage=alma.storage,
        )

        # Get summaries
        progressive_slice = progressive.retrieve_summaries(
            query=query,
            agent=agent,
            project_id=alma.project_id,
            disclosure_level=level,
            max_items=max_summaries,
            user_id=user_id,
            scope=alma.scopes.get(agent),
        )

        # Format for context
        context_text = progressive.format_summaries_for_context(
            progressive_slice,
            include_fetch_hint=True,
        )

        # Build response
        summaries = []
        for summary in progressive_slice.summaries:
            summary_dict = {
                "memory_id": summary.memory_id,
                "memory_type": summary.memory_type,
                "one_liner": summary.one_liner,
                "relevance_score": round(summary.relevance_score, 3),
            }
            if level in [
                DisclosureLevel.SUMMARY,
                DisclosureLevel.KEY_DETAILS,
                DisclosureLevel.FULL,
            ]:
                summary_dict["summary"] = summary.summary
            if level in [DisclosureLevel.KEY_DETAILS, DisclosureLevel.FULL]:
                summary_dict["key_details"] = summary.key_details
            summaries.append(summary_dict)

        return {
            "success": True,
            "disclosure_level": disclosure_level,
            "summaries": summaries,
            "context_injection": context_text,
            "total_available": progressive_slice.total_available,
            "can_fetch_more": progressive_slice.total_available > len(summaries),
            "fetch_instructions": (
                "Use alma_get_memory_full to retrieve complete content for any memory_id"
                if level != "full"
                else None
            ),
        }

    except Exception as e:
        logger.exception(f"Error in alma_retrieve_progressive: {e}")
        return {"success": False, "error": str(e)}


def alma_get_memory_full(
    alma: ALMA,
    memory_id: str,
    memory_type: str,
) -> Dict[str, Any]:
    """
    Get full content of a specific memory.

    Use this to fetch complete details after progressive retrieval
    returns summaries. Supports lazy-loading pattern.

    Args:
        alma: ALMA instance
        memory_id: ID of the memory to fetch
        memory_type: Type (heuristic, outcome, domain_knowledge, anti_pattern, preference)

    Returns:
        Dict with full memory content
    """
    if not memory_id or not memory_id.strip():
        return {"success": False, "error": "memory_id cannot be empty"}
    if not memory_type or not memory_type.strip():
        return {"success": False, "error": "memory_type cannot be empty"}

    valid_types = [
        "heuristic",
        "outcome",
        "domain_knowledge",
        "anti_pattern",
        "preference",
    ]
    if memory_type not in valid_types:
        return {
            "success": False,
            "error": f"memory_type must be one of: {', '.join(valid_types)}",
        }

    try:
        from alma.retrieval.progressive import ProgressiveRetrieval

        progressive = ProgressiveRetrieval(
            retrieval_engine=alma.retrieval,
            storage=alma.storage,
        )

        memory = progressive.get_full_item(memory_id, memory_type)

        if memory is None:
            return {
                "success": False,
                "error": f"Memory not found: {memory_id}",
            }

        # Serialize based on type
        if memory_type == "heuristic":
            content = {
                "id": memory.id,
                "condition": memory.condition,
                "strategy": memory.strategy,
                "confidence": memory.confidence,
                "occurrence_count": memory.occurrence_count,
                "success_rate": getattr(memory, "success_rate", None),
                "last_validated": memory.last_validated.isoformat()
                if memory.last_validated
                else None,
            }
        elif memory_type == "outcome":
            content = {
                "id": memory.id,
                "task_type": memory.task_type,
                "task_description": memory.task_description,
                "success": memory.success,
                "strategy_used": memory.strategy_used,
                "duration_ms": memory.duration_ms,
                "error_message": getattr(memory, "error_message", None),
            }
        elif memory_type == "domain_knowledge":
            content = {
                "id": memory.id,
                "domain": memory.domain,
                "fact": memory.fact,
                "source": memory.source,
                "confidence": memory.confidence,
            }
        elif memory_type == "anti_pattern":
            content = {
                "id": memory.id,
                "pattern": memory.pattern,
                "why_bad": memory.why_bad,
                "better_alternative": memory.better_alternative,
                "severity": getattr(memory, "severity", None),
            }
        else:  # preference
            content = {
                "id": memory.id,
                "category": memory.category,
                "preference": memory.preference,
                "source": memory.source,
            }

        return {
            "success": True,
            "memory_type": memory_type,
            "content": content,
        }

    except Exception as e:
        logger.exception(f"Error in alma_get_memory_full: {e}")
        return {"success": False, "error": str(e)}


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
# ASYNC TRUST-INTEGRATED RETRIEVAL TOOLS
# =============================================================================


async def async_alma_retrieve_with_trust(
    alma: ALMA,
    query: str,
    agent: str,
    requesting_agent_id: str,
    requesting_agent_trust: float = 0.5,
    trust_behaviors: Optional[Dict[str, float]] = None,
    user_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Async version of alma_retrieve_with_trust."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_retrieve_with_trust(
            alma,
            query,
            agent,
            requesting_agent_id,
            requesting_agent_trust,
            trust_behaviors,
            user_id,
            top_k,
        ),
    )


async def async_alma_retrieve_with_budget(
    alma: ALMA,
    query: str,
    agent: str,
    max_tokens: int = 4000,
    must_see_types: Optional[list] = None,
    should_see_types: Optional[list] = None,
    user_id: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Async version of alma_retrieve_with_budget."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_retrieve_with_budget(
            alma,
            query,
            agent,
            max_tokens,
            must_see_types,
            should_see_types,
            user_id,
            top_k,
        ),
    )


async def async_alma_retrieve_progressive(
    alma: ALMA,
    query: str,
    agent: str,
    disclosure_level: str = "summary",
    max_summaries: int = 10,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Async version of alma_retrieve_progressive."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_retrieve_progressive(
            alma, query, agent, disclosure_level, max_summaries, user_id
        ),
    )


async def async_alma_get_memory_full(
    alma: ALMA,
    memory_id: str,
    memory_type: str,
) -> Dict[str, Any]:
    """Async version of alma_get_memory_full."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: alma_get_memory_full(alma, memory_id, memory_type),
    )


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
