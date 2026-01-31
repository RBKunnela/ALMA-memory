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
