"""
ALMA MCP Learning Tools.

Provides sync and async learning tool functions for the MCP protocol.
Includes outcome recording, preferences, knowledge, forgetting,
consolidation, compression, and heuristic extraction.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from alma import ALMA

logger = logging.getLogger(__name__)


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
# ASYNC LEARNING TOOLS
# =============================================================================


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
