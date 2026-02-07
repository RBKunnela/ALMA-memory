"""
ALMA MCP Retrieval Tools.

Provides sync and async retrieval tool functions for the MCP protocol.
Includes standard, mode-aware, scoped, verified, trust-integrated,
budget-aware, and progressive retrieval.
"""

import logging
from typing import Any, Dict, Optional

from alma import ALMA
from alma.retrieval.modes import RetrievalMode

from ._common import _serialize_memory_slice

logger = logging.getLogger(__name__)


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
    - Queries with "error", "bug", "fix", "debug" -> DIAGNOSTIC mode
    - Queries with "how should", "options", "plan" -> BROAD mode
    - Queries with "what was", "remember when" -> RECALL mode
    - Queries with "pattern", "similar" -> LEARNING mode
    - Default -> PRECISE mode for implementation tasks

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


# =============================================================================
# ASYNC RETRIEVAL TOOLS
# =============================================================================


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
