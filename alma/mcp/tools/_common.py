"""
Shared utilities for ALMA MCP tools.

Contains helper functions used across multiple tool modules.
"""

import logging
from typing import Any, Dict

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
