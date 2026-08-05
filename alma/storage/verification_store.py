"""
Helpers to persist VerificationStatus (Atlas G1).

Works with SQLite (and any backend that implements the same methods).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from alma.types import MemoryType

logger = logging.getLogger(__name__)

TABLE_BY_TYPE = {
    MemoryType.HEURISTIC: "heuristics",
    MemoryType.OUTCOME: "outcomes",
    MemoryType.DOMAIN_KNOWLEDGE: "domain_knowledge",
    MemoryType.USER_PREFERENCE: "preferences",
    MemoryType.ANTI_PATTERN: "anti_patterns",
    "heuristic": "heuristics",
    "outcome": "outcomes",
    "domain_knowledge": "domain_knowledge",
    "user_preference": "preferences",
    "anti_pattern": "anti_patterns",
}


def infer_memory_type(memory: Any) -> Optional[str]:
    """Best-effort memory type from object class/name."""
    name = type(memory).__name__.lower()
    if "heuristic" in name:
        return MemoryType.HEURISTIC.value
    if "outcome" in name:
        return MemoryType.OUTCOME.value
    if "preference" in name or "userpreference" in name:
        return MemoryType.USER_PREFERENCE.value
    if "domain" in name or "knowledge" in name:
        return MemoryType.DOMAIN_KNOWLEDGE.value
    if "anti" in name:
        return MemoryType.ANTI_PATTERN.value
    # duck type fields
    if hasattr(memory, "strategy") and hasattr(memory, "condition"):
        return MemoryType.HEURISTIC.value
    if hasattr(memory, "task_description") and hasattr(memory, "success"):
        return MemoryType.OUTCOME.value
    if hasattr(memory, "fact"):
        return MemoryType.DOMAIN_KNOWLEDGE.value
    if hasattr(memory, "pattern") and hasattr(memory, "why_bad"):
        return MemoryType.ANTI_PATTERN.value
    if hasattr(memory, "preference"):
        return MemoryType.USER_PREFERENCE.value
    return None


def persist_verification(storage: Any, memory: Any, verification: Any) -> bool:
    """
    Persist verification fields for a memory if storage supports it.

    Returns True if a write was attempted successfully.
    """
    updater = getattr(storage, "update_memory_verification", None)
    if updater is None:
        return False

    mid = getattr(memory, "id", None)
    if not mid:
        return False
    mtype = infer_memory_type(memory)
    if not mtype:
        return False

    status = getattr(verification, "status", None)
    status_val = status.value if hasattr(status, "value") else str(status)
    method = getattr(verification, "method", None)
    method_val = method.value if hasattr(method, "value") else str(method or "none")

    try:
        updater(
            memory_type=mtype,
            memory_id=mid,
            verification_status=status_val,
            verification_method=method_val,
            verification_confidence=float(getattr(verification, "confidence", 0.0) or 0.0),
            verification_reason=str(getattr(verification, "reason", "") or ""),
            contradicting_source=getattr(verification, "contradicting_source", None),
            verified_at=datetime.now(timezone.utc).isoformat(),
        )
        return True
    except Exception as e:
        logger.warning("persist_verification failed id=%s: %s", mid, e)
        return False
