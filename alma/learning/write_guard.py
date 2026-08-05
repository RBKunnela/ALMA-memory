"""
Anti-pattern write guard (Atlas G2 / Chefe 561 / Code-Hub 1624).

Before learning outcomes/heuristics/domain knowledge, check whether the
text matches a known anti-pattern for the project. Default: ON via env
ALMA_ANTI_PATTERN_WRITE_GUARD=1.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


def write_guard_enabled() -> bool:
    raw = os.environ.get("ALMA_ANTI_PATTERN_WRITE_GUARD", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


@dataclass
class WriteGuardResult:
    blocked: bool
    matched_pattern: Optional[str] = None
    anti_pattern_id: Optional[str] = None
    reason: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]{3,}", _normalize(text)))


def text_matches_anti_pattern(candidate: str, pattern: str, min_overlap: float = 0.45) -> bool:
    """True if candidate contains pattern or shares enough tokens."""
    c = _normalize(candidate)
    p = _normalize(pattern)
    if not c or not p:
        return False
    if p in c or c in p:
        return True
    ct, pt = _tokens(c), _tokens(p)
    if not pt:
        return False
    overlap = len(ct & pt) / max(len(pt), 1)
    return overlap >= min_overlap


def check_write_guard(
    storage: Any,
    project_id: str,
    agent: str,
    texts: Sequence[str],
) -> WriteGuardResult:
    """
    Return blocked=True if any text strongly matches a stored anti-pattern.

    Fail-open if storage cannot list anti-patterns (non-SQLite / missing method).
    """
    if not write_guard_enabled():
        return WriteGuardResult(blocked=False, reason="write_guard_disabled")

    getter = getattr(storage, "get_anti_patterns", None)
    if getter is None:
        return WriteGuardResult(blocked=False, reason="storage_has_no_anti_patterns")

    try:
        patterns: List[Any] = getter(project_id=project_id, agent=None, top_k=200) or []
    except TypeError:
        try:
            patterns = getter(project_id) or []
        except Exception as e:
            logger.warning("write_guard: get_anti_patterns failed: %s", e)
            return WriteGuardResult(blocked=False, reason=f"lookup_failed:{e}")
    except Exception as e:
        logger.warning("write_guard: get_anti_patterns failed: %s", e)
        return WriteGuardResult(blocked=False, reason=f"lookup_failed:{e}")

    joined = " ".join(t for t in texts if t)
    for ap in patterns:
        pattern = getattr(ap, "pattern", None) or (ap.get("pattern") if isinstance(ap, dict) else "")
        ap_id = getattr(ap, "id", None) or (ap.get("id") if isinstance(ap, dict) else None)
        if pattern and text_matches_anti_pattern(joined, pattern):
            why = getattr(ap, "why_bad", "") or ""
            logger.info(
                "write_guard blocked learn project=%s agent=%s anti_pattern=%s",
                project_id,
                agent,
                ap_id,
            )
            return WriteGuardResult(
                blocked=True,
                matched_pattern=pattern,
                anti_pattern_id=ap_id,
                reason=why or "matches_anti_pattern",
            )
    return WriteGuardResult(blocked=False, reason="ok")
