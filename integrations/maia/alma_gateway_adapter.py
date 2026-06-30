"""ALMA <-> Maia gateway adapter (Phase 1: facts-first read + write).

This module is self-contained and framework-agnostic. It exposes two JSON-RPC
handlers that the Maia gateway dispatcher (running on the user's VPS) wires into
its `memory.query` / `memory.store` methods, backing Maia's memory with ALMA.

Device contract (verified in the iMetaClaw/OpenGlasses app,
OpenClawBridge.swift:1172-1206):

    memory.query  params {query: str, limit: int}
                  -> {"ok": True, "payload": {"results": [{"content": str}, ...]}}
    memory.store  params {content: str, metadata: object}
                  -> {"ok": True}

Design notes:
- `alma` is passed in as a duck-typed dependency, so this module imports WITHOUT
  the `alma-memory` package installed. Only `build_alma()` imports `alma` lazily.
- Handlers NEVER raise into the RPC layer. Any failure is mapped to
  `{"ok": False, "error": "..."}` so the WebSocket dispatcher stays alive.
- Empty memory is a success with empty results, NOT an error.

Scope: Phase 1 only. Outcome/confidence/strategy enrichment via `alma.learn()`
is Phase 3 (see TODOs). RN parity / cross-device / HIPAA pruning is Phase 4.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["handle_memory_query", "handle_memory_store", "build_alma"]

# Default ALMA "domain" bucket for facts written through the gateway.
_DEFAULT_DOMAIN = "maia_memory"
# Default top_k when the device omits `limit`.
_DEFAULT_LIMIT = 10


def _slice_to_contents(memory_slice: Any) -> List[str]:
    """Flatten an ALMA MemorySlice into a list of human-readable strings.

    Phase 1 is facts-first, so domain-knowledge facts lead. We also surface the
    other readable memory types so a query is not silently empty when only
    heuristics / preferences exist for the agent.
    """
    contents: List[str] = []

    # Facts first (Phase 1 primary path).
    for dk in getattr(memory_slice, "domain_knowledge", None) or []:
        fact = getattr(dk, "fact", None)
        if fact:
            contents.append(str(fact))

    for h in getattr(memory_slice, "heuristics", None) or []:
        condition = getattr(h, "condition", "")
        strategy = getattr(h, "strategy", "")
        if condition or strategy:
            contents.append(f"When {condition}: {strategy}".strip())

    for pref in getattr(memory_slice, "preferences", None) or []:
        text = getattr(pref, "preference", None)
        if text:
            contents.append(str(text))

    for ap in getattr(memory_slice, "anti_patterns", None) or []:
        pattern = getattr(ap, "pattern", "")
        better = getattr(ap, "better_alternative", "")
        if pattern:
            contents.append(f"Avoid: {pattern}. Instead: {better}".strip())

    return contents


def handle_memory_query(
    params: Dict[str, Any],
    *,
    alma: Any,
    agent: str = "maia",
) -> Dict[str, Any]:
    """Handle a `memory.query` RPC call.

    Args:
        params: RPC params. Expects `query: str` and optional `limit: int`.
        alma: An ALMA instance (or compatible) exposing `.retrieve(...)`.
        agent: ALMA agent namespace to read from. May be overridden per call by
            `params["metadata"]["persona"]` when present.

    Returns:
        `{"ok": True, "payload": {"results": [{"content": str}, ...]}}` on
        success (empty results when memory is empty), or
        `{"ok": False, "error": str}` on any failure.
    """
    try:
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "missing or invalid 'query'"}

        limit = params.get("limit", _DEFAULT_LIMIT)
        if not isinstance(limit, int) or limit <= 0:
            limit = _DEFAULT_LIMIT

        resolved_agent = _resolve_agent(params.get("metadata"), agent)
        user_id = _resolve_user_id(params.get("metadata"))

        memory_slice = alma.retrieve(
            task=query,
            agent=resolved_agent,
            top_k=limit,
            **({"user_id": user_id} if user_id else {}),
        )

        results = [{"content": c} for c in _slice_to_contents(memory_slice)]
        return {"ok": True, "payload": {"results": results}}
    except Exception as exc:  # never raise into the RPC layer
        return {"ok": False, "error": f"memory.query failed: {exc}"}


def handle_memory_store(
    params: Dict[str, Any],
    *,
    alma: Any,
    agent: str = "maia",
) -> Dict[str, Any]:
    """Handle a `memory.store` RPC call.

    Stores the content as ALMA domain knowledge (a fact). The `persona` in
    metadata, when present, selects the ALMA agent/scope; a `user_id` in
    metadata is derived and threaded through where ALMA supports it.

    Note: Recording task outcome / strategy / confidence via `alma.learn()`
    is Phase 3 and intentionally NOT done here. Phase 1 stores facts only.

    Args:
        params: RPC params. Expects `content: str` and optional `metadata: dict`.
        alma: An ALMA instance (or compatible) exposing `.add_domain_knowledge`.
        agent: Default ALMA agent namespace; overridden by metadata persona.

    Returns:
        `{"ok": True}` on success, or `{"ok": False, "error": str}` on failure.
    """
    try:
        content = params.get("content")
        if not isinstance(content, str) or not content.strip():
            return {"ok": False, "error": "missing or invalid 'content'"}

        metadata = params.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            return {"ok": False, "error": "'metadata' must be an object"}

        resolved_agent = _resolve_agent(metadata, agent)

        alma.add_domain_knowledge(
            agent=resolved_agent,
            domain=_DEFAULT_DOMAIN,
            fact=content,
            source="user_stated",
        )
        return {"ok": True}
    except Exception as exc:  # never raise into the RPC layer
        return {"ok": False, "error": f"memory.store failed: {exc}"}


def _resolve_agent(metadata: Optional[Dict[str, Any]], default: str) -> str:
    """Map `metadata.persona` to an ALMA agent/scope name, falling back to default."""
    if isinstance(metadata, dict):
        persona = metadata.get("persona")
        if isinstance(persona, str) and persona.strip():
            return persona.strip()
    return default


def _resolve_user_id(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Derive a user_id from metadata when the device supplies one."""
    if isinstance(metadata, dict):
        for key in ("user_id", "userId", "user"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_alma(config_path: Optional[str] = None) -> Any:
    """Factory: build a ready-to-use ALMA instance.

    Imports `alma` lazily so this module is importable without the package.

    Args:
        config_path: Path to an ALMA config.yaml. If None, uses the zero-key
            local backend via `ALMA.quickstart()` (SQLite + sentence-transformers;
            requires `pip install 'alma-memory[local]'`).

    Returns:
        An ALMA instance.
    """
    from alma import ALMA  # lazy: keeps the adapter importable without alma

    if config_path:
        return ALMA.from_config(config_path)
    return ALMA.quickstart(project_id="maia")
