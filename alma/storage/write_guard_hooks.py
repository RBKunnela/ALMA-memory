"""
Storage-layer anti-pattern write guard (Agent Memory Atlas + Chefe 1756).

Atlas re-read (e2178ad): the guard existed only on ``learn()`` — one door of six.
This module wraps StorageBackend ``save_*`` methods so every write path shares
one door: extractor, consolidation, MCP, ``add_domain_knowledge``, etc.

``save_anti_pattern`` is intentionally NOT wrapped — that table *is* the
rejection record (tombstone source); guarding it would block learning what not to do.

Idempotent: ``install_storage_write_guards(cls)`` is safe to call multiple times.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence, Type

logger = logging.getLogger(__name__)

_INSTALLED: set[int] = set()


def enforce_storage_write_guard(
    storage: Any,
    project_id: str,
    agent: str,
    texts: Sequence[str],
) -> None:
    """
    Raise ScopeViolationError if texts match a stored anti-pattern.

    Fail-open behaviour is owned by ``check_write_guard`` (missing getter,
    lookup errors, env off).
    """
    from alma.exceptions import ScopeViolationError
    from alma.learning.write_guard import check_write_guard

    result = check_write_guard(
        storage,
        project_id=project_id or "",
        agent=agent or "",
        texts=list(texts),
    )
    if result.blocked:
        raise ScopeViolationError(
            f"Write blocked by anti-pattern write guard "
            f"(id={result.anti_pattern_id}): {result.matched_pattern}"
        )


def _pref_project_id(preference: Any) -> str:
    meta = getattr(preference, "metadata", None) or {}
    if isinstance(meta, dict):
        return str(meta.get("project_id") or "")
    return ""


def _wrap_save(
    orig: Callable[..., str],
    texts_fn: Callable[[Any], Sequence[str]],
    project_fn: Callable[[Any], str],
    agent_fn: Callable[[Any], str],
) -> Callable[..., str]:
    def wrapped(self: Any, obj: Any) -> str:
        enforce_storage_write_guard(
            self,
            project_id=project_fn(obj),
            agent=agent_fn(obj),
            texts=texts_fn(obj),
        )
        return orig(self, obj)

    wrapped.__name__ = getattr(orig, "__name__", "save_wrapped")
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    return wrapped


def _wrap_batch(
    orig: Callable[..., List[str]],
    texts_fn: Callable[[Any], Sequence[str]],
    project_fn: Callable[[Any], str],
    agent_fn: Callable[[Any], str],
) -> Callable[..., List[str]]:
    def wrapped(self: Any, items: Optional[List[Any]]) -> List[str]:
        for obj in items or []:
            enforce_storage_write_guard(
                self,
                project_id=project_fn(obj),
                agent=agent_fn(obj),
                texts=texts_fn(obj),
            )
        return orig(self, items)

    wrapped.__name__ = getattr(orig, "__name__", "save_batch_wrapped")
    wrapped.__doc__ = getattr(orig, "__doc__", None)
    return wrapped


def install_storage_write_guards(cls: Type[Any]) -> Type[Any]:
    """
    Wrap write methods on a StorageBackend subclass (in place).

    Guarded:
      - save_heuristic / save_heuristics
      - save_outcome / save_outcomes
      - save_domain_knowledge / save_domain_knowledge_batch
      - save_user_preference

    Not guarded:
      - save_anti_pattern (source of tombstones)
    """
    marker = id(cls)
    if marker in _INSTALLED or getattr(cls, "_alma_write_guards_installed", False):
        return cls

    # --- single-row ---
    if hasattr(cls, "save_heuristic"):
        cls.save_heuristic = _wrap_save(  # type: ignore[method-assign]
            cls.save_heuristic,
            texts_fn=lambda h: [getattr(h, "condition", ""), getattr(h, "strategy", "")],
            project_fn=lambda h: getattr(h, "project_id", "") or "",
            agent_fn=lambda h: getattr(h, "agent", "") or "",
        )

    if hasattr(cls, "save_outcome"):
        cls.save_outcome = _wrap_save(  # type: ignore[method-assign]
            cls.save_outcome,
            texts_fn=lambda o: [
                getattr(o, "task_description", "") or "",
                getattr(o, "strategy_used", "") or "",
                getattr(o, "error_message", "") or "",
            ],
            project_fn=lambda o: getattr(o, "project_id", "") or "",
            agent_fn=lambda o: getattr(o, "agent", "") or "",
        )

    if hasattr(cls, "save_domain_knowledge"):
        cls.save_domain_knowledge = _wrap_save(  # type: ignore[method-assign]
            cls.save_domain_knowledge,
            texts_fn=lambda k: [
                getattr(k, "domain", "") or "",
                getattr(k, "fact", "") or "",
            ],
            project_fn=lambda k: getattr(k, "project_id", "") or "",
            agent_fn=lambda k: getattr(k, "agent", "") or "",
        )

    if hasattr(cls, "save_user_preference"):
        cls.save_user_preference = _wrap_save(  # type: ignore[method-assign]
            cls.save_user_preference,
            texts_fn=lambda p: [
                getattr(p, "category", "") or "",
                getattr(p, "preference", "") or "",
            ],
            project_fn=_pref_project_id,
            agent_fn=lambda p: getattr(p, "user_id", "") or "",
        )

    # --- batch (must not bypass single-row guard) ---
    if hasattr(cls, "save_heuristics"):
        cls.save_heuristics = _wrap_batch(  # type: ignore[method-assign]
            cls.save_heuristics,
            texts_fn=lambda h: [getattr(h, "condition", ""), getattr(h, "strategy", "")],
            project_fn=lambda h: getattr(h, "project_id", "") or "",
            agent_fn=lambda h: getattr(h, "agent", "") or "",
        )

    if hasattr(cls, "save_outcomes"):
        cls.save_outcomes = _wrap_batch(  # type: ignore[method-assign]
            cls.save_outcomes,
            texts_fn=lambda o: [
                getattr(o, "task_description", "") or "",
                getattr(o, "strategy_used", "") or "",
                getattr(o, "error_message", "") or "",
            ],
            project_fn=lambda o: getattr(o, "project_id", "") or "",
            agent_fn=lambda o: getattr(o, "agent", "") or "",
        )

    if hasattr(cls, "save_domain_knowledge_batch"):
        cls.save_domain_knowledge_batch = _wrap_batch(  # type: ignore[method-assign]
            cls.save_domain_knowledge_batch,
            texts_fn=lambda k: [
                getattr(k, "domain", "") or "",
                getattr(k, "fact", "") or "",
            ],
            project_fn=lambda k: getattr(k, "project_id", "") or "",
            agent_fn=lambda k: getattr(k, "agent", "") or "",
        )

    cls._alma_write_guards_installed = True  # type: ignore[attr-defined]
    _INSTALLED.add(marker)
    logger.debug("Installed storage write guards on %s", cls.__name__)
    return cls


def install_all_known_storage_guards() -> None:
    """Install guards on every known StorageBackend implementation."""
    from alma.storage.file_based import FileBasedStorage
    from alma.storage.sqlite_local import SQLiteStorage

    install_storage_write_guards(FileBasedStorage)
    install_storage_write_guards(SQLiteStorage)

    try:
        from alma.storage.postgresql import PostgreSQLStorage

        install_storage_write_guards(PostgreSQLStorage)
    except ImportError:
        pass

    try:
        from alma.storage.azure_cosmos import AzureCosmosStorage

        if AzureCosmosStorage is not None:
            install_storage_write_guards(AzureCosmosStorage)
    except ImportError:
        pass

    for mod_path, cls_name in (
        ("alma.storage.chroma", "ChromaStorage"),
        ("alma.storage.qdrant", "QdrantStorage"),
        ("alma.storage.pinecone", "PineconeStorage"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                install_storage_write_guards(cls)
        except ImportError:
            pass

    try:
        from alma.testing.mocks import MockStorage

        install_storage_write_guards(MockStorage)
    except ImportError:
        pass
