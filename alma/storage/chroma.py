"""
ALMA Chroma Storage Backend.

Vector database storage using ChromaDB for semantic search capabilities.
Supports both persistent local storage and client-server mode.

Recommended for:
- Semantic search-focused deployments
- Local development with vector search
- Small to medium scale applications
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)

# Try to import chromadb
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning(
        "chromadb not installed. Install with: pip install 'alma-memory[chroma]'"
    )


class ChromaStorage(StorageBackend):
    """
    ChromaDB storage backend with native vector search.

    Uses ChromaDB collections for each memory type with built-in
    embedding storage and similarity search.

    Collections:
        - alma_heuristics: Learned strategies
        - alma_outcomes: Task execution records
        - alma_preferences: User preferences
        - alma_domain_knowledge: Domain facts
        - alma_anti_patterns: Patterns to avoid

    Modes:
        - Persistent: Local storage with persist_directory
        - Client-Server: Remote server with host/port
        - Ephemeral: In-memory (for testing)
    """

    # Collection names
    HEURISTICS_COLLECTION = "alma_heuristics"
    OUTCOMES_COLLECTION = "alma_outcomes"
    PREFERENCES_COLLECTION = "alma_preferences"
    DOMAIN_KNOWLEDGE_COLLECTION = "alma_domain_knowledge"
    ANTI_PATTERNS_COLLECTION = "alma_anti_patterns"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embedding_dim: int = 384,
        collection_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Chroma storage.

        Args:
            persist_directory: Path for persistent local storage (mutually exclusive with host/port)
            host: Chroma server host (mutually exclusive with persist_directory)
            port: Chroma server port (required if host is specified)
            embedding_dim: Dimension of embedding vectors (for validation)
            collection_metadata: Optional metadata for collections (e.g., distance function)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Install with: pip install 'alma-memory[chroma]'"
            )

        self.embedding_dim = embedding_dim
        self._collection_metadata = collection_metadata or {
            "hnsw:space": "cosine"  # Use cosine similarity
        }

        # Initialize client based on mode
        if host and port:
            # Client-server mode
            self._client = chromadb.HttpClient(host=host, port=port)
            self._mode = "client-server"
            logger.info(f"ChromaDB client-server mode: {host}:{port}")
        elif persist_directory:
            # Persistent local mode
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._mode = "persistent"
            logger.info(f"ChromaDB persistent mode: {persist_directory}")
        else:
            # Ephemeral mode (in-memory, for testing)
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
            self._mode = "ephemeral"
            logger.info("ChromaDB ephemeral mode (in-memory)")

        # Initialize collections
        self._init_collections()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChromaStorage":
        """Create instance from configuration."""
        chroma_config = config.get("chroma", {})

        # Support environment variable expansion
        def get_value(key: str, default: Any = None) -> Any:
            value = chroma_config.get(key, default)
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                return os.environ.get(env_var, default)
            return value

        persist_directory = get_value("persist_directory")
        host = get_value("host")
        port = get_value("port")

        if port is not None:
            port = int(port)

        return cls(
            persist_directory=persist_directory,
            host=host,
            port=port,
            embedding_dim=int(config.get("embedding_dim", 384)),
            collection_metadata=chroma_config.get("collection_metadata"),
        )

    def _init_collections(self):
        """Initialize or get all collections."""
        self._heuristics = self._client.get_or_create_collection(
            name=self.HEURISTICS_COLLECTION,
            metadata=self._collection_metadata,
        )
        self._outcomes = self._client.get_or_create_collection(
            name=self.OUTCOMES_COLLECTION,
            metadata=self._collection_metadata,
        )
        self._preferences = self._client.get_or_create_collection(
            name=self.PREFERENCES_COLLECTION,
            metadata=self._collection_metadata,
        )
        self._domain_knowledge = self._client.get_or_create_collection(
            name=self.DOMAIN_KNOWLEDGE_COLLECTION,
            metadata=self._collection_metadata,
        )
        self._anti_patterns = self._client.get_or_create_collection(
            name=self.ANTI_PATTERNS_COLLECTION,
            metadata=self._collection_metadata,
        )

    def _format_get_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Reformat get() results to match query() format."""
        emb = results.get("embeddings")
        has_embeddings = emb is not None and (
            (hasattr(emb, "__len__") and len(emb) > 0)
            or (hasattr(emb, "size") and emb.size > 0)
        )
        return {
            "ids": [results.get("ids", [])],
            "metadatas": [results.get("metadatas", [])],
            "documents": [results.get("documents", [])],
            "embeddings": [emb] if has_embeddings else None,
        }

    def _datetime_to_str(self, dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO string for storage."""
        if dt is None:
            return None
        return dt.isoformat()

    def _str_to_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Convert ISO string to datetime."""
        if s is None:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        metadata = {
            "agent": heuristic.agent,
            "project_id": heuristic.project_id,
            "condition": heuristic.condition,
            "strategy": heuristic.strategy,
            "confidence": heuristic.confidence,
            "occurrence_count": heuristic.occurrence_count,
            "success_count": heuristic.success_count,
            "last_validated": self._datetime_to_str(heuristic.last_validated),
            "created_at": self._datetime_to_str(heuristic.created_at),
            "extra_metadata": json.dumps(heuristic.metadata)
            if heuristic.metadata
            else "{}",
        }

        # Chroma requires documents - use condition + strategy as document
        document = f"{heuristic.condition}\n{heuristic.strategy}"

        if heuristic.embedding:
            self._heuristics.upsert(
                ids=[heuristic.id],
                embeddings=[heuristic.embedding],
                metadatas=[metadata],
                documents=[document],
            )
        else:
            self._heuristics.upsert(
                ids=[heuristic.id],
                metadatas=[metadata],
                documents=[document],
            )

        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        metadata = {
            "agent": outcome.agent,
            "project_id": outcome.project_id,
            "task_type": outcome.task_type or "general",
            "success": outcome.success,
            "strategy_used": outcome.strategy_used or "",
            "duration_ms": outcome.duration_ms or 0,
            "error_message": outcome.error_message or "",
            "user_feedback": outcome.user_feedback or "",
            "timestamp": self._datetime_to_str(outcome.timestamp),
            "extra_metadata": json.dumps(outcome.metadata)
            if outcome.metadata
            else "{}",
        }

        document = outcome.task_description

        if outcome.embedding:
            self._outcomes.upsert(
                ids=[outcome.id],
                embeddings=[outcome.embedding],
                metadatas=[metadata],
                documents=[document],
            )
        else:
            self._outcomes.upsert(
                ids=[outcome.id],
                metadatas=[metadata],
                documents=[document],
            )

        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        metadata = {
            "user_id": preference.user_id,
            "category": preference.category or "general",
            "source": preference.source or "unknown",
            "confidence": preference.confidence,
            "timestamp": self._datetime_to_str(preference.timestamp),
            "extra_metadata": json.dumps(preference.metadata)
            if preference.metadata
            else "{}",
        }

        document = preference.preference

        self._preferences.upsert(
            ids=[preference.id],
            metadatas=[metadata],
            documents=[document],
        )

        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        metadata = {
            "agent": knowledge.agent,
            "project_id": knowledge.project_id,
            "domain": knowledge.domain or "general",
            "source": knowledge.source or "unknown",
            "confidence": knowledge.confidence,
            "last_verified": self._datetime_to_str(knowledge.last_verified),
            "extra_metadata": json.dumps(knowledge.metadata)
            if knowledge.metadata
            else "{}",
        }

        document = knowledge.fact

        if knowledge.embedding:
            self._domain_knowledge.upsert(
                ids=[knowledge.id],
                embeddings=[knowledge.embedding],
                metadatas=[metadata],
                documents=[document],
            )
        else:
            self._domain_knowledge.upsert(
                ids=[knowledge.id],
                metadatas=[metadata],
                documents=[document],
            )

        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        metadata = {
            "agent": anti_pattern.agent,
            "project_id": anti_pattern.project_id,
            "why_bad": anti_pattern.why_bad or "",
            "better_alternative": anti_pattern.better_alternative or "",
            "occurrence_count": anti_pattern.occurrence_count,
            "last_seen": self._datetime_to_str(anti_pattern.last_seen),
            "created_at": self._datetime_to_str(anti_pattern.created_at),
            "extra_metadata": json.dumps(anti_pattern.metadata)
            if anti_pattern.metadata
            else "{}",
        }

        document = anti_pattern.pattern

        if anti_pattern.embedding:
            self._anti_patterns.upsert(
                ids=[anti_pattern.id],
                embeddings=[anti_pattern.embedding],
                metadatas=[metadata],
                documents=[document],
            )
        else:
            self._anti_patterns.upsert(
                ids=[anti_pattern.id],
                metadatas=[metadata],
                documents=[document],
            )

        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch."""
        if not heuristics:
            return []

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        has_embeddings = False

        for h in heuristics:
            ids.append(h.id)
            metadatas.append(
                {
                    "agent": h.agent,
                    "project_id": h.project_id,
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                    "occurrence_count": h.occurrence_count,
                    "success_count": h.success_count,
                    "last_validated": self._datetime_to_str(h.last_validated),
                    "created_at": self._datetime_to_str(h.created_at),
                    "extra_metadata": json.dumps(h.metadata) if h.metadata else "{}",
                }
            )
            documents.append(f"{h.condition}\n{h.strategy}")
            if h.embedding:
                embeddings.append(h.embedding)
                has_embeddings = True
            else:
                embeddings.append(None)

        if has_embeddings and all(e is not None for e in embeddings):
            self._heuristics.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        else:
            self._heuristics.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )

        logger.debug(f"Batch saved {len(heuristics)} heuristics")
        return ids

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch."""
        if not outcomes:
            return []

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        has_embeddings = False

        for o in outcomes:
            ids.append(o.id)
            metadatas.append(
                {
                    "agent": o.agent,
                    "project_id": o.project_id,
                    "task_type": o.task_type or "general",
                    "success": o.success,
                    "strategy_used": o.strategy_used or "",
                    "duration_ms": o.duration_ms or 0,
                    "error_message": o.error_message or "",
                    "user_feedback": o.user_feedback or "",
                    "timestamp": self._datetime_to_str(o.timestamp),
                    "extra_metadata": json.dumps(o.metadata) if o.metadata else "{}",
                }
            )
            documents.append(o.task_description)
            if o.embedding:
                embeddings.append(o.embedding)
                has_embeddings = True
            else:
                embeddings.append(None)

        if has_embeddings and all(e is not None for e in embeddings):
            self._outcomes.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        else:
            self._outcomes.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )

        logger.debug(f"Batch saved {len(outcomes)} outcomes")
        return ids

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch."""
        if not knowledge_items:
            return []

        ids = []
        embeddings = []
        metadatas = []
        documents = []
        has_embeddings = False

        for k in knowledge_items:
            ids.append(k.id)
            metadatas.append(
                {
                    "agent": k.agent,
                    "project_id": k.project_id,
                    "domain": k.domain or "general",
                    "source": k.source or "unknown",
                    "confidence": k.confidence,
                    "last_verified": self._datetime_to_str(k.last_verified),
                    "extra_metadata": json.dumps(k.metadata) if k.metadata else "{}",
                }
            )
            documents.append(k.fact)
            if k.embedding:
                embeddings.append(k.embedding)
                has_embeddings = True
            else:
                embeddings.append(None)

        if has_embeddings and all(e is not None for e in embeddings):
            self._domain_knowledge.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        else:
            self._domain_knowledge.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )

        logger.debug(f"Batch saved {len(knowledge_items)} domain knowledge items")
        return ids

    # ==================== READ OPERATIONS ====================

    def _build_where_filter(
        self,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        success_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Build Chroma where filter from parameters."""
        conditions = []

        if project_id:
            conditions.append({"project_id": {"$eq": project_id}})
        if agent:
            conditions.append({"agent": {"$eq": agent}})
        if user_id:
            conditions.append({"user_id": {"$eq": user_id}})
        if domain:
            conditions.append({"domain": {"$eq": domain}})
        if task_type:
            conditions.append({"task_type": {"$eq": task_type}})
        if min_confidence is not None and min_confidence > 0:
            conditions.append({"confidence": {"$gte": min_confidence}})
        if success_only:
            conditions.append({"success": {"$eq": True}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _has_embedding(self, emb: Any) -> bool:
        """Safely check if embedding is not None/empty (handles numpy arrays)."""
        if emb is None:
            return False
        if hasattr(emb, "__len__"):
            try:
                return len(emb) > 0
            except (TypeError, ValueError):
                pass
        if hasattr(emb, "size"):
            return emb.size > 0
        return True

    def _get_embedding_list(self, results: Dict[str, Any], num_ids: int) -> List[Any]:
        """Safely extract embeddings list from results."""
        emb_data = results.get("embeddings")
        if emb_data is None:
            return [None] * num_ids
        # Handle both nested list format (query results) and flat format
        if isinstance(emb_data, list) and len(emb_data) > 0:
            first = emb_data[0]
            # Check if it's a nested list (query format: [[emb1, emb2, ...]])
            if isinstance(first, list) or (
                hasattr(first, "__iter__") and not isinstance(first, (str, bytes))
            ):
                # Could be list of embeddings or numpy array
                try:
                    if hasattr(first, "tolist"):
                        # numpy array
                        return list(emb_data[0])
                    return list(first) if isinstance(first, list) else [first]
                except (TypeError, IndexError):
                    return [None] * num_ids
            return list(emb_data)
        return [None] * num_ids

    def _results_to_heuristics(self, results: Dict[str, Any]) -> List[Heuristic]:
        """Convert Chroma query results to Heuristic objects."""
        heuristics = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return heuristics

        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]
        embeddings = self._get_embedding_list(results, len(ids))

        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            emb = embeddings[i] if i < len(embeddings) else None

            extra = json.loads(meta.get("extra_metadata", "{}"))

            heuristics.append(
                Heuristic(
                    id=id_,
                    agent=meta.get("agent", ""),
                    project_id=meta.get("project_id", ""),
                    condition=meta.get("condition", ""),
                    strategy=meta.get("strategy", ""),
                    confidence=meta.get("confidence", 0.0),
                    occurrence_count=meta.get("occurrence_count", 0),
                    success_count=meta.get("success_count", 0),
                    last_validated=self._str_to_datetime(meta.get("last_validated"))
                    or datetime.now(timezone.utc),
                    created_at=self._str_to_datetime(meta.get("created_at"))
                    or datetime.now(timezone.utc),
                    embedding=list(emb)
                    if emb is not None and hasattr(emb, "__iter__")
                    else emb,
                    metadata=extra,
                )
            )

        return heuristics

    def _results_to_outcomes(self, results: Dict[str, Any]) -> List[Outcome]:
        """Convert Chroma query results to Outcome objects."""
        outcomes = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return outcomes

        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        embeddings = self._get_embedding_list(results, len(ids))

        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            emb = embeddings[i] if i < len(embeddings) else None

            extra = json.loads(meta.get("extra_metadata", "{}"))

            outcomes.append(
                Outcome(
                    id=id_,
                    agent=meta.get("agent", ""),
                    project_id=meta.get("project_id", ""),
                    task_type=meta.get("task_type", "general"),
                    task_description=doc,
                    success=meta.get("success", False),
                    strategy_used=meta.get("strategy_used", ""),
                    duration_ms=meta.get("duration_ms"),
                    error_message=meta.get("error_message") or None,
                    user_feedback=meta.get("user_feedback") or None,
                    timestamp=self._str_to_datetime(meta.get("timestamp"))
                    or datetime.now(timezone.utc),
                    embedding=list(emb)
                    if emb is not None and hasattr(emb, "__iter__")
                    else emb,
                    metadata=extra,
                )
            )

        return outcomes

    def _results_to_preferences(self, results: Dict[str, Any]) -> List[UserPreference]:
        """Convert Chroma query results to UserPreference objects."""
        preferences = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return preferences

        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""

            extra = json.loads(meta.get("extra_metadata", "{}"))

            preferences.append(
                UserPreference(
                    id=id_,
                    user_id=meta.get("user_id", ""),
                    category=meta.get("category", "general"),
                    preference=doc,
                    source=meta.get("source", "unknown"),
                    confidence=meta.get("confidence", 1.0),
                    timestamp=self._str_to_datetime(meta.get("timestamp"))
                    or datetime.now(timezone.utc),
                    metadata=extra,
                )
            )

        return preferences

    def _results_to_domain_knowledge(
        self, results: Dict[str, Any]
    ) -> List[DomainKnowledge]:
        """Convert Chroma query results to DomainKnowledge objects."""
        knowledge = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return knowledge

        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        embeddings = self._get_embedding_list(results, len(ids))

        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            emb = embeddings[i] if i < len(embeddings) else None

            extra = json.loads(meta.get("extra_metadata", "{}"))

            knowledge.append(
                DomainKnowledge(
                    id=id_,
                    agent=meta.get("agent", ""),
                    project_id=meta.get("project_id", ""),
                    domain=meta.get("domain", "general"),
                    fact=doc,
                    source=meta.get("source", "unknown"),
                    confidence=meta.get("confidence", 1.0),
                    last_verified=self._str_to_datetime(meta.get("last_verified"))
                    or datetime.now(timezone.utc),
                    embedding=list(emb)
                    if emb is not None and hasattr(emb, "__iter__")
                    else emb,
                    metadata=extra,
                )
            )

        return knowledge

    def _results_to_anti_patterns(self, results: Dict[str, Any]) -> List[AntiPattern]:
        """Convert Chroma query results to AntiPattern objects."""
        patterns = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return patterns

        ids = results["ids"][0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        embeddings = self._get_embedding_list(results, len(ids))

        for i, id_ in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            emb = embeddings[i] if i < len(embeddings) else None

            extra = json.loads(meta.get("extra_metadata", "{}"))

            patterns.append(
                AntiPattern(
                    id=id_,
                    agent=meta.get("agent", ""),
                    project_id=meta.get("project_id", ""),
                    pattern=doc,
                    why_bad=meta.get("why_bad", ""),
                    better_alternative=meta.get("better_alternative", ""),
                    occurrence_count=meta.get("occurrence_count", 1),
                    last_seen=self._str_to_datetime(meta.get("last_seen"))
                    or datetime.now(timezone.utc),
                    created_at=self._str_to_datetime(meta.get("created_at"))
                    or datetime.now(timezone.utc),
                    embedding=list(emb)
                    if emb is not None and hasattr(emb, "__iter__")
                    else emb,
                    metadata=extra,
                )
            )

        return patterns

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics with optional vector search."""
        where_filter = self._build_where_filter(
            project_id=project_id,
            agent=agent,
            min_confidence=min_confidence,
        )

        if embedding:
            results = self._heuristics.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._heuristics.get(
                where=where_filter,
                limit=top_k,
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_heuristics(results)

    def get_outcomes(
        self,
        project_id: str,
        agent: Optional[str] = None,
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes with optional vector search."""
        where_filter = self._build_where_filter(
            project_id=project_id,
            agent=agent,
            task_type=task_type,
            success_only=success_only,
        )

        if embedding:
            results = self._outcomes.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._outcomes.get(
                where=where_filter,
                limit=top_k,
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_outcomes(results)

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        where_filter = self._build_where_filter(user_id=user_id)
        if category:
            if where_filter:
                where_filter = {"$and": [where_filter, {"category": {"$eq": category}}]}
            else:
                where_filter = {"category": {"$eq": category}}

        results = self._preferences.get(
            where=where_filter,
            include=["metadatas", "documents"],
        )
        results = {
            "ids": [results.get("ids", [])],
            "metadatas": [results.get("metadatas", [])],
            "documents": [results.get("documents", [])],
        }

        return self._results_to_preferences(results)

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge with optional vector search."""
        where_filter = self._build_where_filter(
            project_id=project_id,
            agent=agent,
            domain=domain,
        )

        if embedding:
            results = self._domain_knowledge.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._domain_knowledge.get(
                where=where_filter,
                limit=top_k,
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_domain_knowledge(results)

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        where_filter = self._build_where_filter(
            project_id=project_id,
            agent=agent,
        )

        if embedding:
            results = self._anti_patterns.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._anti_patterns.get(
                where=where_filter,
                limit=top_k,
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_anti_patterns(results)

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def _build_agents_filter(
        self,
        project_id: str,
        agents: List[str],
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Build filter for multiple agents."""
        if not agents:
            return None

        agent_conditions = [{"agent": {"$eq": a}} for a in agents]
        agents_filter = (
            {"$or": agent_conditions}
            if len(agent_conditions) > 1
            else agent_conditions[0]
        )

        base_filter = self._build_where_filter(project_id=project_id, **kwargs)

        if base_filter:
            return {"$and": [base_filter, agents_filter]}
        return {"$and": [{"project_id": {"$eq": project_id}}, agents_filter]}

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics from multiple agents."""
        if not agents:
            return []

        where_filter = self._build_agents_filter(
            project_id=project_id,
            agents=agents,
            min_confidence=min_confidence,
        )

        if embedding:
            results = self._heuristics.query(
                query_embeddings=[embedding],
                n_results=top_k * len(agents),
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._heuristics.get(
                where=where_filter,
                limit=top_k * len(agents),
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_heuristics(results)

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes from multiple agents."""
        if not agents:
            return []

        where_filter = self._build_agents_filter(
            project_id=project_id,
            agents=agents,
            task_type=task_type,
            success_only=success_only,
        )

        if embedding:
            results = self._outcomes.query(
                query_embeddings=[embedding],
                n_results=top_k * len(agents),
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._outcomes.get(
                where=where_filter,
                limit=top_k * len(agents),
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_outcomes(results)

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge from multiple agents."""
        if not agents:
            return []

        where_filter = self._build_agents_filter(
            project_id=project_id,
            agents=agents,
            domain=domain,
        )

        if embedding:
            results = self._domain_knowledge.query(
                query_embeddings=[embedding],
                n_results=top_k * len(agents),
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._domain_knowledge.get(
                where=where_filter,
                limit=top_k * len(agents),
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_domain_knowledge(results)

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns from multiple agents."""
        if not agents:
            return []

        where_filter = self._build_agents_filter(
            project_id=project_id,
            agents=agents,
        )

        if embedding:
            results = self._anti_patterns.query(
                query_embeddings=[embedding],
                n_results=top_k * len(agents),
                where=where_filter,
                include=["metadatas", "documents", "embeddings"],
            )
        else:
            results = self._anti_patterns.get(
                where=where_filter,
                limit=top_k * len(agents),
                include=["metadatas", "documents", "embeddings"],
            )
            results = self._format_get_results(results)

        return self._results_to_anti_patterns(results)

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        if not updates:
            return False

        try:
            # Get existing heuristic
            existing = self._heuristics.get(
                ids=[heuristic_id], include=["metadatas", "documents", "embeddings"]
            )
            if not existing or not existing.get("ids"):
                return False

            metadata = existing["metadatas"][0] if existing.get("metadatas") else {}
            document = existing["documents"][0] if existing.get("documents") else ""
            emb_list = existing.get("embeddings")
            embedding = (
                emb_list[0] if emb_list is not None and len(emb_list) > 0 else None
            )

            # Apply updates
            for key, value in updates.items():
                if key == "condition":
                    metadata["condition"] = value
                    # Update document as well
                    parts = document.split("\n", 1)
                    document = f"{value}\n{parts[1] if len(parts) > 1 else ''}"
                elif key == "strategy":
                    metadata["strategy"] = value
                    parts = document.split("\n", 1)
                    document = f"{parts[0] if parts else ''}\n{value}"
                elif key == "metadata":
                    metadata["extra_metadata"] = json.dumps(value)
                elif key in ("last_validated", "created_at") and isinstance(
                    value, datetime
                ):
                    metadata[key] = value.isoformat()
                elif key in metadata:
                    metadata[key] = value

            # Upsert with updated values
            if self._has_embedding(embedding):
                self._heuristics.upsert(
                    ids=[heuristic_id],
                    embeddings=[
                        list(embedding) if hasattr(embedding, "__iter__") else embedding
                    ],
                    metadatas=[metadata],
                    documents=[document],
                )
            else:
                self._heuristics.upsert(
                    ids=[heuristic_id],
                    metadatas=[metadata],
                    documents=[document],
                )

            return True
        except Exception as e:
            logger.warning(f"Failed to update heuristic {heuristic_id}: {e}")
            return False

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        try:
            existing = self._heuristics.get(
                ids=[heuristic_id], include=["metadatas", "documents", "embeddings"]
            )
            if not existing or not existing.get("ids"):
                return False

            metadata = existing["metadatas"][0] if existing.get("metadatas") else {}
            document = existing["documents"][0] if existing.get("documents") else ""
            emb_list = existing.get("embeddings")
            embedding = (
                emb_list[0] if emb_list is not None and len(emb_list) > 0 else None
            )

            metadata["occurrence_count"] = metadata.get("occurrence_count", 0) + 1
            if success:
                metadata["success_count"] = metadata.get("success_count", 0) + 1
            metadata["last_validated"] = datetime.now(timezone.utc).isoformat()

            if self._has_embedding(embedding):
                self._heuristics.upsert(
                    ids=[heuristic_id],
                    embeddings=[
                        list(embedding) if hasattr(embedding, "__iter__") else embedding
                    ],
                    metadatas=[metadata],
                    documents=[document],
                )
            else:
                self._heuristics.upsert(
                    ids=[heuristic_id],
                    metadatas=[metadata],
                    documents=[document],
                )

            return True
        except Exception as e:
            logger.warning(f"Failed to increment occurrence for {heuristic_id}: {e}")
            return False

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update a heuristic's confidence value."""
        return self.update_heuristic(heuristic_id, {"confidence": new_confidence})

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update domain knowledge confidence value."""
        try:
            existing = self._domain_knowledge.get(
                ids=[knowledge_id], include=["metadatas", "documents", "embeddings"]
            )
            if not existing or not existing.get("ids"):
                return False

            metadata = existing["metadatas"][0] if existing.get("metadatas") else {}
            document = existing["documents"][0] if existing.get("documents") else ""
            emb_list = existing.get("embeddings")
            embedding = (
                emb_list[0] if emb_list is not None and len(emb_list) > 0 else None
            )

            metadata["confidence"] = new_confidence

            if self._has_embedding(embedding):
                self._domain_knowledge.upsert(
                    ids=[knowledge_id],
                    embeddings=[
                        list(embedding) if hasattr(embedding, "__iter__") else embedding
                    ],
                    metadatas=[metadata],
                    documents=[document],
                )
            else:
                self._domain_knowledge.upsert(
                    ids=[knowledge_id],
                    metadatas=[metadata],
                    documents=[document],
                )

            return True
        except Exception as e:
            logger.warning(f"Failed to update knowledge confidence {knowledge_id}: {e}")
            return False

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        try:
            existing = self._heuristics.get(ids=[heuristic_id])
            if not existing or not existing.get("ids"):
                return False
            self._heuristics.delete(ids=[heuristic_id])
            logger.debug(f"Deleted heuristic: {heuristic_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete heuristic {heuristic_id}: {e}")
            return False

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        try:
            existing = self._outcomes.get(ids=[outcome_id])
            if not existing or not existing.get("ids"):
                return False
            self._outcomes.delete(ids=[outcome_id])
            logger.debug(f"Deleted outcome: {outcome_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete outcome {outcome_id}: {e}")
            return False

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        try:
            existing = self._domain_knowledge.get(ids=[knowledge_id])
            if not existing or not existing.get("ids"):
                return False
            self._domain_knowledge.delete(ids=[knowledge_id])
            logger.debug(f"Deleted domain knowledge: {knowledge_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete domain knowledge {knowledge_id}: {e}")
            return False

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        try:
            existing = self._anti_patterns.get(ids=[anti_pattern_id])
            if not existing or not existing.get("ids"):
                return False
            self._anti_patterns.delete(ids=[anti_pattern_id])
            logger.debug(f"Deleted anti-pattern: {anti_pattern_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete anti-pattern {anti_pattern_id}: {e}")
            return False

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        where_filter = self._build_where_filter(project_id=project_id, agent=agent)

        # Get all matching outcomes
        results = self._outcomes.get(
            where=where_filter,
            include=["metadatas"],
        )

        if not results or not results.get("ids"):
            return 0

        older_than_str = older_than.isoformat()
        ids_to_delete = []

        for i, id_ in enumerate(results["ids"]):
            meta = (
                results["metadatas"][i] if i < len(results.get("metadatas", [])) else {}
            )
            timestamp_str = meta.get("timestamp", "")
            if timestamp_str and timestamp_str < older_than_str:
                ids_to_delete.append(id_)

        if ids_to_delete:
            self._outcomes.delete(ids=ids_to_delete)

        logger.info(f"Deleted {len(ids_to_delete)} old outcomes")
        return len(ids_to_delete)

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        where_filter = self._build_where_filter(project_id=project_id, agent=agent)

        # Get all matching heuristics
        results = self._heuristics.get(
            where=where_filter,
            include=["metadatas"],
        )

        if not results or not results.get("ids"):
            return 0

        ids_to_delete = []

        for i, id_ in enumerate(results["ids"]):
            meta = (
                results["metadatas"][i] if i < len(results.get("metadatas", [])) else {}
            )
            confidence = meta.get("confidence", 0.0)
            if confidence < below_confidence:
                ids_to_delete.append(id_)

        if ids_to_delete:
            self._heuristics.delete(ids=ids_to_delete)

        logger.info(f"Deleted {len(ids_to_delete)} low-confidence heuristics")
        return len(ids_to_delete)

    # ==================== STATS ====================

    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "project_id": project_id,
            "agent": agent,
            "storage_type": "chroma",
            "mode": self._mode,
        }

        where_filter = self._build_where_filter(project_id=project_id, agent=agent)

        # Count items in each collection
        for name, collection in [
            ("heuristics", self._heuristics),
            ("outcomes", self._outcomes),
            ("domain_knowledge", self._domain_knowledge),
            ("anti_patterns", self._anti_patterns),
        ]:
            try:
                results = collection.get(where=where_filter)
                stats[f"{name}_count"] = len(results.get("ids", []))
            except Exception:
                stats[f"{name}_count"] = 0

        # Preferences don't have project_id filter
        try:
            results = self._preferences.get()
            stats["preferences_count"] = len(results.get("ids", []))
        except Exception:
            stats["preferences_count"] = 0

        stats["total_count"] = sum(
            stats.get(k, 0) for k in stats if k.endswith("_count")
        )

        return stats

    def close(self):
        """Close the Chroma client (if applicable)."""
        # ChromaDB handles cleanup automatically
        logger.info("ChromaDB storage closed")
