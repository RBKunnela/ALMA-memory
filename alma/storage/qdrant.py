"""
ALMA Qdrant Storage Backend.

Production-ready storage using Qdrant vector database for efficient
semantic search with native vector operations.

Recommended for:
- High-performance vector similarity search
- Self-hosted or cloud Qdrant deployments
- Scalable production environments
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from alma.storage.base import StorageBackend
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)

logger = logging.getLogger(__name__)

# Try to import qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(
        "qdrant-client not installed. Install with: pip install 'alma-memory[qdrant]'"
    )


class QdrantStorage(StorageBackend):
    """
    Qdrant vector database storage backend.

    Uses Qdrant for native vector similarity search with efficient indexing.

    Collections:
        - {prefix}heuristics: Learned strategies and rules
        - {prefix}outcomes: Task execution records
        - {prefix}preferences: User preferences
        - {prefix}domain_knowledge: Domain-specific facts
        - {prefix}anti_patterns: Patterns to avoid

    Vector search:
        - Uses cosine similarity for semantic search
        - Supports filtering by metadata fields
    """

    # Collection names
    HEURISTICS = "heuristics"
    OUTCOMES = "outcomes"
    PREFERENCES = "preferences"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    ANTI_PATTERNS = "anti_patterns"

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_prefix: str = "alma_",
        embedding_dim: int = 384,
        timeout: int = 30,
        prefer_grpc: bool = False,
    ):
        """
        Initialize Qdrant storage.

        Args:
            url: Qdrant server URL (default: http://localhost:6333)
            api_key: Optional API key for authentication
            collection_prefix: Prefix for collection names (default: alma_)
            embedding_dim: Dimension of embedding vectors
            timeout: Request timeout in seconds
            prefer_grpc: Use gRPC instead of HTTP
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install 'alma-memory[qdrant]'"
            )

        self.url = url
        self.api_key = api_key
        self.collection_prefix = collection_prefix
        self.embedding_dim = embedding_dim
        self.timeout = timeout

        # Initialize client
        self._client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )

        # Initialize collections
        self._init_collections()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QdrantStorage":
        """Create instance from configuration."""
        qdrant_config = config.get("qdrant", {})

        # Support environment variable expansion
        def get_value(key: str, default: Any = None) -> Any:
            value = qdrant_config.get(key, default)
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                return os.environ.get(env_var, default)
            return value

        return cls(
            url=get_value("url", "http://localhost:6333"),
            api_key=get_value("api_key"),
            collection_prefix=get_value("collection_prefix", "alma_"),
            embedding_dim=int(config.get("embedding_dim", 384)),
            timeout=int(get_value("timeout", 30)),
            prefer_grpc=bool(get_value("prefer_grpc", False)),
        )

    def _collection_name(self, base_name: str) -> str:
        """Get full collection name with prefix."""
        return f"{self.collection_prefix}{base_name}"

    def _init_collections(self) -> None:
        """Initialize all required collections."""
        collections = [
            self.HEURISTICS,
            self.OUTCOMES,
            self.PREFERENCES,
            self.DOMAIN_KNOWLEDGE,
            self.ANTI_PATTERNS,
        ]

        for collection in collections:
            full_name = self._collection_name(collection)
            try:
                # Check if collection exists
                self._client.get_collection(full_name)
                logger.debug(f"Collection {full_name} already exists")
            except (UnexpectedResponse, Exception):
                # Create collection
                self._client.create_collection(
                    collection_name=full_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {full_name}")

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid4())

    def _datetime_to_str(self, dt: Optional[datetime]) -> Optional[str]:
        """Convert datetime to ISO string."""
        if dt is None:
            return None
        return dt.isoformat()

    def _datetime_to_timestamp(self, dt: Optional[datetime]) -> Optional[float]:
        """Convert datetime to Unix timestamp for range filtering."""
        if dt is None:
            return None
        return dt.timestamp()

    def _str_to_datetime(self, s: Optional[str]) -> Optional[datetime]:
        """Convert ISO string to datetime."""
        if s is None:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _get_dummy_vector(self) -> List[float]:
        """Get a dummy zero vector for items without embeddings."""
        return [0.0] * self.embedding_dim

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        collection = self._collection_name(self.HEURISTICS)

        payload = {
            "id": heuristic.id,
            "agent": heuristic.agent,
            "project_id": heuristic.project_id,
            "condition": heuristic.condition,
            "strategy": heuristic.strategy,
            "confidence": heuristic.confidence,
            "occurrence_count": heuristic.occurrence_count,
            "success_count": heuristic.success_count,
            "last_validated": self._datetime_to_str(heuristic.last_validated),
            "created_at": self._datetime_to_str(heuristic.created_at),
            "metadata": heuristic.metadata or {},
        }

        vector = (
            heuristic.embedding if heuristic.embedding else self._get_dummy_vector()
        )

        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=heuristic.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        collection = self._collection_name(self.OUTCOMES)

        payload = {
            "id": outcome.id,
            "agent": outcome.agent,
            "project_id": outcome.project_id,
            "task_type": outcome.task_type,
            "task_description": outcome.task_description,
            "success": outcome.success,
            "strategy_used": outcome.strategy_used,
            "duration_ms": outcome.duration_ms,
            "error_message": outcome.error_message,
            "user_feedback": outcome.user_feedback,
            "timestamp": self._datetime_to_str(outcome.timestamp),
            "timestamp_unix": self._datetime_to_timestamp(outcome.timestamp),
            "metadata": outcome.metadata or {},
        }

        vector = outcome.embedding if outcome.embedding else self._get_dummy_vector()

        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=outcome.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        collection = self._collection_name(self.PREFERENCES)

        payload = {
            "id": preference.id,
            "user_id": preference.user_id,
            "category": preference.category,
            "preference": preference.preference,
            "source": preference.source,
            "confidence": preference.confidence,
            "timestamp": self._datetime_to_str(preference.timestamp),
            "metadata": preference.metadata or {},
        }

        # Preferences don't have embeddings, use dummy vector
        vector = self._get_dummy_vector()

        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=preference.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        payload = {
            "id": knowledge.id,
            "agent": knowledge.agent,
            "project_id": knowledge.project_id,
            "domain": knowledge.domain,
            "fact": knowledge.fact,
            "source": knowledge.source,
            "confidence": knowledge.confidence,
            "last_verified": self._datetime_to_str(knowledge.last_verified),
            "metadata": knowledge.metadata or {},
        }

        vector = (
            knowledge.embedding if knowledge.embedding else self._get_dummy_vector()
        )

        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=knowledge.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        collection = self._collection_name(self.ANTI_PATTERNS)

        payload = {
            "id": anti_pattern.id,
            "agent": anti_pattern.agent,
            "project_id": anti_pattern.project_id,
            "pattern": anti_pattern.pattern,
            "why_bad": anti_pattern.why_bad,
            "better_alternative": anti_pattern.better_alternative,
            "occurrence_count": anti_pattern.occurrence_count,
            "last_seen": self._datetime_to_str(anti_pattern.last_seen),
            "created_at": self._datetime_to_str(anti_pattern.created_at),
            "metadata": anti_pattern.metadata or {},
        }

        vector = (
            anti_pattern.embedding
            if anti_pattern.embedding
            else self._get_dummy_vector()
        )

        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=anti_pattern.id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch."""
        if not heuristics:
            return []

        collection = self._collection_name(self.HEURISTICS)

        points = []
        for h in heuristics:
            payload = {
                "id": h.id,
                "agent": h.agent,
                "project_id": h.project_id,
                "condition": h.condition,
                "strategy": h.strategy,
                "confidence": h.confidence,
                "occurrence_count": h.occurrence_count,
                "success_count": h.success_count,
                "last_validated": self._datetime_to_str(h.last_validated),
                "created_at": self._datetime_to_str(h.created_at),
                "metadata": h.metadata or {},
            }
            vector = h.embedding if h.embedding else self._get_dummy_vector()
            points.append(models.PointStruct(id=h.id, vector=vector, payload=payload))

        self._client.upsert(collection_name=collection, points=points)
        logger.debug(f"Batch saved {len(heuristics)} heuristics")
        return [h.id for h in heuristics]

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch."""
        if not outcomes:
            return []

        collection = self._collection_name(self.OUTCOMES)

        points = []
        for o in outcomes:
            payload = {
                "id": o.id,
                "agent": o.agent,
                "project_id": o.project_id,
                "task_type": o.task_type,
                "task_description": o.task_description,
                "success": o.success,
                "strategy_used": o.strategy_used,
                "duration_ms": o.duration_ms,
                "error_message": o.error_message,
                "user_feedback": o.user_feedback,
                "timestamp": self._datetime_to_str(o.timestamp),
                "timestamp_unix": self._datetime_to_timestamp(o.timestamp),
                "metadata": o.metadata or {},
            }
            vector = o.embedding if o.embedding else self._get_dummy_vector()
            points.append(models.PointStruct(id=o.id, vector=vector, payload=payload))

        self._client.upsert(collection_name=collection, points=points)
        logger.debug(f"Batch saved {len(outcomes)} outcomes")
        return [o.id for o in outcomes]

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch."""
        if not knowledge_items:
            return []

        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        points = []
        for k in knowledge_items:
            payload = {
                "id": k.id,
                "agent": k.agent,
                "project_id": k.project_id,
                "domain": k.domain,
                "fact": k.fact,
                "source": k.source,
                "confidence": k.confidence,
                "last_verified": self._datetime_to_str(k.last_verified),
                "metadata": k.metadata or {},
            }
            vector = k.embedding if k.embedding else self._get_dummy_vector()
            points.append(models.PointStruct(id=k.id, vector=vector, payload=payload))

        self._client.upsert(collection_name=collection, points=points)
        logger.debug(f"Batch saved {len(knowledge_items)} domain knowledge items")
        return [k.id for k in knowledge_items]

    # ==================== READ OPERATIONS ====================

    def _build_filter(
        self,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        agents: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        success_only: bool = False,
        min_confidence: float = 0.0,
    ) -> Optional[Any]:  # Returns models.Filter when qdrant-client is installed
        """Build a Qdrant filter from parameters."""
        conditions = []

        if project_id:
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id),
                )
            )

        if agent:
            conditions.append(
                models.FieldCondition(
                    key="agent",
                    match=models.MatchValue(value=agent),
                )
            )

        if agents:
            conditions.append(
                models.FieldCondition(
                    key="agent",
                    match=models.MatchAny(any=agents),
                )
            )

        if user_id:
            conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            )

        if task_type:
            conditions.append(
                models.FieldCondition(
                    key="task_type",
                    match=models.MatchValue(value=task_type),
                )
            )

        if domain:
            conditions.append(
                models.FieldCondition(
                    key="domain",
                    match=models.MatchValue(value=domain),
                )
            )

        if category:
            conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category),
                )
            )

        if success_only:
            conditions.append(
                models.FieldCondition(
                    key="success",
                    match=models.MatchValue(value=True),
                )
            )

        if min_confidence > 0.0:
            conditions.append(
                models.FieldCondition(
                    key="confidence",
                    range=models.Range(gte=min_confidence),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics with optional vector search."""
        collection = self._collection_name(self.HEURISTICS)

        query_filter = self._build_filter(
            project_id=project_id,
            agent=agent,
            min_confidence=min_confidence,
        )

        if embedding:
            # Vector search
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            # Scroll without vector search
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_heuristic(r) for r in results]

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
        collection = self._collection_name(self.OUTCOMES)

        query_filter = self._build_filter(
            project_id=project_id,
            agent=agent,
            task_type=task_type,
            success_only=success_only,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_outcome(r) for r in results]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        collection = self._collection_name(self.PREFERENCES)

        query_filter = self._build_filter(
            user_id=user_id,
            category=category,
        )

        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=100,  # Get all preferences for user
            with_payload=True,
            with_vectors=False,
        )

        return [self._point_to_preference(r) for r in results]

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge with optional vector search."""
        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        query_filter = self._build_filter(
            project_id=project_id,
            agent=agent,
            domain=domain,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_domain_knowledge(r) for r in results]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        collection = self._collection_name(self.ANTI_PATTERNS)

        query_filter = self._build_filter(
            project_id=project_id,
            agent=agent,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_anti_pattern(r) for r in results]

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics from multiple agents using optimized query."""
        if not agents:
            return []

        collection = self._collection_name(self.HEURISTICS)

        query_filter = self._build_filter(
            project_id=project_id,
            agents=agents,
            min_confidence=min_confidence,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_heuristic(r) for r in results]

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes from multiple agents using optimized query."""
        if not agents:
            return []

        collection = self._collection_name(self.OUTCOMES)

        query_filter = self._build_filter(
            project_id=project_id,
            agents=agents,
            task_type=task_type,
            success_only=success_only,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_outcome(r) for r in results]

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge from multiple agents using optimized query."""
        if not agents:
            return []

        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        query_filter = self._build_filter(
            project_id=project_id,
            agents=agents,
            domain=domain,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_domain_knowledge(r) for r in results]

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns from multiple agents using optimized query."""
        if not agents:
            return []

        collection = self._collection_name(self.ANTI_PATTERNS)

        query_filter = self._build_filter(
            project_id=project_id,
            agents=agents,
        )

        if embedding:
            results = self._client.search(
                collection_name=collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=top_k * len(agents),
                with_payload=True,
                with_vectors=False,
            )

        return [self._point_to_anti_pattern(r) for r in results]

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        if not updates:
            return False

        collection = self._collection_name(self.HEURISTICS)

        # Convert datetime fields to strings
        payload_updates: Dict[str, Any] = {}
        for key, value in updates.items():
            if isinstance(value, datetime):
                payload_updates[key] = self._datetime_to_str(value)
            elif key == "metadata" and isinstance(value, dict):
                payload_updates[key] = value
            else:
                payload_updates[key] = value

        try:
            self._client.set_payload(
                collection_name=collection,
                payload=payload_updates,
                points=[heuristic_id],
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
        collection = self._collection_name(self.HEURISTICS)

        try:
            # Get current values
            results = self._client.retrieve(
                collection_name=collection,
                ids=[heuristic_id],
                with_payload=True,
            )

            if not results:
                return False

            payload = results[0].payload or {}
            new_occurrence = int(payload.get("occurrence_count") or 0) + 1
            new_success = int(payload.get("success_count") or 0)
            if success:
                new_success += 1

            self._client.set_payload(
                collection_name=collection,
                payload={
                    "occurrence_count": new_occurrence,
                    "success_count": new_success,
                    "last_validated": self._datetime_to_str(datetime.now(timezone.utc)),
                },
                points=[heuristic_id],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to increment heuristic {heuristic_id}: {e}")
            return False

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """Update a heuristic's confidence value."""
        collection = self._collection_name(self.HEURISTICS)

        try:
            self._client.set_payload(
                collection_name=collection,
                payload={"confidence": new_confidence},
                points=[heuristic_id],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update confidence for {heuristic_id}: {e}")
            return False

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """Update domain knowledge confidence value."""
        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        try:
            self._client.set_payload(
                collection_name=collection,
                payload={"confidence": new_confidence},
                points=[knowledge_id],
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update confidence for {knowledge_id}: {e}")
            return False

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        collection = self._collection_name(self.HEURISTICS)

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[heuristic_id]),
            )
            logger.debug(f"Deleted heuristic: {heuristic_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete heuristic {heuristic_id}: {e}")
            return False

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        collection = self._collection_name(self.OUTCOMES)

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[outcome_id]),
            )
            logger.debug(f"Deleted outcome: {outcome_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete outcome {outcome_id}: {e}")
            return False

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        collection = self._collection_name(self.DOMAIN_KNOWLEDGE)

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[knowledge_id]),
            )
            logger.debug(f"Deleted domain knowledge: {knowledge_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete domain knowledge {knowledge_id}: {e}")
            return False

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        collection = self._collection_name(self.ANTI_PATTERNS)

        try:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[anti_pattern_id]),
            )
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
        collection = self._collection_name(self.OUTCOMES)

        # Build filter for deletion using Unix timestamp for range comparison
        conditions = [
            models.FieldCondition(
                key="project_id",
                match=models.MatchValue(value=project_id),
            ),
            models.FieldCondition(
                key="timestamp_unix",
                range=models.Range(lt=self._datetime_to_timestamp(older_than)),
            ),
        ]

        if agent:
            conditions.append(
                models.FieldCondition(
                    key="agent",
                    match=models.MatchValue(value=agent),
                )
            )

        delete_filter = models.Filter(must=conditions)

        # Get count before deletion
        count_before = self._client.count(
            collection_name=collection,
            count_filter=delete_filter,
            exact=True,
        ).count

        # Delete matching points
        self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=delete_filter),
        )

        logger.info(f"Deleted {count_before} old outcomes")
        return count_before

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        collection = self._collection_name(self.HEURISTICS)

        # Build filter for deletion
        conditions = [
            models.FieldCondition(
                key="project_id",
                match=models.MatchValue(value=project_id),
            ),
            models.FieldCondition(
                key="confidence",
                range=models.Range(lt=below_confidence),
            ),
        ]

        if agent:
            conditions.append(
                models.FieldCondition(
                    key="agent",
                    match=models.MatchValue(value=agent),
                )
            )

        delete_filter = models.Filter(must=conditions)

        # Get count before deletion
        count_before = self._client.count(
            collection_name=collection,
            count_filter=delete_filter,
            exact=True,
        ).count

        # Delete matching points
        self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=delete_filter),
        )

        logger.info(f"Deleted {count_before} low-confidence heuristics")
        return count_before

    # ==================== STATS ====================

    def get_stats(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        stats: Dict[str, Any] = {
            "project_id": project_id,
            "agent": agent,
            "storage_type": "qdrant",
            "url": self.url,
        }

        collections_map = {
            "heuristics": self.HEURISTICS,
            "outcomes": self.OUTCOMES,
            "domain_knowledge": self.DOMAIN_KNOWLEDGE,
            "anti_patterns": self.ANTI_PATTERNS,
        }

        for stat_name, collection_base in collections_map.items():
            collection = self._collection_name(collection_base)

            query_filter = self._build_filter(
                project_id=project_id,
                agent=agent,
            )

            try:
                count = self._client.count(
                    collection_name=collection,
                    count_filter=query_filter,
                    exact=True,
                ).count
                stats[f"{stat_name}_count"] = count
            except Exception:
                stats[f"{stat_name}_count"] = 0

        # Preferences don't have project_id filter
        try:
            prefs_collection = self._collection_name(self.PREFERENCES)
            prefs_count = self._client.count(
                collection_name=prefs_collection,
                exact=True,
            ).count
            stats["preferences_count"] = prefs_count
        except Exception:
            stats["preferences_count"] = 0

        stats["total_count"] = sum(
            int(stats.get(k, 0)) for k in stats if k.endswith("_count")
        )

        return stats

    # ==================== HELPERS ====================

    def _point_to_heuristic(self, point: Any) -> Heuristic:
        """Convert Qdrant point to Heuristic."""
        payload = point.payload
        return Heuristic(
            id=payload["id"],
            agent=payload["agent"],
            project_id=payload["project_id"],
            condition=payload["condition"],
            strategy=payload["strategy"],
            confidence=payload.get("confidence") or 0.0,
            occurrence_count=payload.get("occurrence_count") or 0,
            success_count=payload.get("success_count") or 0,
            last_validated=self._str_to_datetime(payload.get("last_validated"))
            or datetime.now(timezone.utc),
            created_at=self._str_to_datetime(payload.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=getattr(point, "vector", None),
            metadata=payload.get("metadata") or {},
        )

    def _point_to_outcome(self, point: Any) -> Outcome:
        """Convert Qdrant point to Outcome."""
        payload = point.payload
        return Outcome(
            id=payload["id"],
            agent=payload["agent"],
            project_id=payload["project_id"],
            task_type=payload.get("task_type") or "general",
            task_description=payload["task_description"],
            success=bool(payload.get("success")),
            strategy_used=payload.get("strategy_used") or "",
            duration_ms=payload.get("duration_ms"),
            error_message=payload.get("error_message"),
            user_feedback=payload.get("user_feedback"),
            timestamp=self._str_to_datetime(payload.get("timestamp"))
            or datetime.now(timezone.utc),
            embedding=getattr(point, "vector", None),
            metadata=payload.get("metadata") or {},
        )

    def _point_to_preference(self, point: Any) -> UserPreference:
        """Convert Qdrant point to UserPreference."""
        payload = point.payload
        return UserPreference(
            id=payload["id"],
            user_id=payload["user_id"],
            category=payload.get("category") or "general",
            preference=payload["preference"],
            source=payload.get("source") or "unknown",
            confidence=payload.get("confidence") or 1.0,
            timestamp=self._str_to_datetime(payload.get("timestamp"))
            or datetime.now(timezone.utc),
            metadata=payload.get("metadata") or {},
        )

    def _point_to_domain_knowledge(self, point: Any) -> DomainKnowledge:
        """Convert Qdrant point to DomainKnowledge."""
        payload = point.payload
        return DomainKnowledge(
            id=payload["id"],
            agent=payload["agent"],
            project_id=payload["project_id"],
            domain=payload.get("domain") or "general",
            fact=payload["fact"],
            source=payload.get("source") or "unknown",
            confidence=payload.get("confidence") or 1.0,
            last_verified=self._str_to_datetime(payload.get("last_verified"))
            or datetime.now(timezone.utc),
            embedding=getattr(point, "vector", None),
            metadata=payload.get("metadata") or {},
        )

    def _point_to_anti_pattern(self, point: Any) -> AntiPattern:
        """Convert Qdrant point to AntiPattern."""
        payload = point.payload
        return AntiPattern(
            id=payload["id"],
            agent=payload["agent"],
            project_id=payload["project_id"],
            pattern=payload["pattern"],
            why_bad=payload.get("why_bad") or "",
            better_alternative=payload.get("better_alternative") or "",
            occurrence_count=payload.get("occurrence_count") or 1,
            last_seen=self._str_to_datetime(payload.get("last_seen"))
            or datetime.now(timezone.utc),
            created_at=self._str_to_datetime(payload.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=getattr(point, "vector", None),
            metadata=payload.get("metadata") or {},
        )

    def close(self) -> None:
        """Close the Qdrant client connection."""
        if self._client:
            self._client.close()
