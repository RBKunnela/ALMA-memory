"""
ALMA Pinecone Storage Backend.

Production-ready storage using Pinecone vector database for
native vector similarity search with serverless infrastructure.

Recommended for:
- Cloud-native deployments
- Large-scale vector search workloads
- Serverless architecture preferences
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

# Try to import pinecone
try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore
    logger.warning(
        "pinecone not installed. Install with: pip install 'alma-memory[pinecone]'"
    )

# Namespace constants for memory types
NAMESPACE_HEURISTICS = "heuristics"
NAMESPACE_OUTCOMES = "outcomes"
NAMESPACE_DOMAIN_KNOWLEDGE = "domain_knowledge"
NAMESPACE_ANTI_PATTERNS = "anti_patterns"
NAMESPACE_PREFERENCES = "preferences"


class PineconeStorage(StorageBackend):
    """
    Pinecone storage backend for ALMA.

    Uses Pinecone's vector database with namespaces for different memory types.
    Supports serverless deployment for automatic scaling.

    Features:
        - One namespace per memory type (heuristics, outcomes, etc.)
        - Embeddings stored as vectors with metadata
        - Efficient vector similarity search
        - Automatic index creation with serverless spec

    Usage:
        storage = PineconeStorage(
            api_key="your-api-key",
            index_name="alma-memory",
            embedding_dim=384,
        )
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "alma-memory",
        embedding_dim: int = 384,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
    ):
        """
        Initialize Pinecone storage.

        Args:
            api_key: Pinecone API key (required)
            index_name: Name of the Pinecone index (default: alma-memory)
            embedding_dim: Dimension of embedding vectors (default: 384)
            cloud: Cloud provider for serverless (default: aws)
            region: Cloud region for serverless (default: us-east-1)
            metric: Distance metric (default: cosine)
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone not installed. Install with: pip install 'alma-memory[pinecone]'"
            )

        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.cloud = cloud
        self.region = region
        self.metric = metric

        # Initialize Pinecone client
        self._pc = Pinecone(api_key=api_key)

        # Create or get index
        self._init_index()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PineconeStorage":
        """Create instance from configuration."""
        pinecone_config = config.get("pinecone", {})

        # Support environment variable expansion
        def get_value(key: str, default: Any = None) -> Any:
            value = pinecone_config.get(key, default)
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                return os.environ.get(env_var, default)
            return value

        return cls(
            api_key=get_value("api_key", os.environ.get("PINECONE_API_KEY", "")),
            index_name=get_value("index_name", "alma-memory"),
            embedding_dim=int(config.get("embedding_dim", 384)),
            cloud=get_value("cloud", "aws"),
            region=get_value("region", "us-east-1"),
            metric=get_value("metric", "cosine"),
        )

    def _init_index(self):
        """Initialize or get the Pinecone index."""
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self._pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            logger.info(f"Created index: {self.index_name}")

        self._index = self._pc.Index(self.index_name)

    def _get_zero_vector(self) -> List[float]:
        """Get a zero vector for records without embeddings."""
        return [0.0] * self.embedding_dim

    def _metadata_to_pinecone(self, obj: Any, memory_type: str) -> Dict[str, Any]:
        """Convert a memory object to Pinecone metadata format."""
        # Pinecone metadata must be flat (no nested dicts/lists of dicts)
        # and values must be strings, numbers, booleans, or lists of strings
        if memory_type == NAMESPACE_HEURISTICS:
            return {
                "agent": obj.agent,
                "project_id": obj.project_id,
                "condition": obj.condition,
                "strategy": obj.strategy,
                "confidence": float(obj.confidence),
                "occurrence_count": int(obj.occurrence_count),
                "success_count": int(obj.success_count),
                "last_validated": obj.last_validated.isoformat()
                if obj.last_validated
                else "",
                "created_at": obj.created_at.isoformat() if obj.created_at else "",
                "metadata_json": json.dumps(obj.metadata) if obj.metadata else "{}",
            }
        elif memory_type == NAMESPACE_OUTCOMES:
            return {
                "agent": obj.agent,
                "project_id": obj.project_id,
                "task_type": obj.task_type or "general",
                "task_description": obj.task_description,
                "success": obj.success,
                "strategy_used": obj.strategy_used or "",
                "duration_ms": int(obj.duration_ms) if obj.duration_ms else 0,
                "error_message": obj.error_message or "",
                "user_feedback": obj.user_feedback or "",
                "timestamp": obj.timestamp.isoformat() if obj.timestamp else "",
                "metadata_json": json.dumps(obj.metadata) if obj.metadata else "{}",
            }
        elif memory_type == NAMESPACE_PREFERENCES:
            return {
                "user_id": obj.user_id,
                "category": obj.category or "general",
                "preference": obj.preference,
                "source": obj.source or "unknown",
                "confidence": float(obj.confidence),
                "timestamp": obj.timestamp.isoformat() if obj.timestamp else "",
                "metadata_json": json.dumps(obj.metadata) if obj.metadata else "{}",
            }
        elif memory_type == NAMESPACE_DOMAIN_KNOWLEDGE:
            return {
                "agent": obj.agent,
                "project_id": obj.project_id,
                "domain": obj.domain or "general",
                "fact": obj.fact,
                "source": obj.source or "unknown",
                "confidence": float(obj.confidence),
                "last_verified": obj.last_verified.isoformat()
                if obj.last_verified
                else "",
                "metadata_json": json.dumps(obj.metadata) if obj.metadata else "{}",
            }
        elif memory_type == NAMESPACE_ANTI_PATTERNS:
            return {
                "agent": obj.agent,
                "project_id": obj.project_id,
                "pattern": obj.pattern,
                "why_bad": obj.why_bad or "",
                "better_alternative": obj.better_alternative or "",
                "occurrence_count": int(obj.occurrence_count),
                "last_seen": obj.last_seen.isoformat() if obj.last_seen else "",
                "created_at": obj.created_at.isoformat() if obj.created_at else "",
                "metadata_json": json.dumps(obj.metadata) if obj.metadata else "{}",
            }
        return {}

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from string."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _metadata_to_heuristic(self, id: str, metadata: Dict[str, Any]) -> Heuristic:
        """Convert Pinecone metadata to Heuristic."""
        return Heuristic(
            id=id,
            agent=metadata.get("agent", ""),
            project_id=metadata.get("project_id", ""),
            condition=metadata.get("condition", ""),
            strategy=metadata.get("strategy", ""),
            confidence=float(metadata.get("confidence", 0.0)),
            occurrence_count=int(metadata.get("occurrence_count", 0)),
            success_count=int(metadata.get("success_count", 0)),
            last_validated=self._parse_datetime(metadata.get("last_validated"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(metadata.get("created_at"))
            or datetime.now(timezone.utc),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
        )

    def _metadata_to_outcome(self, id: str, metadata: Dict[str, Any]) -> Outcome:
        """Convert Pinecone metadata to Outcome."""
        return Outcome(
            id=id,
            agent=metadata.get("agent", ""),
            project_id=metadata.get("project_id", ""),
            task_type=metadata.get("task_type", "general"),
            task_description=metadata.get("task_description", ""),
            success=bool(metadata.get("success", False)),
            strategy_used=metadata.get("strategy_used", ""),
            duration_ms=int(metadata.get("duration_ms", 0)) or None,
            error_message=metadata.get("error_message") or None,
            user_feedback=metadata.get("user_feedback") or None,
            timestamp=self._parse_datetime(metadata.get("timestamp"))
            or datetime.now(timezone.utc),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
        )

    def _metadata_to_preference(
        self, id: str, metadata: Dict[str, Any]
    ) -> UserPreference:
        """Convert Pinecone metadata to UserPreference."""
        return UserPreference(
            id=id,
            user_id=metadata.get("user_id", ""),
            category=metadata.get("category", "general"),
            preference=metadata.get("preference", ""),
            source=metadata.get("source", "unknown"),
            confidence=float(metadata.get("confidence", 1.0)),
            timestamp=self._parse_datetime(metadata.get("timestamp"))
            or datetime.now(timezone.utc),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
        )

    def _metadata_to_domain_knowledge(
        self, id: str, metadata: Dict[str, Any]
    ) -> DomainKnowledge:
        """Convert Pinecone metadata to DomainKnowledge."""
        return DomainKnowledge(
            id=id,
            agent=metadata.get("agent", ""),
            project_id=metadata.get("project_id", ""),
            domain=metadata.get("domain", "general"),
            fact=metadata.get("fact", ""),
            source=metadata.get("source", "unknown"),
            confidence=float(metadata.get("confidence", 1.0)),
            last_verified=self._parse_datetime(metadata.get("last_verified"))
            or datetime.now(timezone.utc),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
        )

    def _metadata_to_anti_pattern(
        self, id: str, metadata: Dict[str, Any]
    ) -> AntiPattern:
        """Convert Pinecone metadata to AntiPattern."""
        return AntiPattern(
            id=id,
            agent=metadata.get("agent", ""),
            project_id=metadata.get("project_id", ""),
            pattern=metadata.get("pattern", ""),
            why_bad=metadata.get("why_bad", ""),
            better_alternative=metadata.get("better_alternative", ""),
            occurrence_count=int(metadata.get("occurrence_count", 1)),
            last_seen=self._parse_datetime(metadata.get("last_seen"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(metadata.get("created_at"))
            or datetime.now(timezone.utc),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
        )

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        vector = heuristic.embedding or self._get_zero_vector()
        metadata = self._metadata_to_pinecone(heuristic, NAMESPACE_HEURISTICS)

        self._index.upsert(
            vectors=[{"id": heuristic.id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE_HEURISTICS,
        )

        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        vector = outcome.embedding or self._get_zero_vector()
        metadata = self._metadata_to_pinecone(outcome, NAMESPACE_OUTCOMES)

        self._index.upsert(
            vectors=[{"id": outcome.id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE_OUTCOMES,
        )

        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        # User preferences don't typically have embeddings
        vector = self._get_zero_vector()
        metadata = self._metadata_to_pinecone(preference, NAMESPACE_PREFERENCES)

        self._index.upsert(
            vectors=[{"id": preference.id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE_PREFERENCES,
        )

        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        vector = knowledge.embedding or self._get_zero_vector()
        metadata = self._metadata_to_pinecone(knowledge, NAMESPACE_DOMAIN_KNOWLEDGE)

        self._index.upsert(
            vectors=[{"id": knowledge.id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE_DOMAIN_KNOWLEDGE,
        )

        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        vector = anti_pattern.embedding or self._get_zero_vector()
        metadata = self._metadata_to_pinecone(anti_pattern, NAMESPACE_ANTI_PATTERNS)

        self._index.upsert(
            vectors=[{"id": anti_pattern.id, "values": vector, "metadata": metadata}],
            namespace=NAMESPACE_ANTI_PATTERNS,
        )

        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== BATCH WRITE OPERATIONS ====================

    def save_heuristics(self, heuristics: List[Heuristic]) -> List[str]:
        """Save multiple heuristics in a batch."""
        if not heuristics:
            return []

        vectors = []
        for h in heuristics:
            vector = h.embedding or self._get_zero_vector()
            metadata = self._metadata_to_pinecone(h, NAMESPACE_HEURISTICS)
            vectors.append({"id": h.id, "values": vector, "metadata": metadata})

        # Pinecone supports batches of up to 100 vectors
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=NAMESPACE_HEURISTICS)

        logger.debug(f"Batch saved {len(heuristics)} heuristics")
        return [h.id for h in heuristics]

    def save_outcomes(self, outcomes: List[Outcome]) -> List[str]:
        """Save multiple outcomes in a batch."""
        if not outcomes:
            return []

        vectors = []
        for o in outcomes:
            vector = o.embedding or self._get_zero_vector()
            metadata = self._metadata_to_pinecone(o, NAMESPACE_OUTCOMES)
            vectors.append({"id": o.id, "values": vector, "metadata": metadata})

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=NAMESPACE_OUTCOMES)

        logger.debug(f"Batch saved {len(outcomes)} outcomes")
        return [o.id for o in outcomes]

    def save_domain_knowledge_batch(
        self, knowledge_items: List[DomainKnowledge]
    ) -> List[str]:
        """Save multiple domain knowledge items in a batch."""
        if not knowledge_items:
            return []

        vectors = []
        for k in knowledge_items:
            vector = k.embedding or self._get_zero_vector()
            metadata = self._metadata_to_pinecone(k, NAMESPACE_DOMAIN_KNOWLEDGE)
            vectors.append({"id": k.id, "values": vector, "metadata": metadata})

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=NAMESPACE_DOMAIN_KNOWLEDGE)

        logger.debug(f"Batch saved {len(knowledge_items)} domain knowledge items")
        return [k.id for k in knowledge_items]

    # ==================== READ OPERATIONS ====================

    def _build_filter(
        self,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        success_only: bool = False,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """Build Pinecone metadata filter."""
        conditions = []

        if project_id:
            conditions.append({"project_id": {"$eq": project_id}})

        if agent:
            conditions.append({"agent": {"$eq": agent}})

        if user_id:
            conditions.append({"user_id": {"$eq": user_id}})

        if task_type:
            conditions.append({"task_type": {"$eq": task_type}})

        if domain:
            conditions.append({"domain": {"$eq": domain}})

        if category:
            conditions.append({"category": {"$eq": category}})

        if success_only:
            conditions.append({"success": {"$eq": True}})

        if min_confidence > 0.0:
            conditions.append({"confidence": {"$gte": min_confidence}})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics with optional vector search."""
        filter_dict = self._build_filter(
            project_id=project_id,
            agent=agent,
            min_confidence=min_confidence,
        )

        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE_HEURISTICS,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            self._metadata_to_heuristic(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

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
        filter_dict = self._build_filter(
            project_id=project_id,
            agent=agent,
            task_type=task_type,
            success_only=success_only,
        )

        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE_OUTCOMES,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            self._metadata_to_outcome(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        filter_dict = self._build_filter(
            user_id=user_id,
            category=category,
        )

        # For preferences, we use a zero vector query since we filter by metadata
        query_vector = self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=100,  # Get all preferences for user
            namespace=NAMESPACE_PREFERENCES,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            self._metadata_to_preference(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge with optional vector search."""
        filter_dict = self._build_filter(
            project_id=project_id,
            agent=agent,
            domain=domain,
        )

        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE_DOMAIN_KNOWLEDGE,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            self._metadata_to_domain_knowledge(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        filter_dict = self._build_filter(
            project_id=project_id,
            agent=agent,
        )

        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE_ANTI_PATTERNS,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        return [
            self._metadata_to_anti_pattern(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    # ==================== MULTI-AGENT MEMORY SHARING ====================

    def get_heuristics_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics from multiple agents using $in filter."""
        if not agents:
            return []

        conditions = [
            {"project_id": {"$eq": project_id}},
            {"agent": {"$in": agents}},
        ]

        if min_confidence > 0.0:
            conditions.append({"confidence": {"$gte": min_confidence}})

        filter_dict = {"$and": conditions}
        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k * len(agents),
            namespace=NAMESPACE_HEURISTICS,
            filter=filter_dict,
            include_metadata=True,
        )

        return [
            self._metadata_to_heuristic(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_outcomes_for_agents(
        self,
        project_id: str,
        agents: List[str],
        task_type: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        success_only: bool = False,
    ) -> List[Outcome]:
        """Get outcomes from multiple agents using $in filter."""
        if not agents:
            return []

        conditions = [
            {"project_id": {"$eq": project_id}},
            {"agent": {"$in": agents}},
        ]

        if task_type:
            conditions.append({"task_type": {"$eq": task_type}})

        if success_only:
            conditions.append({"success": {"$eq": True}})

        filter_dict = {"$and": conditions}
        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k * len(agents),
            namespace=NAMESPACE_OUTCOMES,
            filter=filter_dict,
            include_metadata=True,
        )

        return [
            self._metadata_to_outcome(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_domain_knowledge_for_agents(
        self,
        project_id: str,
        agents: List[str],
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge from multiple agents using $in filter."""
        if not agents:
            return []

        conditions = [
            {"project_id": {"$eq": project_id}},
            {"agent": {"$in": agents}},
        ]

        if domain:
            conditions.append({"domain": {"$eq": domain}})

        filter_dict = {"$and": conditions}
        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k * len(agents),
            namespace=NAMESPACE_DOMAIN_KNOWLEDGE,
            filter=filter_dict,
            include_metadata=True,
        )

        return [
            self._metadata_to_domain_knowledge(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    def get_anti_patterns_for_agents(
        self,
        project_id: str,
        agents: List[str],
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns from multiple agents using $in filter."""
        if not agents:
            return []

        conditions = [
            {"project_id": {"$eq": project_id}},
            {"agent": {"$in": agents}},
        ]

        filter_dict = {"$and": conditions}
        query_vector = embedding or self._get_zero_vector()

        results = self._index.query(
            vector=query_vector,
            top_k=top_k * len(agents),
            namespace=NAMESPACE_ANTI_PATTERNS,
            filter=filter_dict,
            include_metadata=True,
        )

        return [
            self._metadata_to_anti_pattern(match["id"], match.get("metadata", {}))
            for match in results.get("matches", [])
        ]

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        if not updates:
            return False

        # Fetch existing record
        results = self._index.fetch(ids=[heuristic_id], namespace=NAMESPACE_HEURISTICS)

        if heuristic_id not in results.get("vectors", {}):
            return False

        existing = results["vectors"][heuristic_id]
        metadata = existing.get("metadata", {})

        # Apply updates to metadata
        for key, value in updates.items():
            if key == "metadata":
                metadata["metadata_json"] = json.dumps(value) if value else "{}"
            elif isinstance(value, datetime):
                metadata[key] = value.isoformat()
            else:
                metadata[key] = value

        # Upsert with updated metadata
        self._index.upsert(
            vectors=[
                {
                    "id": heuristic_id,
                    "values": existing.get("values", self._get_zero_vector()),
                    "metadata": metadata,
                }
            ],
            namespace=NAMESPACE_HEURISTICS,
        )

        return True

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        # Fetch existing record
        results = self._index.fetch(ids=[heuristic_id], namespace=NAMESPACE_HEURISTICS)

        if heuristic_id not in results.get("vectors", {}):
            return False

        existing = results["vectors"][heuristic_id]
        metadata = existing.get("metadata", {})

        # Increment counts
        metadata["occurrence_count"] = int(metadata.get("occurrence_count", 0)) + 1
        if success:
            metadata["success_count"] = int(metadata.get("success_count", 0)) + 1
        metadata["last_validated"] = datetime.now(timezone.utc).isoformat()

        # Upsert with updated metadata
        self._index.upsert(
            vectors=[
                {
                    "id": heuristic_id,
                    "values": existing.get("values", self._get_zero_vector()),
                    "metadata": metadata,
                }
            ],
            namespace=NAMESPACE_HEURISTICS,
        )

        return True

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
        # Fetch existing record
        results = self._index.fetch(
            ids=[knowledge_id], namespace=NAMESPACE_DOMAIN_KNOWLEDGE
        )

        if knowledge_id not in results.get("vectors", {}):
            return False

        existing = results["vectors"][knowledge_id]
        metadata = existing.get("metadata", {})
        metadata["confidence"] = new_confidence

        # Upsert with updated metadata
        self._index.upsert(
            vectors=[
                {
                    "id": knowledge_id,
                    "values": existing.get("values", self._get_zero_vector()),
                    "metadata": metadata,
                }
            ],
            namespace=NAMESPACE_DOMAIN_KNOWLEDGE,
        )

        return True

    # ==================== DELETE OPERATIONS ====================

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a heuristic by ID."""
        try:
            self._index.delete(ids=[heuristic_id], namespace=NAMESPACE_HEURISTICS)
            logger.debug(f"Deleted heuristic: {heuristic_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete heuristic {heuristic_id}: {e}")
            return False

    def delete_outcome(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        try:
            self._index.delete(ids=[outcome_id], namespace=NAMESPACE_OUTCOMES)
            logger.debug(f"Deleted outcome: {outcome_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete outcome {outcome_id}: {e}")
            return False

    def delete_domain_knowledge(self, knowledge_id: str) -> bool:
        """Delete domain knowledge by ID."""
        try:
            self._index.delete(ids=[knowledge_id], namespace=NAMESPACE_DOMAIN_KNOWLEDGE)
            logger.debug(f"Deleted domain knowledge: {knowledge_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete domain knowledge {knowledge_id}: {e}")
            return False

    def delete_anti_pattern(self, anti_pattern_id: str) -> bool:
        """Delete an anti-pattern by ID."""
        try:
            self._index.delete(ids=[anti_pattern_id], namespace=NAMESPACE_ANTI_PATTERNS)
            logger.debug(f"Deleted anti-pattern: {anti_pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete anti-pattern {anti_pattern_id}: {e}")
            return False

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes.

        Note: Pinecone doesn't support bulk delete by filter directly,
        so we query first then delete by IDs.
        """
        filter_dict = self._build_filter(project_id=project_id, agent=agent)
        query_vector = self._get_zero_vector()

        # Query to get all matching IDs
        results = self._index.query(
            vector=query_vector,
            top_k=10000,  # Large number to get all
            namespace=NAMESPACE_OUTCOMES,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        older_than_iso = older_than.isoformat()
        ids_to_delete = []

        for match in results.get("matches", []):
            timestamp = match.get("metadata", {}).get("timestamp", "")
            if timestamp and timestamp < older_than_iso:
                ids_to_delete.append(match["id"])

        if ids_to_delete:
            # Delete in batches of 1000
            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i : i + batch_size]
                self._index.delete(ids=batch, namespace=NAMESPACE_OUTCOMES)

        deleted = len(ids_to_delete)
        logger.info(f"Deleted {deleted} old outcomes")
        return deleted

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        filter_dict = self._build_filter(project_id=project_id, agent=agent)
        query_vector = self._get_zero_vector()

        # Query to get all matching IDs
        results = self._index.query(
            vector=query_vector,
            top_k=10000,
            namespace=NAMESPACE_HEURISTICS,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        ids_to_delete = []
        for match in results.get("matches", []):
            confidence = float(match.get("metadata", {}).get("confidence", 0.0))
            if confidence < below_confidence:
                ids_to_delete.append(match["id"])

        if ids_to_delete:
            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i : i + batch_size]
                self._index.delete(ids=batch, namespace=NAMESPACE_HEURISTICS)

        deleted = len(ids_to_delete)
        logger.info(f"Deleted {deleted} low-confidence heuristics")
        return deleted

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
            "storage_type": "pinecone",
            "index_name": self.index_name,
        }

        # Get index stats
        try:
            index_stats = self._index.describe_index_stats()

            # Count by namespace
            namespaces = index_stats.get("namespaces", {})

            stats["heuristics_count"] = namespaces.get(NAMESPACE_HEURISTICS, {}).get(
                "vector_count", 0
            )
            stats["outcomes_count"] = namespaces.get(NAMESPACE_OUTCOMES, {}).get(
                "vector_count", 0
            )
            stats["domain_knowledge_count"] = namespaces.get(
                NAMESPACE_DOMAIN_KNOWLEDGE, {}
            ).get("vector_count", 0)
            stats["anti_patterns_count"] = namespaces.get(
                NAMESPACE_ANTI_PATTERNS, {}
            ).get("vector_count", 0)
            stats["preferences_count"] = namespaces.get(NAMESPACE_PREFERENCES, {}).get(
                "vector_count", 0
            )
            stats["total_count"] = index_stats.get("total_vector_count", 0)

        except Exception as e:
            logger.warning(f"Failed to get index stats: {e}")
            stats["error"] = str(e)

        return stats

    def close(self):
        """Close the Pinecone connection (no-op for Pinecone client)."""
        # Pinecone client doesn't require explicit cleanup
        pass
