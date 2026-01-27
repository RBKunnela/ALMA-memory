"""
ALMA Azure Cosmos DB Storage Backend.

Production storage using Azure Cosmos DB with vector search capabilities.
Uses Azure Key Vault for secrets management.

Requirements:
    pip install azure-cosmos azure-identity azure-keyvault-secrets

Configuration (config.yaml):
    alma:
      storage: azure
      azure:
        endpoint: ${AZURE_COSMOS_ENDPOINT}
        key: ${KEYVAULT:cosmos-db-key}
        database: alma-memory
        embedding_dim: 384
"""

import logging
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

# Try to import Azure SDK
try:
    from azure.cosmos import CosmosClient, PartitionKey, exceptions
    from azure.cosmos.container import ContainerProxy
    from azure.cosmos.database import DatabaseProxy

    AZURE_COSMOS_AVAILABLE = True
except ImportError:
    AZURE_COSMOS_AVAILABLE = False
    # Define placeholders for type hints when SDK not available
    CosmosClient = None  # type: ignore
    PartitionKey = None  # type: ignore
    exceptions = None  # type: ignore
    ContainerProxy = Any  # type: ignore
    DatabaseProxy = Any  # type: ignore
    logger.warning(
        "azure-cosmos package not installed. Install with: pip install azure-cosmos"
    )


class AzureCosmosStorage(StorageBackend):
    """
    Azure Cosmos DB storage backend with vector search.

    Uses:
    - NoSQL API for document storage
    - DiskANN vector indexing for similarity search
    - Partition key: project_id for efficient queries

    Container structure:
    - alma-heuristics: Heuristics with vector embeddings
    - alma-outcomes: Task outcomes with vector embeddings
    - alma-preferences: User preferences (no vectors)
    - alma-knowledge: Domain knowledge with vector embeddings
    - alma-antipatterns: Anti-patterns with vector embeddings
    """

    CONTAINER_NAMES = {
        "heuristics": "alma-heuristics",
        "outcomes": "alma-outcomes",
        "preferences": "alma-preferences",
        "knowledge": "alma-knowledge",
        "antipatterns": "alma-antipatterns",
    }

    def __init__(
        self,
        endpoint: str,
        key: str,
        database_name: str = "alma-memory",
        embedding_dim: int = 384,
        create_if_not_exists: bool = True,
    ):
        """
        Initialize Azure Cosmos DB storage.

        Args:
            endpoint: Cosmos DB account endpoint
            key: Cosmos DB account key
            database_name: Name of the database
            embedding_dim: Dimension of embedding vectors
            create_if_not_exists: Create database/containers if missing
        """
        if not AZURE_COSMOS_AVAILABLE:
            raise ImportError(
                "azure-cosmos package required. Install with: pip install azure-cosmos"
            )

        self.endpoint = endpoint
        self.database_name = database_name
        self.embedding_dim = embedding_dim

        # Initialize client
        self.client = CosmosClient(endpoint, credential=key)

        # Get or create database
        if create_if_not_exists:
            self.database = self.client.create_database_if_not_exists(id=database_name)
            self._init_containers()
        else:
            self.database = self.client.get_database_client(database_name)

        # Cache container clients
        self._containers: Dict[str, ContainerProxy] = {}
        for key_name, container_name in self.CONTAINER_NAMES.items():
            self._containers[key_name] = self.database.get_container_client(
                container_name
            )

        logger.info(f"Connected to Azure Cosmos DB: {database_name}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AzureCosmosStorage":
        """Create instance from configuration."""
        azure_config = config.get("azure", {})

        endpoint = azure_config.get("endpoint")
        key = azure_config.get("key")

        if not endpoint or not key:
            raise ValueError(
                "Azure Cosmos DB requires 'azure.endpoint' and 'azure.key' in config"
            )

        return cls(
            endpoint=endpoint,
            key=key,
            database_name=azure_config.get("database", "alma-memory"),
            embedding_dim=azure_config.get("embedding_dim", 384),
            create_if_not_exists=azure_config.get("create_if_not_exists", True),
        )

    def _init_containers(self):
        """Initialize containers with vector search indexing."""
        # Container configs with indexing policies
        container_configs = {
            "heuristics": {
                "partition_key": "/project_id",
                "vector_path": "/embedding",
                "vector_indexes": True,
            },
            "outcomes": {
                "partition_key": "/project_id",
                "vector_path": "/embedding",
                "vector_indexes": True,
            },
            "preferences": {
                "partition_key": "/user_id",
                "vector_path": None,
                "vector_indexes": False,
            },
            "knowledge": {
                "partition_key": "/project_id",
                "vector_path": "/embedding",
                "vector_indexes": True,
            },
            "antipatterns": {
                "partition_key": "/project_id",
                "vector_path": "/embedding",
                "vector_indexes": True,
            },
        }

        for key_name, cfg in container_configs.items():
            container_name = self.CONTAINER_NAMES[key_name]

            # Build indexing policy
            indexing_policy = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
            }

            # Add vector embedding policy if needed
            vector_embedding_policy = None
            if cfg["vector_indexes"] and cfg["vector_path"]:
                # Exclude vector path from regular indexing
                indexing_policy["excludedPaths"].append(
                    {"path": f"{cfg['vector_path']}/*"}
                )

                # Vector embedding policy for DiskANN
                vector_embedding_policy = {
                    "vectorEmbeddings": [
                        {
                            "path": cfg["vector_path"],
                            "dataType": "float32",
                            "dimensions": self.embedding_dim,
                            "distanceFunction": "cosine",
                        }
                    ]
                }

            try:
                container_properties = {
                    "id": container_name,
                    "partition_key": PartitionKey(path=cfg["partition_key"]),
                    "indexing_policy": indexing_policy,
                }

                if vector_embedding_policy:
                    container_properties["vector_embedding_policy"] = (
                        vector_embedding_policy
                    )

                self.database.create_container_if_not_exists(**container_properties)
                logger.debug(f"Container ready: {container_name}")

            except exceptions.CosmosHttpResponseError as e:
                if e.status_code == 409:
                    logger.debug(f"Container already exists: {container_name}")
                else:
                    raise

    def _get_container(self, container_key: str) -> ContainerProxy:
        """Get container client by key."""
        return self._containers[container_key]

    # ==================== WRITE OPERATIONS ====================

    def save_heuristic(self, heuristic: Heuristic) -> str:
        """Save a heuristic."""
        container = self._get_container("heuristics")

        doc = {
            "id": heuristic.id,
            "agent": heuristic.agent,
            "project_id": heuristic.project_id,
            "condition": heuristic.condition,
            "strategy": heuristic.strategy,
            "confidence": heuristic.confidence,
            "occurrence_count": heuristic.occurrence_count,
            "success_count": heuristic.success_count,
            "last_validated": heuristic.last_validated.isoformat()
            if heuristic.last_validated
            else None,
            "created_at": heuristic.created_at.isoformat()
            if heuristic.created_at
            else None,
            "metadata": heuristic.metadata or {},
            "embedding": heuristic.embedding,
            "type": "heuristic",
        }

        container.upsert_item(doc)
        logger.debug(f"Saved heuristic: {heuristic.id}")
        return heuristic.id

    def save_outcome(self, outcome: Outcome) -> str:
        """Save an outcome."""
        container = self._get_container("outcomes")

        doc = {
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
            "timestamp": outcome.timestamp.isoformat() if outcome.timestamp else None,
            "metadata": outcome.metadata or {},
            "embedding": outcome.embedding,
            "type": "outcome",
        }

        container.upsert_item(doc)
        logger.debug(f"Saved outcome: {outcome.id}")
        return outcome.id

    def save_user_preference(self, preference: UserPreference) -> str:
        """Save a user preference."""
        container = self._get_container("preferences")

        doc = {
            "id": preference.id,
            "user_id": preference.user_id,
            "category": preference.category,
            "preference": preference.preference,
            "source": preference.source,
            "confidence": preference.confidence,
            "timestamp": preference.timestamp.isoformat()
            if preference.timestamp
            else None,
            "metadata": preference.metadata or {},
            "type": "preference",
        }

        container.upsert_item(doc)
        logger.debug(f"Saved preference: {preference.id}")
        return preference.id

    def save_domain_knowledge(self, knowledge: DomainKnowledge) -> str:
        """Save domain knowledge."""
        container = self._get_container("knowledge")

        doc = {
            "id": knowledge.id,
            "agent": knowledge.agent,
            "project_id": knowledge.project_id,
            "domain": knowledge.domain,
            "fact": knowledge.fact,
            "source": knowledge.source,
            "confidence": knowledge.confidence,
            "last_verified": knowledge.last_verified.isoformat()
            if knowledge.last_verified
            else None,
            "metadata": knowledge.metadata or {},
            "embedding": knowledge.embedding,
            "type": "domain_knowledge",
        }

        container.upsert_item(doc)
        logger.debug(f"Saved domain knowledge: {knowledge.id}")
        return knowledge.id

    def save_anti_pattern(self, anti_pattern: AntiPattern) -> str:
        """Save an anti-pattern."""
        container = self._get_container("antipatterns")

        doc = {
            "id": anti_pattern.id,
            "agent": anti_pattern.agent,
            "project_id": anti_pattern.project_id,
            "pattern": anti_pattern.pattern,
            "why_bad": anti_pattern.why_bad,
            "better_alternative": anti_pattern.better_alternative,
            "occurrence_count": anti_pattern.occurrence_count,
            "last_seen": anti_pattern.last_seen.isoformat()
            if anti_pattern.last_seen
            else None,
            "created_at": anti_pattern.created_at.isoformat()
            if anti_pattern.created_at
            else None,
            "metadata": anti_pattern.metadata or {},
            "embedding": anti_pattern.embedding,
            "type": "anti_pattern",
        }

        container.upsert_item(doc)
        logger.debug(f"Saved anti-pattern: {anti_pattern.id}")
        return anti_pattern.id

    # ==================== READ OPERATIONS ====================

    def get_heuristics(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Heuristic]:
        """Get heuristics with optional vector search."""
        container = self._get_container("heuristics")

        if embedding:
            # Vector search query
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            AND c.confidence >= @min_confidence
            """
            if agent:
                query += " AND c.agent = @agent"
            query += " ORDER BY VectorDistance(c.embedding, @embedding)"

            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
                {"name": "@min_confidence", "value": min_confidence},
                {"name": "@embedding", "value": embedding},
            ]
            if agent:
                parameters.append({"name": "@agent", "value": agent})

        else:
            # Regular query
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            AND c.confidence >= @min_confidence
            """
            if agent:
                query += " AND c.agent = @agent"
            query += " ORDER BY c.confidence DESC"

            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
                {"name": "@min_confidence", "value": min_confidence},
            ]
            if agent:
                parameters.append({"name": "@agent", "value": agent})

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        return [self._doc_to_heuristic(doc) for doc in items]

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
        container = self._get_container("outcomes")

        if embedding:
            # Vector search query
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
                {"name": "@embedding", "value": embedding},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})
            if task_type:
                query += " AND c.task_type = @task_type"
                parameters.append({"name": "@task_type", "value": task_type})
            if success_only:
                query += " AND c.success = true"

            query += " ORDER BY VectorDistance(c.embedding, @embedding)"

        else:
            # Regular query
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})
            if task_type:
                query += " AND c.task_type = @task_type"
                parameters.append({"name": "@task_type", "value": task_type})
            if success_only:
                query += " AND c.success = true"

            query += " ORDER BY c.timestamp DESC"

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        return [self._doc_to_outcome(doc) for doc in items]

    def get_user_preferences(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[UserPreference]:
        """Get user preferences."""
        container = self._get_container("preferences")

        query = "SELECT * FROM c WHERE c.user_id = @user_id"
        parameters = [{"name": "@user_id", "value": user_id}]

        if category:
            query += " AND c.category = @category"
            parameters.append({"name": "@category", "value": category})

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=user_id,
            )
        )

        return [self._doc_to_preference(doc) for doc in items]

    def get_domain_knowledge(
        self,
        project_id: str,
        agent: Optional[str] = None,
        domain: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[DomainKnowledge]:
        """Get domain knowledge with optional vector search."""
        container = self._get_container("knowledge")

        if embedding:
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
                {"name": "@embedding", "value": embedding},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})
            if domain:
                query += " AND c.domain = @domain"
                parameters.append({"name": "@domain", "value": domain})

            query += " ORDER BY VectorDistance(c.embedding, @embedding)"

        else:
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})
            if domain:
                query += " AND c.domain = @domain"
                parameters.append({"name": "@domain", "value": domain})

            query += " ORDER BY c.confidence DESC"

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        return [self._doc_to_domain_knowledge(doc) for doc in items]

    def get_anti_patterns(
        self,
        project_id: str,
        agent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[AntiPattern]:
        """Get anti-patterns with optional vector search."""
        container = self._get_container("antipatterns")

        if embedding:
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
                {"name": "@embedding", "value": embedding},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})

            query += " ORDER BY VectorDistance(c.embedding, @embedding)"

        else:
            query = """
            SELECT TOP @top_k *
            FROM c
            WHERE c.project_id = @project_id
            """
            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@project_id", "value": project_id},
            ]

            if agent:
                query += " AND c.agent = @agent"
                parameters.append({"name": "@agent", "value": agent})

            query += " ORDER BY c.occurrence_count DESC"

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        return [self._doc_to_anti_pattern(doc) for doc in items]

    # ==================== UPDATE OPERATIONS ====================

    def update_heuristic(
        self,
        heuristic_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a heuristic's fields."""
        container = self._get_container("heuristics")

        # We need project_id to read the item (partition key)
        # First try to find the heuristic
        query = "SELECT * FROM c WHERE c.id = @id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@id", "value": heuristic_id}],
                enable_cross_partition_query=True,
            )
        )

        if not items:
            return False

        doc = items[0]
        doc["project_id"]

        # Apply updates
        for key, value in updates.items():
            if isinstance(value, datetime):
                doc[key] = value.isoformat()
            else:
                doc[key] = value

        container.replace_item(item=heuristic_id, body=doc)
        return True

    def increment_heuristic_occurrence(
        self,
        heuristic_id: str,
        success: bool,
    ) -> bool:
        """Increment heuristic occurrence count."""
        container = self._get_container("heuristics")

        # Find the heuristic
        query = "SELECT * FROM c WHERE c.id = @id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@id", "value": heuristic_id}],
                enable_cross_partition_query=True,
            )
        )

        if not items:
            return False

        doc = items[0]
        doc["occurrence_count"] = doc.get("occurrence_count", 0) + 1
        if success:
            doc["success_count"] = doc.get("success_count", 0) + 1
        doc["last_validated"] = datetime.now(timezone.utc).isoformat()

        container.replace_item(item=heuristic_id, body=doc)
        return True

    def update_heuristic_confidence(
        self,
        heuristic_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update confidence score for a heuristic.

        Note: This requires a cross-partition query since we only have the ID.
        For better performance, consider using update_heuristic() with the
        project_id if available, which enables point reads.
        """
        container = self._get_container("heuristics")

        # Find the heuristic (cross-partition query required without project_id)
        query = "SELECT * FROM c WHERE c.id = @id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@id", "value": heuristic_id}],
                enable_cross_partition_query=True,
            )
        )

        if not items:
            return False

        doc = items[0]
        doc["confidence"] = new_confidence

        container.replace_item(item=heuristic_id, body=doc)
        logger.debug(
            f"Updated heuristic confidence: {heuristic_id} -> {new_confidence}"
        )
        return True

    def update_knowledge_confidence(
        self,
        knowledge_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update confidence score for domain knowledge.

        Note: This requires a cross-partition query since we only have the ID.
        For better performance when project_id is known, fetch the document
        directly using point read and update via save_domain_knowledge().
        """
        container = self._get_container("knowledge")

        # Find the knowledge item (cross-partition query required without project_id)
        query = "SELECT * FROM c WHERE c.id = @id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@id", "value": knowledge_id}],
                enable_cross_partition_query=True,
            )
        )

        if not items:
            return False

        doc = items[0]
        doc["confidence"] = new_confidence

        container.replace_item(item=knowledge_id, body=doc)
        logger.debug(
            f"Updated knowledge confidence: {knowledge_id} -> {new_confidence}"
        )
        return True

    # ==================== DELETE OPERATIONS ====================

    def delete_outcomes_older_than(
        self,
        project_id: str,
        older_than: datetime,
        agent: Optional[str] = None,
    ) -> int:
        """Delete old outcomes."""
        container = self._get_container("outcomes")

        query = """
        SELECT c.id FROM c
        WHERE c.project_id = @project_id
        AND c.timestamp < @older_than
        """
        parameters = [
            {"name": "@project_id", "value": project_id},
            {"name": "@older_than", "value": older_than.isoformat()},
        ]

        if agent:
            query += " AND c.agent = @agent"
            parameters.append({"name": "@agent", "value": agent})

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        deleted = 0
        for item in items:
            try:
                container.delete_item(item=item["id"], partition_key=project_id)
                deleted += 1
            except exceptions.CosmosResourceNotFoundError:
                pass

        logger.info(f"Deleted {deleted} old outcomes")
        return deleted

    def delete_low_confidence_heuristics(
        self,
        project_id: str,
        below_confidence: float,
        agent: Optional[str] = None,
    ) -> int:
        """Delete low-confidence heuristics."""
        container = self._get_container("heuristics")

        query = """
        SELECT c.id FROM c
        WHERE c.project_id = @project_id
        AND c.confidence < @below_confidence
        """
        parameters = [
            {"name": "@project_id", "value": project_id},
            {"name": "@below_confidence", "value": below_confidence},
        ]

        if agent:
            query += " AND c.agent = @agent"
            parameters.append({"name": "@agent", "value": agent})

        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False,
                partition_key=project_id,
            )
        )

        deleted = 0
        for item in items:
            try:
                container.delete_item(item=item["id"], partition_key=project_id)
                deleted += 1
            except exceptions.CosmosResourceNotFoundError:
                pass

        logger.info(f"Deleted {deleted} low-confidence heuristics")
        return deleted

    def delete_heuristic(self, heuristic_id: str) -> bool:
        """Delete a specific heuristic by ID."""
        container = self._get_container("heuristics")

        # Find the heuristic to get project_id
        query = "SELECT c.project_id FROM c WHERE c.id = @id"
        items = list(
            container.query_items(
                query=query,
                parameters=[{"name": "@id", "value": heuristic_id}],
                enable_cross_partition_query=True,
            )
        )

        if not items:
            return False

        project_id = items[0]["project_id"]

        try:
            container.delete_item(item=heuristic_id, partition_key=project_id)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False

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
            "storage_type": "azure_cosmos",
            "database": self.database_name,
        }

        # Count items in each container
        container_keys = ["heuristics", "outcomes", "knowledge", "antipatterns"]
        for key in container_keys:
            container = self._get_container(key)
            query = "SELECT VALUE COUNT(1) FROM c WHERE c.project_id = @project_id"
            parameters = [{"name": "@project_id", "value": project_id}]

            if agent and key != "preferences":
                query = """
                SELECT VALUE COUNT(1) FROM c
                WHERE c.project_id = @project_id AND c.agent = @agent
                """
                parameters.append({"name": "@agent", "value": agent})

            result = list(
                container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=False,
                    partition_key=project_id,
                )
            )
            stats[f"{key}_count"] = result[0] if result else 0

        # Preferences count (no project_id filter)
        container = self._get_container("preferences")
        result = list(
            container.query_items(
                query="SELECT VALUE COUNT(1) FROM c",
                enable_cross_partition_query=True,
            )
        )
        stats["preferences_count"] = result[0] if result else 0

        stats["total_count"] = sum(
            stats.get(k, 0) for k in stats if k.endswith("_count")
        )

        return stats

    # ==================== HELPERS ====================

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from string."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _doc_to_heuristic(self, doc: Dict[str, Any]) -> Heuristic:
        """Convert Cosmos DB document to Heuristic."""
        return Heuristic(
            id=doc["id"],
            agent=doc["agent"],
            project_id=doc["project_id"],
            condition=doc["condition"],
            strategy=doc["strategy"],
            confidence=doc.get("confidence", 0.0),
            occurrence_count=doc.get("occurrence_count", 0),
            success_count=doc.get("success_count", 0),
            last_validated=self._parse_datetime(doc.get("last_validated"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(doc.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=doc.get("embedding"),
            metadata=doc.get("metadata", {}),
        )

    def _doc_to_outcome(self, doc: Dict[str, Any]) -> Outcome:
        """Convert Cosmos DB document to Outcome."""
        return Outcome(
            id=doc["id"],
            agent=doc["agent"],
            project_id=doc["project_id"],
            task_type=doc.get("task_type", "general"),
            task_description=doc["task_description"],
            success=doc.get("success", False),
            strategy_used=doc.get("strategy_used", ""),
            duration_ms=doc.get("duration_ms"),
            error_message=doc.get("error_message"),
            user_feedback=doc.get("user_feedback"),
            timestamp=self._parse_datetime(doc.get("timestamp"))
            or datetime.now(timezone.utc),
            embedding=doc.get("embedding"),
            metadata=doc.get("metadata", {}),
        )

    def _doc_to_preference(self, doc: Dict[str, Any]) -> UserPreference:
        """Convert Cosmos DB document to UserPreference."""
        return UserPreference(
            id=doc["id"],
            user_id=doc["user_id"],
            category=doc.get("category", "general"),
            preference=doc["preference"],
            source=doc.get("source", "unknown"),
            confidence=doc.get("confidence", 1.0),
            timestamp=self._parse_datetime(doc.get("timestamp"))
            or datetime.now(timezone.utc),
            metadata=doc.get("metadata", {}),
        )

    def _doc_to_domain_knowledge(self, doc: Dict[str, Any]) -> DomainKnowledge:
        """Convert Cosmos DB document to DomainKnowledge."""
        return DomainKnowledge(
            id=doc["id"],
            agent=doc["agent"],
            project_id=doc["project_id"],
            domain=doc.get("domain", "general"),
            fact=doc["fact"],
            source=doc.get("source", "unknown"),
            confidence=doc.get("confidence", 1.0),
            last_verified=self._parse_datetime(doc.get("last_verified"))
            or datetime.now(timezone.utc),
            embedding=doc.get("embedding"),
            metadata=doc.get("metadata", {}),
        )

    def _doc_to_anti_pattern(self, doc: Dict[str, Any]) -> AntiPattern:
        """Convert Cosmos DB document to AntiPattern."""
        return AntiPattern(
            id=doc["id"],
            agent=doc["agent"],
            project_id=doc["project_id"],
            pattern=doc["pattern"],
            why_bad=doc.get("why_bad", ""),
            better_alternative=doc.get("better_alternative", ""),
            occurrence_count=doc.get("occurrence_count", 1),
            last_seen=self._parse_datetime(doc.get("last_seen"))
            or datetime.now(timezone.utc),
            created_at=self._parse_datetime(doc.get("created_at"))
            or datetime.now(timezone.utc),
            embedding=doc.get("embedding"),
            metadata=doc.get("metadata", {}),
        )
