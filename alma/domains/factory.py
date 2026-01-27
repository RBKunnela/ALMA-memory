"""
Domain Memory Factory.

Factory pattern for creating domain-specific ALMA instances.
"""

import logging
from typing import Any, Dict, List, Optional

from alma.domains.schemas import (
    get_coding_schema,
    get_content_creation_schema,
    get_customer_support_schema,
    get_general_schema,
    get_research_schema,
    get_sales_schema,
)
from alma.domains.types import DomainSchema

logger = logging.getLogger(__name__)


class DomainMemoryFactory:
    """
    Factory for creating domain-specific ALMA instances.

    Provides:
    - Pre-built schemas for common domains
    - Custom schema creation from config
    - ALMA instance creation with domain-specific settings

    Usage:
        # Use pre-built schema
        factory = DomainMemoryFactory()
        schema = factory.get_schema("coding")
        alma = factory.create_alma(schema, "my-project", storage)

        # Create custom schema
        schema = factory.create_schema("my-domain", {
            "description": "Custom domain",
            "entity_types": [...],
            "learning_categories": [...]
        })
    """

    # Registry of pre-built schemas
    _builtin_schemas: Dict[str, callable] = {
        "coding": get_coding_schema,
        "research": get_research_schema,
        "sales": get_sales_schema,
        "general": get_general_schema,
        "customer_support": get_customer_support_schema,
        "content_creation": get_content_creation_schema,
    }

    # Custom schema registry (for user-defined schemas)
    _custom_schemas: Dict[str, DomainSchema] = {}

    def __init__(self):
        """Initialize the factory."""
        pass

    @classmethod
    def list_schemas(cls) -> List[str]:
        """List all available schema names (built-in and custom)."""
        return list(cls._builtin_schemas.keys()) + list(cls._custom_schemas.keys())

    @classmethod
    def get_schema(cls, name: str) -> Optional[DomainSchema]:
        """
        Get a schema by name.

        Args:
            name: Schema name (e.g., "coding", "research")

        Returns:
            DomainSchema or None if not found
        """
        # Check custom schemas first
        if name in cls._custom_schemas:
            return cls._custom_schemas[name]

        # Check built-in schemas
        if name in cls._builtin_schemas:
            return cls._builtin_schemas[name]()

        return None

    @classmethod
    def create_schema(
        cls,
        name: str,
        config: Dict[str, Any],
        register: bool = True,
    ) -> DomainSchema:
        """
        Create a new domain schema from configuration.

        Args:
            name: Schema name
            config: Schema configuration dictionary
            register: If True, register schema for later retrieval

        Config format:
            {
                "description": "Schema description",
                "entity_types": [
                    {"name": "entity1", "description": "...", "attributes": ["a", "b"]}
                ],
                "relationship_types": [
                    {"name": "rel1", "description": "...", "source": "e1", "target": "e2"}
                ],
                "learning_categories": ["cat1", "cat2"],
                "excluded_categories": ["cat3"],
                "min_occurrences_for_heuristic": 3,
                "confidence_decay_days": 30.0
            }

        Returns:
            Created DomainSchema
        """
        schema = DomainSchema.create(
            name=name,
            description=config.get("description", f"Custom schema: {name}"),
            learning_categories=config.get("learning_categories", []),
            excluded_categories=config.get("excluded_categories", []),
            min_occurrences_for_heuristic=config.get("min_occurrences_for_heuristic", 3),
            confidence_decay_days=config.get("confidence_decay_days", 30.0),
        )

        # Add entity types
        for entity_config in config.get("entity_types", []):
            schema.add_entity_type(
                name=entity_config["name"],
                description=entity_config.get("description", ""),
                attributes=entity_config.get("attributes", []),
            )

        # Add relationship types
        for rel_config in config.get("relationship_types", []):
            schema.add_relationship_type(
                name=rel_config["name"],
                description=rel_config.get("description", ""),
                source_type=rel_config.get("source_type", rel_config.get("source", "")),
                target_type=rel_config.get("target_type", rel_config.get("target", "")),
            )

        # Validate
        errors = schema.validate()
        if errors:
            logger.warning(f"Schema validation warnings for '{name}': {errors}")

        # Register if requested
        if register:
            cls._custom_schemas[name] = schema
            logger.info(f"Registered custom schema: {name}")

        return schema

    @classmethod
    def register_schema(cls, schema: DomainSchema) -> None:
        """Register an existing schema for later retrieval."""
        cls._custom_schemas[schema.name] = schema
        logger.info(f"Registered schema: {schema.name}")

    @classmethod
    def unregister_schema(cls, name: str) -> bool:
        """Unregister a custom schema."""
        if name in cls._custom_schemas:
            del cls._custom_schemas[name]
            return True
        return False

    def create_alma(
        self,
        schema: DomainSchema,
        project_id: str,
        storage: Optional[Any] = None,
        embedding_provider: str = "mock",
        **config,
    ) -> Any:
        """
        Create ALMA instance configured for a domain.

        Args:
            schema: Domain schema to use
            project_id: Project identifier
            storage: Optional storage backend (FileBasedStorage created if None)
            embedding_provider: Embedding provider ("mock", "local", "openai")
            **config: Additional ALMA configuration

        Returns:
            Configured ALMA instance

        Note:
            This integrates with the core ALMA class.
            The schema is used to:
            - Configure allowed learning categories
            - Set heuristic thresholds
            - Initialize domain-specific entity tracking
        """
        # Import here to avoid circular dependency
        from alma import ALMA
        from alma.learning import LearningProtocol
        from alma.retrieval import RetrievalEngine
        from alma.storage.file_based import FileBasedStorage
        from alma.types import MemoryScope

        # Create storage if not provided
        if storage is None:
            import tempfile
            from pathlib import Path
            storage_dir = Path(tempfile.mkdtemp()) / ".alma" / project_id
            storage = FileBasedStorage(storage_dir)

        # Create retrieval engine
        retrieval = RetrievalEngine(
            storage=storage,
            embedding_provider=embedding_provider,
        )

        # Create scope based on schema
        default_agent = config.get("default_agent", "agent")
        scopes = {
            default_agent: MemoryScope(
                agent_name=default_agent,
                can_learn=schema.learning_categories,
                cannot_learn=schema.excluded_categories,
                min_occurrences_for_heuristic=schema.min_occurrences_for_heuristic,
            )
        }

        # Create learning protocol
        learning = LearningProtocol(
            storage=storage,
            scopes=scopes,
        )

        # Create ALMA instance
        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id=project_id,
        )

        # Store schema reference for domain-aware operations
        alma._domain_schema = schema

        logger.info(
            f"Created ALMA instance for project '{project_id}' "
            f"with domain schema '{schema.name}'"
        )

        return alma

    def create_alma_for_agent(
        self,
        schema_name: str,
        agent: str,
        project_id: str,
        storage: Optional[Any] = None,
        scope_restrictions: Optional[List[str]] = None,
        **config,
    ) -> Any:
        """
        Create ALMA instance for a specific agent.

        This is a convenience method that:
        1. Gets the appropriate schema
        2. Creates ALMA instance
        3. Configures agent-specific settings

        Args:
            schema_name: Name of schema to use
            agent: Agent identifier
            project_id: Project identifier
            storage: Optional storage backend
            scope_restrictions: Categories agent cannot learn from
            **config: Additional ALMA configuration

        Returns:
            Configured ALMA instance for the agent
        """
        schema = self.get_schema(schema_name)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_name}")

        # Apply scope restrictions
        if scope_restrictions:
            for cat in scope_restrictions:
                schema.add_excluded_category(cat)

        # Create ALMA with agent config
        alma = self.create_alma(
            schema=schema,
            project_id=project_id,
            storage=storage,
            default_agent=agent,
            **config,
        )

        return alma


# Convenience functions for common patterns
def create_coding_alma(
    project_id: str,
    agent: str = "developer",
    storage: Optional[Any] = None,
    **config,
) -> Any:
    """Create ALMA configured for coding workflows."""
    factory = DomainMemoryFactory()
    return factory.create_alma_for_agent(
        schema_name="coding",
        agent=agent,
        project_id=project_id,
        storage=storage,
        **config,
    )


def create_research_alma(
    project_id: str,
    agent: str = "researcher",
    storage: Optional[Any] = None,
    **config,
) -> Any:
    """Create ALMA configured for research workflows."""
    factory = DomainMemoryFactory()
    return factory.create_alma_for_agent(
        schema_name="research",
        agent=agent,
        project_id=project_id,
        storage=storage,
        **config,
    )


def create_general_alma(
    project_id: str,
    agent: str = "assistant",
    storage: Optional[Any] = None,
    **config,
) -> Any:
    """Create ALMA configured for general-purpose agents."""
    factory = DomainMemoryFactory()
    return factory.create_alma_for_agent(
        schema_name="general",
        agent=agent,
        project_id=project_id,
        storage=storage,
        **config,
    )
