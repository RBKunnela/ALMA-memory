"""
ALMA Graph Memory - Abstract Backend Interface.

Defines the abstract base class for graph database backends,
enabling pluggable graph storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from alma.graph.store import Entity, Relationship


class GraphBackend(ABC):
    """
    Abstract base class for graph database backends.

    This interface defines the contract that all graph storage implementations
    must follow, enabling ALMA to work with different graph databases like
    Neo4j, Amazon Neptune, or in-memory stores.
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """
        Add or update an entity in the graph.

        Args:
            entity: The entity to add or update.

        Returns:
            The entity ID.
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add or update a relationship between entities.

        Args:
            relationship: The relationship to add or update.

        Returns:
            The relationship ID.
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_entities(
        self,
        entity_type: Optional[str] = None,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get entities with optional filtering.

        Args:
            entity_type: Filter by entity type (person, organization, etc.).
            project_id: Filter by project ID.
            agent: Filter by agent name.
            limit: Maximum number of entities to return.

        Returns:
            List of matching entities.
        """
        pass

    @abstractmethod
    def get_relationships(self, entity_id: str) -> List[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: The entity ID to get relationships for.

        Returns:
            List of relationships where the entity is source or target.
        """
        pass

    @abstractmethod
    def search_entities(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Entity]:
        """
        Search for entities by name or using vector similarity.

        Args:
            query: Text query to search for in entity names.
            embedding: Optional embedding vector for semantic search.
            top_k: Maximum number of results to return.

        Returns:
            List of matching entities, ordered by relevance.
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: The entity ID to delete.

        Returns:
            True if the entity was deleted, False if not found.
        """
        pass

    @abstractmethod
    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a specific relationship.

        Args:
            relationship_id: The relationship ID to delete.

        Returns:
            True if the relationship was deleted, False if not found.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the backend connection and release resources.

        Should be called when the backend is no longer needed.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close is called."""
        self.close()
        return False
