"""
ALMA Graph Memory - In-Memory Backend.

In-memory implementation of the GraphBackend interface for testing and development.
"""

import logging
from typing import Dict, List, Optional, Set

from alma.graph.base import GraphBackend
from alma.graph.store import Entity, Relationship

logger = logging.getLogger(__name__)


class InMemoryBackend(GraphBackend):
    """
    In-memory graph database backend.

    Suitable for testing, development, and small-scale use cases
    where persistence is not required.

    No external dependencies required.
    """

    def __init__(self):
        """Initialize empty in-memory storage."""
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._outgoing: Dict[str, List[str]] = {}  # entity_id -> [rel_ids]
        self._incoming: Dict[str, List[str]] = {}  # entity_id -> [rel_ids]

    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity."""
        self._entities[entity.id] = entity
        if entity.id not in self._outgoing:
            self._outgoing[entity.id] = []
        if entity.id not in self._incoming:
            self._incoming[entity.id] = []
        return entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add or update a relationship."""
        self._relationships[relationship.id] = relationship

        if relationship.source_id not in self._outgoing:
            self._outgoing[relationship.source_id] = []
        if relationship.id not in self._outgoing[relationship.source_id]:
            self._outgoing[relationship.source_id].append(relationship.id)

        if relationship.target_id not in self._incoming:
            self._incoming[relationship.target_id] = []
        if relationship.id not in self._incoming[relationship.target_id]:
            self._incoming[relationship.target_id].append(relationship.id)

        return relationship.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Get entities with optional filtering."""
        results = []
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if project_id and entity.properties.get("project_id") != project_id:
                continue
            if agent and entity.properties.get("agent") != agent:
                continue
            results.append(entity)
            if len(results) >= limit:
                break
        return results

    def get_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity (both directions)."""
        rel_ids: Set[str] = set()
        rel_ids.update(self._outgoing.get(entity_id, []))
        rel_ids.update(self._incoming.get(entity_id, []))

        return [
            self._relationships[rid]
            for rid in rel_ids
            if rid in self._relationships
        ]

    def search_entities(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Entity]:
        """
        Search for entities by name.

        Note: Vector similarity search is not implemented for in-memory backend.
        Falls back to case-insensitive text search.
        """
        query_lower = query.lower()
        results = []
        for entity in self._entities.values():
            if query_lower in entity.name.lower():
                results.append(entity)
                if len(results) >= top_k:
                    break
        return results

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        if entity_id not in self._entities:
            return False

        # Delete outgoing relationships
        for rel_id in list(self._outgoing.get(entity_id, [])):
            if rel_id in self._relationships:
                rel = self._relationships[rel_id]
                # Remove from target's incoming
                if rel.target_id in self._incoming:
                    if rel_id in self._incoming[rel.target_id]:
                        self._incoming[rel.target_id].remove(rel_id)
                del self._relationships[rel_id]

        # Delete incoming relationships
        for rel_id in list(self._incoming.get(entity_id, [])):
            if rel_id in self._relationships:
                rel = self._relationships[rel_id]
                # Remove from source's outgoing
                if rel.source_id in self._outgoing:
                    if rel_id in self._outgoing[rel.source_id]:
                        self._outgoing[rel.source_id].remove(rel_id)
                del self._relationships[rel_id]

        # Delete entity
        del self._entities[entity_id]
        self._outgoing.pop(entity_id, None)
        self._incoming.pop(entity_id, None)

        return True

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a specific relationship by ID."""
        if relationship_id not in self._relationships:
            return False

        rel = self._relationships[relationship_id]

        # Remove from source's outgoing
        if rel.source_id in self._outgoing:
            if relationship_id in self._outgoing[rel.source_id]:
                self._outgoing[rel.source_id].remove(relationship_id)

        # Remove from target's incoming
        if rel.target_id in self._incoming:
            if relationship_id in self._incoming[rel.target_id]:
                self._incoming[rel.target_id].remove(relationship_id)

        del self._relationships[relationship_id]
        return True

    def close(self) -> None:
        """Clear all data (no-op for in-memory backend)."""
        # In-memory backend doesn't need explicit cleanup
        # but we can optionally clear data
        pass

    def clear(self) -> None:
        """Clear all stored data."""
        self._entities.clear()
        self._relationships.clear()
        self._outgoing.clear()
        self._incoming.clear()

    # Additional methods for compatibility with existing GraphStore API

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Find entities by name or type.

        This method provides compatibility with the existing GraphStore API.
        """
        results = []
        for entity in self._entities.values():
            if name and name.lower() not in entity.name.lower():
                continue
            if entity_type and entity.entity_type != entity_type:
                continue
            results.append(entity)
            if len(results) >= limit:
                break
        return results

    def get_relationships_directional(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get relationships for an entity with direction control.

        This method provides compatibility with the existing GraphStore API.

        Args:
            entity_id: The entity ID.
            direction: "outgoing", "incoming", or "both".
            relation_type: Optional filter by relationship type.

        Returns:
            List of matching relationships.
        """
        rel_ids: Set[str] = set()

        if direction in ("outgoing", "both"):
            rel_ids.update(self._outgoing.get(entity_id, []))
        if direction in ("incoming", "both"):
            rel_ids.update(self._incoming.get(entity_id, []))

        results = []
        for rel_id in rel_ids:
            rel = self._relationships.get(rel_id)
            if rel:
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)
        return results
