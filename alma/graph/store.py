"""
ALMA Graph Memory Module.

Graph-based memory storage for capturing relationships between entities.
Supports Neo4j and in-memory graph for testing.

This module provides two APIs:
1. GraphStore (high-level) - Full-featured graph store with traversal and query support
2. GraphBackend (low-level) - Simple CRUD operations for pluggable backends

The GraphStore classes can optionally use GraphBackend implementations for storage,
enabling easy swapping of database backends while keeping the high-level API.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from alma.graph.base import GraphBackend

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    name: str
    entity_type: str  # person, organization, concept, tool, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Relationship:
    """An edge in the knowledge graph."""

    id: str
    source_id: str
    target_id: str
    relation_type: str  # WORKS_AT, USES, KNOWS, CREATED_BY, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GraphQuery:
    """A query against the knowledge graph."""

    entities: List[str]  # Entity names or IDs to search for
    relation_types: Optional[List[str]] = None  # Filter by relation types
    max_hops: int = 2  # Maximum traversal depth
    limit: int = 20  # Maximum results


@dataclass
class GraphResult:
    """Result from a graph query."""

    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]]  # Paths through the graph
    query_time_ms: int


class GraphStore(ABC):
    """
    Abstract base class for graph storage backends.

    This is the high-level API for graph operations including traversal
    and complex queries. For simple CRUD operations, see GraphBackend.
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity."""
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> str:
        """Add or update a relationship."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        pass

    @abstractmethod
    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Find entities by name or type."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        pass

    @abstractmethod
    def traverse(
        self,
        start_entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> GraphResult:
        """Traverse the graph from a starting entity."""
        pass

    @abstractmethod
    def query(self, query: GraphQuery) -> GraphResult:
        """Execute a graph query."""
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        pass

    def close(self) -> None:  # noqa: B027
        """Close the store connection. Override in subclasses if needed."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close is called."""
        self.close()
        return False


class Neo4jGraphStore(GraphStore):
    """
    Neo4j graph storage backend.

    Requires neo4j Python driver: pip install neo4j
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (bolt:// or neo4j+s://)
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def _get_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                )
            except ImportError as err:
                raise ImportError(
                    "neo4j package required for Neo4j graph store. "
                    "Install with: pip install neo4j"
                ) from err
        return self._driver

    def _run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query."""
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity in Neo4j."""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.properties = $properties,
            e.created_at = $created_at
        RETURN e.id as id
        """
        result = self._run_query(
            query,
            {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": json.dumps(entity.properties),
                "created_at": entity.created_at.isoformat(),
            },
        )
        return result[0]["id"] if result else entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add or update a relationship in Neo4j."""
        # Sanitize relationship type for Cypher
        rel_type = (
            relationship.relation_type.replace("-", "_").replace(" ", "_").upper()
        )
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.id = $id,
            r.properties = $properties,
            r.confidence = $confidence,
            r.created_at = $created_at
        RETURN r.id as id
        """
        result = self._run_query(
            query,
            {
                "id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": json.dumps(relationship.properties),
                "confidence": relationship.confidence,
                "created_at": relationship.created_at.isoformat(),
            },
        )
        return result[0]["id"] if result else relationship.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.properties as properties, e.created_at as created_at
        """
        result = self._run_query(query, {"id": entity_id})
        if not result:
            return None
        r = result[0]
        return Entity(
            id=r["id"],
            name=r["name"],
            entity_type=r["entity_type"],
            properties=json.loads(r["properties"]) if r["properties"] else {},
            created_at=(
                datetime.fromisoformat(r["created_at"])
                if r["created_at"]
                else datetime.now(timezone.utc)
            ),
        )

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Find entities by name or type."""
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if name:
            conditions.append("e.name CONTAINS $name")
            params["name"] = name
        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
        MATCH (e:Entity)
        {where_clause}
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.properties as properties, e.created_at as created_at
        LIMIT $limit
        """

        results = self._run_query(query, params)
        return [
            Entity(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                properties=json.loads(r["properties"]) if r["properties"] else {},
            )
            for r in results
        ]

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        if direction == "outgoing":
            pattern = "(e)-[r]->(other)"
        elif direction == "incoming":
            pattern = "(e)<-[r]-(other)"
        else:
            pattern = "(e)-[r]-(other)"

        type_filter = f":{relation_type}" if relation_type else ""

        query = f"""
        MATCH (e:Entity {{id: $entity_id}}){pattern.replace("[r]", f"[r{type_filter}]")}
        RETURN r.id as id, e.id as source_id, other.id as target_id,
               type(r) as relation_type, r.properties as properties,
               r.confidence as confidence, r.created_at as created_at
        """

        results = self._run_query(query, {"entity_id": entity_id})
        return [
            Relationship(
                id=r["id"] or f"{r['source_id']}-{r['relation_type']}-{r['target_id']}",
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                properties=json.loads(r["properties"]) if r["properties"] else {},
                confidence=r["confidence"] or 1.0,
            )
            for r in results
        ]

    def traverse(
        self,
        start_entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> GraphResult:
        """Traverse the graph from a starting entity."""
        start_time = time.time()

        type_filter = ""
        if relation_types:
            type_filter = ":" + "|".join(relation_types)

        query = f"""
        MATCH path = (start:Entity {{id: $start_id}})-[r{type_filter}*1..{max_hops}]-(end:Entity)
        RETURN nodes(path) as nodes, relationships(path) as rels
        LIMIT 100
        """

        results = self._run_query(query, {"start_id": start_entity_id})

        entities = {}
        relationships = {}
        paths = []

        for r in results:
            path_ids = []
            for node in r["nodes"]:
                if node["id"] not in entities:
                    entities[node["id"]] = Entity(
                        id=node["id"],
                        name=node.get("name", ""),
                        entity_type=node.get("entity_type", "unknown"),
                    )
                path_ids.append(node["id"])
            paths.append(path_ids)

            for rel in r["rels"]:
                rel_id = rel.get("id", f"{rel['source_id']}-{rel['target_id']}")
                if rel_id not in relationships:
                    relationships[rel_id] = Relationship(
                        id=rel_id,
                        source_id=rel.get("source_id", ""),
                        target_id=rel.get("target_id", ""),
                        relation_type=rel.get("type", "RELATED"),
                    )

        query_time_ms = int((time.time() - start_time) * 1000)

        return GraphResult(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
            paths=paths,
            query_time_ms=query_time_ms,
        )

    def query(self, query: GraphQuery) -> GraphResult:
        """Execute a graph query."""
        # Find starting entities
        all_entities = {}
        all_relationships = {}
        all_paths = []

        for entity_name in query.entities:
            entities = self.find_entities(name=entity_name, limit=5)
            for entity in entities:
                all_entities[entity.id] = entity
                result = self.traverse(
                    entity.id,
                    max_hops=query.max_hops,
                    relation_types=query.relation_types,
                )
                for e in result.entities:
                    all_entities[e.id] = e
                for r in result.relationships:
                    all_relationships[r.id] = r
                all_paths.extend(result.paths)

        return GraphResult(
            entities=list(all_entities.values())[: query.limit],
            relationships=list(all_relationships.values()),
            paths=all_paths[: query.limit],
            query_time_ms=0,
        )

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        query = """
        MATCH (e:Entity {id: $id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        result = self._run_query(query, {"id": entity_id})
        return result[0]["deleted"] > 0 if result else False

    def close(self):
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


class InMemoryGraphStore(GraphStore):
    """
    In-memory graph storage for testing and development.

    No external dependencies required.
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._outgoing: Dict[str, List[str]] = {}  # entity_id -> [rel_ids]
        self._incoming: Dict[str, List[str]] = {}  # entity_id -> [rel_ids]

    def add_entity(self, entity: Entity) -> str:
        self._entities[entity.id] = entity
        if entity.id not in self._outgoing:
            self._outgoing[entity.id] = []
        if entity.id not in self._incoming:
            self._incoming[entity.id] = []
        return entity.id

    def add_relationship(self, relationship: Relationship) -> str:
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
        return self._entities.get(entity_id)

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
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

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        rel_ids = set()

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

    def traverse(
        self,
        start_entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> GraphResult:
        start_time = time.time()

        visited_entities = {start_entity_id}
        visited_relationships = set()
        paths = []

        def _traverse(current_id: str, depth: int, current_path: List[str]):
            if depth > max_hops:
                return

            for rel in self.get_relationships(current_id, "both"):
                if relation_types and rel.relation_type not in relation_types:
                    continue

                visited_relationships.add(rel.id)

                next_id = (
                    rel.target_id if rel.source_id == current_id else rel.source_id
                )

                if next_id not in visited_entities:
                    visited_entities.add(next_id)
                    new_path = current_path + [next_id]
                    paths.append(new_path)
                    _traverse(next_id, depth + 1, new_path)

        _traverse(start_entity_id, 0, [start_entity_id])

        entities = [
            self._entities[eid] for eid in visited_entities if eid in self._entities
        ]
        relationships = [
            self._relationships[rid]
            for rid in visited_relationships
            if rid in self._relationships
        ]

        query_time_ms = int((time.time() - start_time) * 1000)

        return GraphResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            query_time_ms=query_time_ms,
        )

    def query(self, query: GraphQuery) -> GraphResult:
        all_entities = {}
        all_relationships = {}
        all_paths = []

        for entity_name in query.entities:
            entities = self.find_entities(name=entity_name, limit=5)
            for entity in entities:
                all_entities[entity.id] = entity
                result = self.traverse(
                    entity.id,
                    max_hops=query.max_hops,
                    relation_types=query.relation_types,
                )
                for e in result.entities:
                    all_entities[e.id] = e
                for r in result.relationships:
                    all_relationships[r.id] = r
                all_paths.extend(result.paths)

        return GraphResult(
            entities=list(all_entities.values())[: query.limit],
            relationships=list(all_relationships.values()),
            paths=all_paths[: query.limit],
            query_time_ms=0,
        )

    def delete_entity(self, entity_id: str) -> bool:
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

    def clear(self) -> None:
        """Clear all stored data."""
        self._entities.clear()
        self._relationships.clear()
        self._outgoing.clear()
        self._incoming.clear()


class BackendGraphStore(GraphStore):
    """
    GraphStore implementation that delegates to a GraphBackend.

    This class bridges the high-level GraphStore API with the pluggable
    GraphBackend interface, enabling use of different database backends
    while maintaining the full GraphStore functionality.

    Example:
        from alma.graph import create_graph_backend, BackendGraphStore

        backend = create_graph_backend("neo4j", uri="bolt://localhost:7687", ...)
        store = BackendGraphStore(backend)
        store.add_entity(entity)
        result = store.traverse(entity.id)
    """

    def __init__(self, backend: "GraphBackend"):
        """
        Initialize with a GraphBackend.

        Args:
            backend: The GraphBackend implementation to use for storage.
        """
        self._backend = backend

    @property
    def backend(self) -> "GraphBackend":
        """Access the underlying backend."""
        return self._backend

    def add_entity(self, entity: Entity) -> str:
        return self._backend.add_entity(entity)

    def add_relationship(self, relationship: Relationship) -> str:
        return self._backend.add_relationship(relationship)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._backend.get_entity(entity_id)

    def find_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Find entities by name or type using the backend."""
        if hasattr(self._backend, "find_entities"):
            return self._backend.find_entities(
                name=name, entity_type=entity_type, limit=limit
            )
        # Fallback to search_entities if find_entities not available
        if name:
            return self._backend.search_entities(query=name, top_k=limit)
        return self._backend.get_entities(entity_type=entity_type, limit=limit)

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None,
    ) -> List[Relationship]:
        """Get relationships for an entity."""
        if hasattr(self._backend, "get_relationships_directional"):
            return self._backend.get_relationships_directional(
                entity_id=entity_id, direction=direction, relation_type=relation_type
            )
        # Fallback - get all relationships and filter
        all_rels = self._backend.get_relationships(entity_id)
        if relation_type:
            all_rels = [r for r in all_rels if r.relation_type == relation_type]
        return all_rels

    def traverse(
        self,
        start_entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> GraphResult:
        """Traverse the graph from a starting entity."""
        start_time = time.time()

        visited_entities = {start_entity_id}
        visited_relationships: Dict[str, Relationship] = {}
        paths: List[List[str]] = []

        def _traverse(current_id: str, depth: int, current_path: List[str]):
            if depth > max_hops:
                return

            rels = self._backend.get_relationships(current_id)
            for rel in rels:
                if relation_types and rel.relation_type not in relation_types:
                    continue

                visited_relationships[rel.id] = rel

                next_id = (
                    rel.target_id if rel.source_id == current_id else rel.source_id
                )

                if next_id not in visited_entities:
                    visited_entities.add(next_id)
                    new_path = current_path + [next_id]
                    paths.append(new_path)
                    _traverse(next_id, depth + 1, new_path)

        _traverse(start_entity_id, 0, [start_entity_id])

        # Fetch all visited entities
        entities = []
        for eid in visited_entities:
            entity = self._backend.get_entity(eid)
            if entity:
                entities.append(entity)

        query_time_ms = int((time.time() - start_time) * 1000)

        return GraphResult(
            entities=entities,
            relationships=list(visited_relationships.values()),
            paths=paths,
            query_time_ms=query_time_ms,
        )

    def query(self, query: GraphQuery) -> GraphResult:
        """Execute a graph query."""
        all_entities: Dict[str, Entity] = {}
        all_relationships: Dict[str, Relationship] = {}
        all_paths: List[List[str]] = []

        for entity_name in query.entities:
            entities = self.find_entities(name=entity_name, limit=5)
            for entity in entities:
                all_entities[entity.id] = entity
                result = self.traverse(
                    entity.id,
                    max_hops=query.max_hops,
                    relation_types=query.relation_types,
                )
                for e in result.entities:
                    all_entities[e.id] = e
                for r in result.relationships:
                    all_relationships[r.id] = r
                all_paths.extend(result.paths)

        return GraphResult(
            entities=list(all_entities.values())[: query.limit],
            relationships=list(all_relationships.values()),
            paths=all_paths[: query.limit],
            query_time_ms=0,
        )

    def delete_entity(self, entity_id: str) -> bool:
        return self._backend.delete_entity(entity_id)

    def close(self) -> None:
        """Close the backend connection."""
        self._backend.close()


def create_graph_store(
    provider: str = "memory",
    **kwargs,
) -> GraphStore:
    """
    Factory function to create a graph store.

    Args:
        provider: "neo4j", "memory", or "backend"
        **kwargs: Provider-specific arguments
            For "neo4j": uri, username, password, database
            For "backend": backend (GraphBackend instance)

    Returns:
        Configured GraphStore instance

    Note:
        Amazon Neptune support is planned for a future release.

    Example:
        # Create in-memory store
        store = create_graph_store("memory")

        # Create Neo4j store
        store = create_graph_store(
            "neo4j",
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )

        # Create store using a backend
        from alma.graph import create_graph_backend
        backend = create_graph_backend("neo4j", uri="...", ...)
        store = create_graph_store("backend", backend=backend)
    """
    if provider == "neo4j":
        return Neo4jGraphStore(**kwargs)
    elif provider == "neptune":
        # Neptune support is planned for a future release
        raise NotImplementedError(
            "Neptune support is not yet implemented. "
            "Use 'neo4j' or 'memory' providers instead."
        )
    elif provider == "backend":
        backend = kwargs.get("backend")
        if backend is None:
            raise ValueError("'backend' argument required for 'backend' provider")
        return BackendGraphStore(backend)
    else:
        return InMemoryGraphStore()
