"""
ALMA Graph Memory - Neo4j Backend.

Neo4j implementation of the GraphBackend interface.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alma.graph.base import GraphBackend
from alma.graph.store import Entity, Relationship

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphBackend):
    """
    Neo4j graph database backend.

    Requires neo4j Python driver: pip install neo4j

    Example usage:
        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        backend.add_entity(entity)
        backend.close()
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
            database: Database name (default: "neo4j")
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
                    "neo4j package required for Neo4j graph backend. "
                    "Install with: pip install neo4j"
                ) from err
        return self._driver

    def _run_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query."""
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity in Neo4j."""
        # Extract project_id and agent from properties if present
        properties = entity.properties.copy()
        project_id = properties.pop("project_id", None)
        agent = properties.pop("agent", None)

        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.properties = $properties,
            e.created_at = $created_at
        """
        params = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "properties": json.dumps(properties),
            "created_at": entity.created_at.isoformat(),
        }

        # Add optional fields if present
        if project_id:
            query += ", e.project_id = $project_id"
            params["project_id"] = project_id
        if agent:
            query += ", e.agent = $agent"
            params["agent"] = agent

        query += " RETURN e.id as id"

        result = self._run_query(query, params)
        return result[0]["id"] if result else entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add or update a relationship in Neo4j."""
        # Sanitize relationship type for Cypher (remove special characters)
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
               e.properties as properties, e.created_at as created_at,
               e.project_id as project_id, e.agent as agent
        """
        result = self._run_query(query, {"id": entity_id})
        if not result:
            return None

        r = result[0]
        properties = json.loads(r["properties"]) if r["properties"] else {}

        # Add project_id and agent back to properties if present
        if r.get("project_id"):
            properties["project_id"] = r["project_id"]
        if r.get("agent"):
            properties["agent"] = r["agent"]

        return Entity(
            id=r["id"],
            name=r["name"],
            entity_type=r["entity_type"],
            properties=properties,
            created_at=(
                datetime.fromisoformat(r["created_at"])
                if r["created_at"]
                else datetime.now(timezone.utc)
            ),
        )

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        project_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 100,
    ) -> List[Entity]:
        """Get entities with optional filtering."""
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if project_id:
            conditions.append("e.project_id = $project_id")
            params["project_id"] = project_id
        if agent:
            conditions.append("e.agent = $agent")
            params["agent"] = agent

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
        MATCH (e:Entity)
        {where_clause}
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.properties as properties, e.created_at as created_at,
               e.project_id as project_id, e.agent as agent
        LIMIT $limit
        """

        results = self._run_query(query, params)
        entities = []
        for r in results:
            properties = json.loads(r["properties"]) if r["properties"] else {}
            if r.get("project_id"):
                properties["project_id"] = r["project_id"]
            if r.get("agent"):
                properties["agent"] = r["agent"]

            entities.append(
                Entity(
                    id=r["id"],
                    name=r["name"],
                    entity_type=r["entity_type"],
                    properties=properties,
                    created_at=(
                        datetime.fromisoformat(r["created_at"])
                        if r["created_at"]
                        else datetime.now(timezone.utc)
                    ),
                )
            )
        return entities

    def get_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity (both directions)."""
        query = """
        MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
        RETURN r.id as id,
               CASE WHEN startNode(r).id = $entity_id THEN e.id ELSE other.id END as source_id,
               CASE WHEN endNode(r).id = $entity_id THEN e.id ELSE other.id END as target_id,
               type(r) as relation_type, r.properties as properties,
               r.confidence as confidence, r.created_at as created_at
        """

        results = self._run_query(query, {"entity_id": entity_id})
        relationships = []
        for r in results:
            rel_id = (
                r["id"] or f"{r['source_id']}-{r['relation_type']}-{r['target_id']}"
            )
            relationships.append(
                Relationship(
                    id=rel_id,
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=r["relation_type"],
                    properties=json.loads(r["properties"]) if r["properties"] else {},
                    confidence=r["confidence"] or 1.0,
                    created_at=(
                        datetime.fromisoformat(r["created_at"])
                        if r["created_at"]
                        else datetime.now(timezone.utc)
                    ),
                )
            )
        return relationships

    def search_entities(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Entity]:
        """
        Search for entities by name.

        Note: Vector similarity search requires Neo4j 5.x with vector index.
        Falls back to text search if embedding is provided but vector index
        is not available.
        """
        # For now, we do text-based search
        # Vector search can be added when Neo4j vector indexes are set up
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        RETURN e.id as id, e.name as name, e.entity_type as entity_type,
               e.properties as properties, e.created_at as created_at,
               e.project_id as project_id, e.agent as agent
        LIMIT $limit
        """

        results = self._run_query(cypher, {"query": query, "limit": top_k})
        entities = []
        for r in results:
            properties = json.loads(r["properties"]) if r["properties"] else {}
            if r.get("project_id"):
                properties["project_id"] = r["project_id"]
            if r.get("agent"):
                properties["agent"] = r["agent"]

            entities.append(
                Entity(
                    id=r["id"],
                    name=r["name"],
                    entity_type=r["entity_type"],
                    properties=properties,
                    created_at=(
                        datetime.fromisoformat(r["created_at"])
                        if r["created_at"]
                        else datetime.now(timezone.utc)
                    ),
                )
            )
        return entities

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relationships."""
        query = """
        MATCH (e:Entity {id: $id})
        DETACH DELETE e
        RETURN count(e) as deleted
        """
        result = self._run_query(query, {"id": entity_id})
        return result[0]["deleted"] > 0 if result else False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a specific relationship by ID."""
        query = """
        MATCH ()-[r]-()
        WHERE r.id = $id
        DELETE r
        RETURN count(r) as deleted
        """
        result = self._run_query(query, {"id": relationship_id})
        return result[0]["deleted"] > 0 if result else False

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

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
        if name:
            return self.search_entities(query=name, top_k=limit)

        return self.get_entities(entity_type=entity_type, limit=limit)

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
        relationships = []
        for r in results:
            rel_id = (
                r["id"] or f"{r['source_id']}-{r['relation_type']}-{r['target_id']}"
            )
            relationships.append(
                Relationship(
                    id=rel_id,
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=r["relation_type"],
                    properties=json.loads(r["properties"]) if r["properties"] else {},
                    confidence=r["confidence"] or 1.0,
                    created_at=(
                        datetime.fromisoformat(r["created_at"])
                        if r["created_at"]
                        else datetime.now(timezone.utc)
                    ),
                )
            )
        return relationships
