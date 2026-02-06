"""
ALMA Graph Memory - Kuzu Backend.

Kuzu embedded graph database implementation of the GraphBackend interface.
Kuzu is an embedded graph database similar to SQLite but for graph data.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from alma.graph.base import GraphBackend
from alma.graph.store import Entity, Relationship

logger = logging.getLogger(__name__)


class KuzuBackend(GraphBackend):
    """
    Kuzu embedded graph database backend.

    Kuzu is an embeddable property graph database management system.
    It supports Cypher-compatible query language and requires no server.

    Requires kuzu Python package: pip install kuzu

    Example usage:
        # Persistent mode (data saved to disk)
        backend = KuzuBackend(database_path="./my_graph_db")
        backend.add_entity(entity)
        backend.close()

        # In-memory mode (data lost when closed)
        backend = KuzuBackend()  # No path = in-memory
        backend.add_entity(entity)
        backend.close()
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        read_only: bool = False,
    ):
        """
        Initialize Kuzu database connection.

        Args:
            database_path: Path to the database directory. If None, creates
                          a temporary in-memory database.
            read_only: If True, open the database in read-only mode.
        """
        self.database_path = database_path
        self.read_only = read_only
        self._db = None
        self._conn = None
        self._schema_initialized = False

    def _get_connection(self):
        """Lazy initialization of Kuzu database and connection."""
        if self._conn is None:
            try:
                import kuzu
            except ImportError as err:
                raise ImportError(
                    "kuzu package required for Kuzu graph backend. "
                    "Install with: pip install kuzu"
                ) from err

            # Determine database path
            if self.database_path is None:
                # In-memory mode: use `:memory:` for true in-memory database
                db_path = ":memory:"
            else:
                db_path = self.database_path
                # For persistent mode, ensure parent directory exists
                # but not the database directory itself (Kuzu will create it)
                parent_dir = os.path.dirname(db_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            self._db = kuzu.Database(db_path, read_only=self.read_only)
            self._conn = kuzu.Connection(self._db)
            self._initialize_schema()

        return self._conn

    def _initialize_schema(self) -> None:
        """Initialize the graph schema if not already done."""
        if self._schema_initialized:
            return

        conn = self._conn

        # Check if Entity table exists
        try:
            conn.execute("MATCH (e:Entity) RETURN e LIMIT 1")
            self._schema_initialized = True
            return
        except Exception:
            # Table doesn't exist, create schema
            pass

        # Create Entity node table
        conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING PRIMARY KEY,
                name STRING,
                entity_type STRING,
                properties STRING,
                project_id STRING,
                agent STRING,
                created_at STRING
            )
        """)

        # Create Relationship edge table
        # In Kuzu, we need a generic edge table since relationship types are dynamic
        conn.execute("""
            CREATE REL TABLE IF NOT EXISTS RELATES_TO(
                FROM Entity TO Entity,
                id STRING,
                relation_type STRING,
                properties STRING,
                confidence DOUBLE,
                created_at STRING
            )
        """)

        self._schema_initialized = True

    def _run_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Execute a Cypher query and return results as list of dicts."""
        conn = self._get_connection()
        try:
            if parameters:
                result = conn.execute(query, parameters)
            else:
                result = conn.execute(query)

            # Convert result to list of dicts
            rows = []
            while result.has_next():
                row = result.get_next()
                # Get column names
                col_names = result.get_column_names()
                row_dict = {}
                for i, col_name in enumerate(col_names):
                    row_dict[col_name] = row[i]
                rows.append(row_dict)
            return rows
        except Exception as e:
            logger.debug(f"Query error: {e}, Query: {query}, Params: {parameters}")
            raise

    def add_entity(self, entity: Entity) -> str:
        """Add or update an entity in Kuzu."""
        # Extract project_id and agent from properties if present
        properties = entity.properties.copy()
        project_id = properties.pop("project_id", None) or ""
        agent = properties.pop("agent", None) or ""

        # Check if entity exists
        existing = self.get_entity(entity.id)

        if existing:
            # Update existing entity
            query = """
                MATCH (e:Entity {id: $id})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.properties = $properties,
                    e.project_id = $project_id,
                    e.agent = $agent,
                    e.created_at = $created_at
            """
        else:
            # Create new entity
            query = """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    entity_type: $entity_type,
                    properties: $properties,
                    project_id: $project_id,
                    agent: $agent,
                    created_at: $created_at
                })
            """

        params = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "properties": json.dumps(properties),
            "project_id": project_id,
            "agent": agent,
            "created_at": entity.created_at.isoformat(),
        }

        self._run_query(query, params)
        return entity.id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add or update a relationship in Kuzu."""
        # Check if relationship exists
        check_query = """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            WHERE r.id = $id
            RETURN r.id
        """
        existing = self._run_query(check_query, {"id": relationship.id})

        if existing:
            # Update existing relationship
            query = """
                MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
                WHERE r.id = $id
                SET r.relation_type = $relation_type,
                    r.properties = $properties,
                    r.confidence = $confidence,
                    r.created_at = $created_at
            """
            params = {
                "id": relationship.id,
                "relation_type": relationship.relation_type,
                "properties": json.dumps(relationship.properties),
                "confidence": relationship.confidence,
                "created_at": relationship.created_at.isoformat(),
            }
        else:
            # Create new relationship
            query = """
                MATCH (s:Entity {id: $source_id}), (t:Entity {id: $target_id})
                CREATE (s)-[r:RELATES_TO {
                    id: $id,
                    relation_type: $relation_type,
                    properties: $properties,
                    confidence: $confidence,
                    created_at: $created_at
                }]->(t)
            """
            params = {
                "id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "relation_type": relationship.relation_type,
                "properties": json.dumps(relationship.properties),
                "confidence": relationship.confidence,
                "created_at": relationship.created_at.isoformat(),
            }

        self._run_query(query, params)
        return relationship.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        query = """
            MATCH (e:Entity {id: $id})
            RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                   e.properties AS properties, e.created_at AS created_at,
                   e.project_id AS project_id, e.agent AS agent
        """
        results = self._run_query(query, {"id": entity_id})

        if not results:
            return None

        r = results[0]
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
            RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                   e.properties AS properties, e.created_at AS created_at,
                   e.project_id AS project_id, e.agent AS agent
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
        # Get outgoing relationships
        outgoing_query = """
            MATCH (s:Entity {id: $entity_id})-[r:RELATES_TO]->(t:Entity)
            RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                   r.relation_type AS relation_type, r.properties AS properties,
                   r.confidence AS confidence, r.created_at AS created_at
        """

        # Get incoming relationships
        incoming_query = """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity {id: $entity_id})
            RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                   r.relation_type AS relation_type, r.properties AS properties,
                   r.confidence AS confidence, r.created_at AS created_at
        """

        params = {"entity_id": entity_id}
        outgoing = self._run_query(outgoing_query, params)
        incoming = self._run_query(incoming_query, params)

        # Deduplicate by relationship ID
        seen_ids: Set[str] = set()
        relationships = []

        for r in outgoing + incoming:
            rel_id = (
                r["id"] or f"{r['source_id']}-{r['relation_type']}-{r['target_id']}"
            )
            if rel_id in seen_ids:
                continue
            seen_ids.add(rel_id)

            relationships.append(
                Relationship(
                    id=rel_id,
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=r["relation_type"] or "RELATES_TO",
                    properties=json.loads(r["properties"]) if r["properties"] else {},
                    confidence=r["confidence"] if r["confidence"] is not None else 1.0,
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

        Note: Vector similarity search is not implemented for Kuzu backend.
        Falls back to case-insensitive text search.
        """
        # Kuzu uses CONTAINS for substring matching
        cypher = """
            MATCH (e:Entity)
            WHERE lower(e.name) CONTAINS lower($query)
            RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
                   e.properties AS properties, e.created_at AS created_at,
                   e.project_id AS project_id, e.agent AS agent
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
        """Delete an entity and all its relationships."""
        # Check if entity exists
        entity = self.get_entity(entity_id)
        if not entity:
            return False

        # Delete relationships first (both directions)
        self._run_query(
            """
            MATCH (s:Entity {id: $id})-[r:RELATES_TO]->()
            DELETE r
            """,
            {"id": entity_id},
        )

        self._run_query(
            """
            MATCH ()-[r:RELATES_TO]->(t:Entity {id: $id})
            DELETE r
            """,
            {"id": entity_id},
        )

        # Delete the entity
        self._run_query(
            """
            MATCH (e:Entity {id: $id})
            DELETE e
            """,
            {"id": entity_id},
        )

        return True

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a specific relationship by ID."""
        # Check if relationship exists
        check_query = """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.id = $id
            RETURN r.id
        """
        existing = self._run_query(check_query, {"id": relationship_id})

        if not existing:
            return False

        # Delete the relationship
        self._run_query(
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.id = $id
            DELETE r
            """,
            {"id": relationship_id},
        )

        return True

    def close(self) -> None:
        """Close the Kuzu database connection and clean up resources."""
        if self._conn is not None:
            self._conn = None

        if self._db is not None:
            self._db = None

        self._schema_initialized = False

    def clear(self) -> None:
        """Clear all data from the database."""
        conn = self._get_connection()

        # Delete all relationships first
        conn.execute("MATCH ()-[r:RELATES_TO]->() DELETE r")

        # Delete all entities
        conn.execute("MATCH (e:Entity) DELETE e")

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
        results = []
        params: Dict[str, Any] = {"entity_id": entity_id}

        type_filter = ""
        if relation_type:
            type_filter = "AND r.relation_type = $relation_type"
            params["relation_type"] = relation_type

        if direction in ("outgoing", "both"):
            outgoing_query = f"""
                MATCH (s:Entity {{id: $entity_id}})-[r:RELATES_TO]->(t:Entity)
                WHERE true {type_filter}
                RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                       r.relation_type AS relation_type, r.properties AS properties,
                       r.confidence AS confidence, r.created_at AS created_at
            """
            results.extend(self._run_query(outgoing_query, params))

        if direction in ("incoming", "both"):
            incoming_query = f"""
                MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity {{id: $entity_id}})
                WHERE true {type_filter}
                RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
                       r.relation_type AS relation_type, r.properties AS properties,
                       r.confidence AS confidence, r.created_at AS created_at
            """
            results.extend(self._run_query(incoming_query, params))

        # Deduplicate by relationship ID
        seen_ids: Set[str] = set()
        relationships = []

        for r in results:
            rel_id = (
                r["id"] or f"{r['source_id']}-{r['relation_type']}-{r['target_id']}"
            )
            if rel_id in seen_ids:
                continue
            seen_ids.add(rel_id)

            relationships.append(
                Relationship(
                    id=rel_id,
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=r["relation_type"] or "RELATES_TO",
                    properties=json.loads(r["properties"]) if r["properties"] else {},
                    confidence=r["confidence"] if r["confidence"] is not None else 1.0,
                    created_at=(
                        datetime.fromisoformat(r["created_at"])
                        if r["created_at"]
                        else datetime.now(timezone.utc)
                    ),
                )
            )

        return relationships
