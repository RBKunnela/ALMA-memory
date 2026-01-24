"""
ALMA Graph Memory Module.

Graph-based memory for capturing relationships between entities.
"""

from alma.graph.store import (
    GraphStore,
    Neo4jGraphStore,
    InMemoryGraphStore,
    Entity,
    Relationship,
    GraphQuery,
    GraphResult,
    create_graph_store,
)
from alma.graph.extraction import (
    EntityExtractor,
    ExtractionConfig,
)

__all__ = [
    # Store classes
    "GraphStore",
    "Neo4jGraphStore",
    "InMemoryGraphStore",
    # Data classes
    "Entity",
    "Relationship", 
    "GraphQuery",
    "GraphResult",
    # Extraction
    "EntityExtractor",
    "ExtractionConfig",
    # Factory
    "create_graph_store",
]
