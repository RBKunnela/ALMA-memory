"""
ALMA Graph Memory Module.

Graph-based memory for capturing relationships between entities.
"""

from alma.graph.extraction import (
    EntityExtractor,
    ExtractionConfig,
)
from alma.graph.store import (
    Entity,
    GraphQuery,
    GraphResult,
    GraphStore,
    InMemoryGraphStore,
    Neo4jGraphStore,
    Relationship,
    create_graph_store,
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
