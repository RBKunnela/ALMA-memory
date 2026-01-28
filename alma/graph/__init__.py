"""
ALMA Graph Memory Module.

Graph-based memory for capturing relationships between entities.
"""

from alma.graph.base import GraphBackend
from alma.graph.extraction import (
    EntityExtractor,
    ExtractionConfig,
)
from alma.graph.store import (
    BackendGraphStore,
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
    # Backend abstract base class
    "GraphBackend",
    # Store classes (high-level API)
    "GraphStore",
    "Neo4jGraphStore",
    "InMemoryGraphStore",
    "BackendGraphStore",
    # Data classes
    "Entity",
    "Relationship",
    "GraphQuery",
    "GraphResult",
    # Extraction
    "EntityExtractor",
    "ExtractionConfig",
    # Factory functions
    "create_graph_store",
    "create_graph_backend",
]


def create_graph_backend(backend: str = "neo4j", **config) -> GraphBackend:
    """
    Factory function to create a graph backend.

    Args:
        backend: Backend type ("neo4j" or "memory")
        **config: Backend-specific configuration options

    Returns:
        Configured GraphBackend instance

    Raises:
        ValueError: If an unknown backend type is specified

    Example:
        # Create Neo4j backend
        backend = create_graph_backend(
            backend="neo4j",
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )

        # Create in-memory backend for testing
        backend = create_graph_backend(backend="memory")
    """
    if backend == "neo4j":
        from alma.graph.backends.neo4j import Neo4jBackend

        return Neo4jBackend(**config)
    elif backend == "memory":
        from alma.graph.backends.memory import InMemoryBackend

        return InMemoryBackend()
    else:
        raise ValueError(f"Unknown graph backend: {backend}")
