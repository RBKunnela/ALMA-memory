"""
ALMA Graph Memory Backends.

This package contains implementations of the GraphBackend interface
for various graph database systems.

Available backends:
- neo4j: Neo4j graph database backend
- memgraph: Memgraph graph database backend (Neo4j Bolt protocol compatible)
- kuzu: Kuzu embedded graph database backend (no server required)
- memory: In-memory backend for testing and development
"""

from alma.graph.backends.memory import InMemoryBackend
from alma.graph.backends.memgraph import MemgraphBackend
from alma.graph.backends.neo4j import Neo4jBackend

# Kuzu is an optional dependency
try:
    from alma.graph.backends.kuzu import KuzuBackend

    _KUZU_AVAILABLE = True
except ImportError:
    KuzuBackend = None  # type: ignore
    _KUZU_AVAILABLE = False

__all__ = [
    "Neo4jBackend",
    "MemgraphBackend",
    "KuzuBackend",
    "InMemoryBackend",
]
