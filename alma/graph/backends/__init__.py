"""
ALMA Graph Memory Backends.

This package contains implementations of the GraphBackend interface
for various graph database systems.

Available backends:
- neo4j: Neo4j graph database backend
- memory: In-memory backend for testing and development
"""

from alma.graph.backends.memory import InMemoryBackend
from alma.graph.backends.neo4j import Neo4jBackend

__all__ = [
    "Neo4jBackend",
    "InMemoryBackend",
]
