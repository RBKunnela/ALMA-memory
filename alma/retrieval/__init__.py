"""
ALMA Retrieval Engine.

Provides semantic search, scoring, and caching for memory retrieval.
"""

from alma.retrieval.engine import RetrievalEngine
from alma.retrieval.scoring import (
    MemoryScorer,
    ScoringWeights,
    ScoredItem,
    compute_composite_score,
)
from alma.retrieval.cache import (
    RetrievalCache,
    NullCache,
    CacheEntry,
    CacheStats,
)
from alma.retrieval.embeddings import (
    EmbeddingProvider,
    LocalEmbedder,
    AzureEmbedder,
    MockEmbedder,
)

__all__ = [
    # Engine
    "RetrievalEngine",
    # Scoring
    "MemoryScorer",
    "ScoringWeights",
    "ScoredItem",
    "compute_composite_score",
    # Cache
    "RetrievalCache",
    "NullCache",
    "CacheEntry",
    "CacheStats",
    # Embeddings
    "EmbeddingProvider",
    "LocalEmbedder",
    "AzureEmbedder",
    "MockEmbedder",
]
