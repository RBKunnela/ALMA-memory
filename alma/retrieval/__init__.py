"""
ALMA Retrieval Engine.

Provides semantic search, scoring, and caching for memory retrieval.
"""

from alma.retrieval.cache import (
    CacheBackend,
    CacheEntry,
    CacheKeyGenerator,
    CacheStats,
    NullCache,
    PerformanceMetrics,
    RedisCache,
    RetrievalCache,
    create_cache,
)
from alma.retrieval.embeddings import (
    AzureEmbedder,
    EmbeddingProvider,
    LocalEmbedder,
    MockEmbedder,
)
from alma.retrieval.engine import RetrievalEngine
from alma.retrieval.scoring import (
    MemoryScorer,
    ScoredItem,
    ScoringWeights,
    compute_composite_score,
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
    "CacheBackend",
    "CacheKeyGenerator",
    "RetrievalCache",
    "RedisCache",
    "NullCache",
    "CacheEntry",
    "CacheStats",
    "PerformanceMetrics",
    "create_cache",
    # Embeddings
    "EmbeddingProvider",
    "LocalEmbedder",
    "AzureEmbedder",
    "MockEmbedder",
]
