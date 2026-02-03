"""
ALMA Retrieval Engine.

Provides semantic search, scoring, and caching for memory retrieval.
Supports mode-aware retrieval for different cognitive tasks.
Includes token budget management, trust-integrated scoring, and progressive disclosure.
"""

from alma.retrieval.budget import (
    BudgetedItem,
    BudgetReport,
    PriorityTier,
    RetrievalBudget,
    TokenEstimator,
)
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
from alma.retrieval.modes import (
    ModeConfig,
    RetrievalMode,
    create_custom_mode,
    get_mode_config,
    get_mode_reason,
    infer_mode_from_query,
    validate_mode_config,
)
from alma.retrieval.progressive import (
    DisclosureLevel,
    MemorySummary,
    ProgressiveRetrieval,
    ProgressiveSlice,
)
from alma.retrieval.scoring import (
    MemoryScorer,
    ScoredItem,
    ScoringWeights,
    compute_composite_score,
)
from alma.retrieval.trust_scoring import (
    AgentTrustContext,
    TrustAwareScorer,
    TrustPatternStore,
    TrustScoredItem,
    TrustScoringWeights,
)
from alma.retrieval.verification import (
    Verification,
    VerificationConfig,
    VerificationMethod,
    VerificationStatus,
    VerifiedMemory,
    VerifiedResults,
    VerifiedRetriever,
    create_verified_retriever,
)

__all__ = [
    # Engine
    "RetrievalEngine",
    # Modes
    "RetrievalMode",
    "ModeConfig",
    "infer_mode_from_query",
    "get_mode_config",
    "get_mode_reason",
    "create_custom_mode",
    "validate_mode_config",
    # Scoring
    "MemoryScorer",
    "ScoringWeights",
    "ScoredItem",
    "compute_composite_score",
    # Trust-Integrated Scoring (v0.8.0+)
    "TrustAwareScorer",
    "TrustScoringWeights",
    "TrustScoredItem",
    "AgentTrustContext",
    "TrustPatternStore",
    # Token Budget (v0.8.0+)
    "PriorityTier",
    "TokenEstimator",
    "BudgetedItem",
    "BudgetReport",
    "RetrievalBudget",
    # Progressive Disclosure (v0.8.0+)
    "DisclosureLevel",
    "MemorySummary",
    "ProgressiveSlice",
    "ProgressiveRetrieval",
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
    # Verification (v0.7.0+)
    "VerificationStatus",
    "VerificationMethod",
    "Verification",
    "VerifiedMemory",
    "VerifiedResults",
    "VerificationConfig",
    "VerifiedRetriever",
    "create_verified_retriever",
]
