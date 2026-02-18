"""
ALMA Consolidation Module.

Provides memory consolidation capabilities for deduplicating and merging
similar memories, inspired by Mem0's core innovation.
"""

from alma.consolidation.config import ConsolidationConfig, get_llm_api_key
from alma.consolidation.deduplication import DeduplicationEngine, DeduplicationResult
from alma.consolidation.engine import ConsolidationEngine, ConsolidationResult
from alma.consolidation.exceptions import (
    CacheError,
    ConsolidationError,
    InvalidLLMResponse,
    LLMError,
    StorageError,
    ValidationError,
)
from alma.consolidation.prompts import (
    MERGE_ANTI_PATTERNS_PROMPT,
    MERGE_DOMAIN_KNOWLEDGE_PROMPT,
    MERGE_HEURISTICS_PROMPT,
    MERGE_OUTCOMES_PROMPT,
)
from alma.consolidation.rate_limit import (
    BoundedCache,
    RateLimiter,
    init_rate_limiter,
    rate_limit_llm_call,
)
from alma.consolidation.strategy import (
    ConsolidationStrategy,
    ConsolidationStrategyFactory,
    HeuristicConsolidationStrategy,
    LLMConsolidationStrategy,
)
from alma.consolidation.validation import (
    validate_anti_pattern_response,
    validate_domain_knowledge_response,
    validate_heuristic_response,
    validate_llm_response,
)

__all__ = [
    # Engine
    "ConsolidationEngine",
    "ConsolidationResult",
    # Exceptions
    "ConsolidationError",
    "LLMError",
    "InvalidLLMResponse",
    "CacheError",
    "ValidationError",
    "StorageError",
    # Config
    "ConsolidationConfig",
    "get_llm_api_key",
    # Rate limiting & cache
    "RateLimiter",
    "BoundedCache",
    "init_rate_limiter",
    "rate_limit_llm_call",
    # Strategy
    "ConsolidationStrategy",
    "ConsolidationStrategyFactory",
    "LLMConsolidationStrategy",
    "HeuristicConsolidationStrategy",
    # Deduplication
    "DeduplicationEngine",
    "DeduplicationResult",
    # Validation
    "validate_llm_response",
    "validate_heuristic_response",
    "validate_domain_knowledge_response",
    "validate_anti_pattern_response",
    # Prompts
    "MERGE_HEURISTICS_PROMPT",
    "MERGE_DOMAIN_KNOWLEDGE_PROMPT",
    "MERGE_ANTI_PATTERNS_PROMPT",
    "MERGE_OUTCOMES_PROMPT",
]
