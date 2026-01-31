"""
ALMA Consolidation Module.

Provides memory consolidation capabilities for deduplicating and merging
similar memories, inspired by Mem0's core innovation.
"""

from alma.consolidation.engine import ConsolidationEngine, ConsolidationResult
from alma.consolidation.prompts import (
    MERGE_ANTI_PATTERNS_PROMPT,
    MERGE_DOMAIN_KNOWLEDGE_PROMPT,
    MERGE_HEURISTICS_PROMPT,
    MERGE_OUTCOMES_PROMPT,
)

__all__ = [
    "ConsolidationEngine",
    "ConsolidationResult",
    "MERGE_HEURISTICS_PROMPT",
    "MERGE_DOMAIN_KNOWLEDGE_PROMPT",
    "MERGE_ANTI_PATTERNS_PROMPT",
    "MERGE_OUTCOMES_PROMPT",
]
