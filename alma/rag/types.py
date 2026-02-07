"""
ALMA RAG Types.

Data structures for the RAG bridge integration layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGChunk:
    """Input from an external RAG system.

    Represents a single chunk/passage retrieved by any RAG pipeline
    (LangChain, LlamaIndex, custom). ALMA enhances these with memory
    intelligence without modifying the original data.

    Attributes:
        id: Unique chunk identifier.
        text: The chunk content.
        score: Relevance score from the RAG system (0-1 or raw).
        source: Source document path/URL.
        metadata: Arbitrary metadata from the RAG system.
        embedding: Pre-computed embedding (optional, avoids recomputation).
    """

    id: str
    text: str
    score: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class MemorySignals:
    """Memory-based signals computed by ALMA for a chunk.

    These signals represent what ALMA's memory system knows about
    the topics covered by a chunk, based on past learning outcomes.

    Attributes:
        related_heuristics: Heuristic IDs relevant to this chunk's content.
        related_outcomes: Outcome IDs relevant to this chunk's content.
        trust_score: Trust level based on historical accuracy (0-1).
        historical_success_rate: Success rate of strategies related to this content.
        confidence: ALMA's confidence in the memory signals.
        anti_pattern_warnings: Anti-patterns related to this chunk's content.
        boost_factor: Multiplier to apply to the chunk's score (>1 = boost, <1 = demote).
    """

    related_heuristics: List[str] = field(default_factory=list)
    related_outcomes: List[str] = field(default_factory=list)
    trust_score: float = 0.5
    historical_success_rate: float = 0.5
    confidence: float = 0.0
    anti_pattern_warnings: List[str] = field(default_factory=list)
    boost_factor: float = 1.0


@dataclass
class EnhancedChunk:
    """A RAG chunk enhanced with ALMA memory signals.

    Wraps the original RAGChunk with additional intelligence from
    ALMA's memory system.

    Attributes:
        chunk: The original RAG chunk.
        signals: Memory-based signals for this chunk.
        enhanced_score: Final score after ALMA enhancement.
        rank: Position in the enhanced ranking (1-based).
    """

    chunk: RAGChunk
    signals: MemorySignals
    enhanced_score: float = 0.0
    rank: int = 0


@dataclass
class RAGContext:
    """Complete enhanced context returned by RAGBridge.

    Contains the enhanced chunks plus memory augmentation text
    (strategies, anti-patterns, preferences) that can be injected
    into the LLM prompt alongside the RAG chunks.

    Attributes:
        enhanced_chunks: RAG chunks enhanced with ALMA signals, sorted by enhanced_score.
        memory_augmentation: Generated text with relevant strategies and anti-patterns.
        query: The original query.
        agent: The agent that requested enhancement.
        total_chunks: Number of input chunks.
        token_budget_used: Estimated tokens used by augmentation text.
        metadata: Additional context metadata.
    """

    enhanced_chunks: List[EnhancedChunk] = field(default_factory=list)
    memory_augmentation: str = ""
    query: str = ""
    agent: str = ""
    total_chunks: int = 0
    token_budget_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
