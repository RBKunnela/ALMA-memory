"""
ALMA RAG Integration.

Makes any RAG system smarter by learning from retrieval outcomes.
ALMA does NOT become a RAG framework -- it accepts RAG output and
enhances it with memory-based intelligence.

Usage:
    from alma.rag import RAGBridge, RAGChunk

    bridge = RAGBridge(alma=alma_instance)
    result = bridge.enhance(
        chunks=[RAGChunk(id="1", text="...", score=0.85)],
        query="how to deploy auth service",
        agent="backend-agent",
    )
"""

from alma.rag.bridge import RAGBridge
from alma.rag.enhancer import MemoryEnhancer
from alma.rag.feedback import RetrievalFeedbackTracker
from alma.rag.feedback_types import (
    RetrievalEffectiveness,
    RetrievalFeedback,
    RetrievalRecord,
)
from alma.rag.metrics import RetrievalMetrics
from alma.rag.metrics_types import (
    MetricsHistory,
    MetricsResult,
    RelevanceJudgment,
)
from alma.rag.types import (
    EnhancedChunk,
    MemorySignals,
    RAGChunk,
    RAGContext,
)

__all__ = [
    # Bridge
    "RAGBridge",
    "RAGChunk",
    "EnhancedChunk",
    "MemorySignals",
    "RAGContext",
    "MemoryEnhancer",
    # Feedback Loop
    "RetrievalFeedbackTracker",
    "RetrievalRecord",
    "RetrievalFeedback",
    "RetrievalEffectiveness",
    # Metrics
    "RetrievalMetrics",
    "RelevanceJudgment",
    "MetricsResult",
    "MetricsHistory",
]
