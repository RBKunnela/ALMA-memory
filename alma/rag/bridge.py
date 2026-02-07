"""
ALMA RAG Bridge.

Main integration class that connects external RAG systems to ALMA's
memory intelligence. Accepts chunks from any RAG pipeline and returns
enhanced results with memory signals.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from alma.rag.enhancer import MemoryEnhancer
from alma.rag.types import RAGChunk, RAGContext

if TYPE_CHECKING:
    from alma.core import ALMA

logger = logging.getLogger(__name__)


class RAGBridge:
    """Bridge between external RAG systems and ALMA memory intelligence.

    Usage:
        bridge = RAGBridge(alma=alma_instance)
        result = bridge.enhance(
            chunks=[RAGChunk(id="1", text="...", score=0.85, source="docs/deploy.md")],
            query="how to deploy auth service",
            agent="backend-agent",
        )
        # result.enhanced_chunks -- reranked by ALMA
        # result.memory_augmentation -- relevant strategies/anti-patterns text
    """

    def __init__(
        self,
        alma: "ALMA",
        enhancer: Optional[MemoryEnhancer] = None,
        augmentation_token_budget: int = 500,
    ) -> None:
        """Initialize the RAG bridge.

        Args:
            alma: ALMA instance for memory retrieval and learning.
            enhancer: Optional custom MemoryEnhancer. Defaults to standard enhancer.
            augmentation_token_budget: Max tokens for memory augmentation text.
        """
        self.alma = alma
        self.enhancer = enhancer or MemoryEnhancer()
        self.augmentation_token_budget = augmentation_token_budget

    def enhance(
        self,
        chunks: List[RAGChunk],
        query: str,
        agent: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> RAGContext:
        """Enhance RAG chunks with ALMA memory intelligence.

        Steps:
        1. Retrieve relevant ALMA memories for the query.
        2. Compute memory signals per chunk.
        3. Score and rerank chunks using memory signals.
        4. Generate memory augmentation text.

        Args:
            chunks: Chunks from external RAG system.
            query: The original query.
            agent: Agent requesting enhancement.
            user_id: Optional user ID for preference retrieval.
            top_k: Max memories to retrieve per type from ALMA.

        Returns:
            RAGContext with enhanced chunks and memory augmentation.
        """
        if not chunks:
            return RAGContext(
                query=query,
                agent=agent,
                total_chunks=0,
            )

        # Step 1: Retrieve relevant ALMA memories
        memory_slice = self.alma.retrieve(
            task=query,
            agent=agent,
            user_id=user_id,
            top_k=top_k,
        )

        # Step 2+3: Enhance chunks with memory signals
        enhanced_chunks = self.enhancer.enhance_chunks(chunks, memory_slice)

        # Step 4: Generate augmentation text
        augmentation = self.enhancer.generate_augmentation(
            memory_slice,
            max_tokens=self.augmentation_token_budget,
        )

        # Estimate token usage (~4 chars per token)
        token_budget_used = len(augmentation) // 4

        return RAGContext(
            enhanced_chunks=enhanced_chunks,
            memory_augmentation=augmentation,
            query=query,
            agent=agent,
            total_chunks=len(chunks),
            token_budget_used=token_budget_used,
            metadata={
                "memories_retrieved": memory_slice.total_items,
                "retrieval_time_ms": memory_slice.retrieval_time_ms,
            },
        )
