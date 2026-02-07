"""
ALMA Hybrid Search Engine.

Combines vector search and keyword search using Reciprocal Rank Fusion (RRF).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from alma.retrieval.text_search import (
    BM25SProvider,
    SimpleTFIDFProvider,
    TextSearchProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    rrf_k: int = 60  # RRF constant (industry standard)
    vector_weight: float = 0.5  # Weight for vector results in final score
    text_weight: float = 0.5  # Weight for text results in final score
    text_provider: str = "auto"  # "auto", "bm25s", or "tfidf"

    def __post_init__(self) -> None:
        total = self.vector_weight + self.text_weight
        if total > 0 and abs(total - 1.0) > 0.01:
            self.vector_weight /= total
            self.text_weight /= total


@dataclass
class HybridResult:
    """A single result from hybrid search with provenance tracking."""

    item: Any
    index: int
    rrf_score: float
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None
    vector_score: float = 0.0
    text_score: float = 0.0


class HybridSearchEngine:
    """
    Fuses vector search results with keyword search results using RRF.

    The engine maintains a text index that must be rebuilt when the corpus
    changes. Vector search results come from the storage backend directly.

    Usage:
        hybrid = HybridSearchEngine()
        # Index corpus texts for keyword search
        texts = [h.strategy for h in heuristics]
        hybrid.index_corpus(texts)
        # Fuse with vector search results
        fused = hybrid.fuse(
            vector_results=[(0, 0.95), (2, 0.82), (1, 0.71)],
            text_results=hybrid.text_search("JWT auth", top_k=10),
            items=heuristics,
        )
    """

    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        text_provider: Optional[TextSearchProvider] = None,
    ) -> None:
        self.config = config or HybridSearchConfig()
        self._text_provider = text_provider or self._create_text_provider()

    def _create_text_provider(self) -> TextSearchProvider:
        if self.config.text_provider == "tfidf":
            return SimpleTFIDFProvider()
        elif self.config.text_provider == "bm25s":
            return BM25SProvider()
        else:
            # "auto" -- try BM25S first, it falls back to TFIDF internally
            return BM25SProvider()

    def index_corpus(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Index document texts for keyword search.

        Args:
            texts: List of document texts (must align with item indices).
            doc_ids: Optional document IDs.
        """
        self._text_provider.index(texts, doc_ids)

    def text_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Run keyword search on indexed corpus.

        Args:
            query: Search query.
            top_k: Maximum results.

        Returns:
            List of (doc_index, score) tuples.
        """
        if not self._text_provider.is_indexed():
            return []
        return self._text_provider.search(query, top_k)

    def fuse(
        self,
        vector_results: List[Tuple[int, float]],
        text_results: List[Tuple[int, float]],
        items: Optional[List[Any]] = None,
    ) -> List[HybridResult]:
        """Fuse vector and text search results using Reciprocal Rank Fusion.

        Args:
            vector_results: List of (index, score) from vector search.
            text_results: List of (index, score) from text search.
            items: Optional list of original items (indexed by position).

        Returns:
            List of HybridResult sorted by fused RRF score descending.
        """
        k = self.config.rrf_k
        vw = self.config.vector_weight
        tw = self.config.text_weight

        # Build RRF scores
        scores: Dict[int, Dict[str, Any]] = {}

        for rank, (idx, score) in enumerate(vector_results):
            if idx not in scores:
                scores[idx] = {
                    "rrf": 0.0,
                    "v_rank": None,
                    "t_rank": None,
                    "v_score": 0.0,
                    "t_score": 0.0,
                }
            scores[idx]["rrf"] += vw * (1.0 / (k + rank + 1))
            scores[idx]["v_rank"] = rank + 1
            scores[idx]["v_score"] = score

        for rank, (idx, score) in enumerate(text_results):
            if idx not in scores:
                scores[idx] = {
                    "rrf": 0.0,
                    "v_rank": None,
                    "t_rank": None,
                    "v_score": 0.0,
                    "t_score": 0.0,
                }
            scores[idx]["rrf"] += tw * (1.0 / (k + rank + 1))
            scores[idx]["t_rank"] = rank + 1
            scores[idx]["t_score"] = score

        # Build results
        results = []
        for idx, data in scores.items():
            results.append(
                HybridResult(
                    item=items[idx] if items and idx < len(items) else None,
                    index=idx,
                    rrf_score=data["rrf"],
                    vector_rank=data["v_rank"],
                    text_rank=data["t_rank"],
                    vector_score=data["v_score"],
                    text_score=data["t_score"],
                )
            )

        results.sort(key=lambda r: -r.rrf_score)
        return results
