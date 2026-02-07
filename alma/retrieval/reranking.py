"""
ALMA Reranking Abstraction.

Provides a pluggable reranking layer between retrieval and scoring.
Supports cross-encoder reranking (via optional `rerankers` lib)
with a no-op fallback that preserves existing behavior.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        items: List[Any],
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank items by relevance to query.

        Args:
            query: The search query.
            items: Original items (for reference, not modified).
            texts: Text representation of each item for scoring.
            top_k: Maximum results to return (None = all).

        Returns:
            List of (original_index, relevance_score) sorted by score descending.
        """


class NoOpReranker(Reranker):
    """Pass-through reranker that preserves original ordering.

    Returns items in their original order with scores of 1.0.
    This is the default when no reranking library is installed.
    """

    def rerank(
        self,
        query: str,
        items: List[Any],
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        n = len(items)
        if top_k is not None:
            n = min(n, top_k)
        return [(i, 1.0) for i in range(n)]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using the optional `rerankers` library.

    Falls back to NoOpReranker if the library is not installed.

    Args:
        model_name: Model name for the cross-encoder.
            Default: "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good quality).
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.model_name = model_name
        self._reranker = None
        self._fallback: Optional[NoOpReranker] = None

        try:
            import rerankers  # noqa: F401

            self._available = True
        except ImportError:
            logger.info(
                "rerankers not installed, using NoOpReranker fallback. "
                "Install with: pip install rerankers"
            )
            self._available = False
            self._fallback = NoOpReranker()

    def _get_reranker(self):
        if self._reranker is None and self._available:
            from rerankers import Reranker as RerankersReranker

            self._reranker = RerankersReranker(
                self.model_name, model_type="cross-encoder"
            )
        return self._reranker

    def rerank(
        self,
        query: str,
        items: List[Any],
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        if not self._available:
            return self._fallback.rerank(query, items, texts, top_k)

        if not texts:
            return []

        try:
            reranker = self._get_reranker()
            results = reranker.rank(query, texts)

            output = []
            for result in results.results:
                output.append((result.doc_id, result.score))

            output.sort(key=lambda x: -x[1])
            if top_k is not None:
                output = output[:top_k]
            return output

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed, falling back to NoOp: {e}")
            if self._fallback is None:
                self._fallback = NoOpReranker()
            return self._fallback.rerank(query, items, texts, top_k)
