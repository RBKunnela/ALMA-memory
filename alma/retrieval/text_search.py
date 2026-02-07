"""
ALMA Text Search Providers.

Provides keyword-based text search to complement vector search.
Supports BM25 (via optional bm25s library) with a pure-Python TF-IDF fallback.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TextSearchProvider(ABC):
    """Abstract base class for text search providers."""

    @abstractmethod
    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Index documents for search.

        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs (default: 0-based indices).
        """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search indexed documents.

        Args:
            query: Search query text.
            top_k: Maximum number of results.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending.
        """

    @abstractmethod
    def is_indexed(self) -> bool:
        """Return True if documents have been indexed."""


class SimpleTFIDFProvider(TextSearchProvider):
    """
    Pure-Python TF-IDF text search. Zero external dependencies.

    Uses standard TF-IDF with cosine similarity. Not as good as BM25
    for long documents, but perfectly adequate for short memory texts
    (heuristic conditions, strategy descriptions, domain facts).
    """

    def __init__(self) -> None:
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._idf: Dict[str, float] = {}
        self._doc_tfidf: List[Dict[str, float]] = []

    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        self._documents = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]

        # Compute IDF
        n = len(documents)
        if n == 0:
            return

        df: Dict[str, int] = Counter()
        tokenized = [self._tokenize(doc) for doc in documents]
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1

        self._idf = {
            term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()
        }

        # Compute TF-IDF vectors per document
        self._doc_tfidf = []
        for tokens in tokenized:
            tf = Counter(tokens)
            doc_len = len(tokens) or 1
            tfidf = {
                term: (count / doc_len) * self._idf.get(term, 0.0)
                for term, count in tf.items()
            }
            self._doc_tfidf.append(tfidf)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self._doc_tfidf:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build query TF-IDF vector
        tf = Counter(query_tokens)
        q_len = len(query_tokens)
        query_tfidf = {
            term: (count / q_len) * self._idf.get(term, 0.0)
            for term, count in tf.items()
        }

        # Cosine similarity with each document
        scores: List[Tuple[int, float]] = []
        q_norm = math.sqrt(sum(v * v for v in query_tfidf.values()))
        if q_norm == 0:
            return []

        for i, doc_vec in enumerate(self._doc_tfidf):
            dot = sum(
                query_tfidf.get(t, 0.0) * doc_vec.get(t, 0.0) for t in query_tfidf
            )
            d_norm = math.sqrt(sum(v * v for v in doc_vec.values()))
            if d_norm > 0:
                sim = dot / (q_norm * d_norm)
                if sim > 0:
                    scores.append((i, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def is_indexed(self) -> bool:
        return len(self._doc_tfidf) > 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()


class BM25SProvider(TextSearchProvider):
    """
    BM25 text search via the optional `bm25s` library.

    Falls back to SimpleTFIDFProvider if bm25s is not installed.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._doc_count = 0
        self._fallback: Optional[SimpleTFIDFProvider] = None

        try:
            import bm25s  # noqa: F401

            self._bm25s_available = True
        except ImportError:
            logger.info("bm25s not installed, using SimpleTFIDFProvider fallback")
            self._bm25s_available = False
            self._fallback = SimpleTFIDFProvider()

    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        if not self._bm25s_available:
            self._fallback.index(documents, doc_ids)
            return

        import bm25s

        self._doc_count = len(documents)
        if self._doc_count == 0:
            return

        self._bm25 = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(documents)
        self._bm25.index(corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self._bm25s_available:
            return self._fallback.search(query, top_k)

        if self._bm25 is None or self._doc_count == 0:
            return []

        import bm25s

        query_tokens = bm25s.tokenize([query])
        results, scores = self._bm25.retrieve(
            query_tokens, k=min(top_k, self._doc_count)
        )

        output = []
        for idx, score in zip(results[0], scores[0], strict=True):
            if score > 0:
                output.append((int(idx), float(score)))
        return output

    def is_indexed(self) -> bool:
        if not self._bm25s_available:
            return self._fallback.is_indexed()
        return self._bm25 is not None and self._doc_count > 0
