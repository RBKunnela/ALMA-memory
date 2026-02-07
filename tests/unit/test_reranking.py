"""Tests for the reranking abstraction."""

from alma.retrieval.reranking import CrossEncoderReranker, NoOpReranker


class TestNoOpReranker:
    """Tests for the pass-through NoOp reranker."""

    def test_preserves_order(self):
        reranker = NoOpReranker()
        items = ["a", "b", "c"]
        texts = ["doc a", "doc b", "doc c"]

        results = reranker.rerank("query", items, texts)

        assert len(results) == 3
        assert results == [(0, 1.0), (1, 1.0), (2, 1.0)]

    def test_respects_top_k(self):
        reranker = NoOpReranker()
        items = ["a", "b", "c", "d", "e"]
        texts = ["doc"] * 5

        results = reranker.rerank("query", items, texts, top_k=2)
        assert len(results) == 2

    def test_empty_input(self):
        reranker = NoOpReranker()
        results = reranker.rerank("query", [], [])
        assert results == []

    def test_top_k_larger_than_items(self):
        reranker = NoOpReranker()
        items = ["a", "b"]
        texts = ["doc a", "doc b"]

        results = reranker.rerank("query", items, texts, top_k=10)
        assert len(results) == 2


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker (tests fallback behavior since rerankers is optional)."""

    def test_falls_back_without_library(self):
        """CrossEncoderReranker should degrade to NoOp when rerankers is not installed."""
        reranker = CrossEncoderReranker()

        # Whether the library is installed or not, it should work
        items = ["a", "b", "c"]
        texts = ["document about auth", "document about testing", "document about deploy"]

        results = reranker.rerank("auth", items, texts)
        assert len(results) >= 1
        # All results should have indices and scores
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_handles_empty_input(self):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", [], [])
        assert results == []

    def test_respects_top_k(self):
        reranker = CrossEncoderReranker()
        items = list(range(10))
        texts = [f"doc {i}" for i in range(10)]

        results = reranker.rerank("query", items, texts, top_k=3)
        assert len(results) <= 3
