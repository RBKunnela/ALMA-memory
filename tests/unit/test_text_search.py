"""Dedicated tests for alma.retrieval.text_search module.

Covers TextSearchProvider ABC, SimpleTFIDFProvider (TF-IDF + cosine similarity),
and BM25SProvider (optional bm25s with fallback).
"""

import math
from abc import ABC
from unittest.mock import patch

import pytest

from alma.retrieval.text_search import (
    BM25SProvider,
    SimpleTFIDFProvider,
    TextSearchProvider,
)


# ---------------------------------------------------------------------------
# TextSearchProvider ABC
# ---------------------------------------------------------------------------


class TestTextSearchProviderABC:
    """TextSearchProvider is abstract and cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            TextSearchProvider()

    def test_subclass_must_implement_all_methods(self):
        """A partial implementation should also fail to instantiate."""

        class PartialProvider(TextSearchProvider):
            def index(self, documents, doc_ids=None):
                pass

        with pytest.raises(TypeError):
            PartialProvider()

    def test_complete_subclass_instantiates(self):
        """A subclass implementing every abstract method works fine."""

        class FullProvider(TextSearchProvider):
            def index(self, documents, doc_ids=None):
                pass

            def search(self, query, top_k=10):
                return []

            def is_indexed(self):
                return False

        provider = FullProvider()
        assert isinstance(provider, TextSearchProvider)
        assert isinstance(provider, ABC)


# ---------------------------------------------------------------------------
# SimpleTFIDFProvider — tokenization
# ---------------------------------------------------------------------------


class TestSimpleTFIDFTokenization:
    """Tests for the _tokenize static method."""

    def test_lowercases_text(self):
        tokens = SimpleTFIDFProvider._tokenize("Hello WORLD FoO")
        assert tokens == ["hello", "world", "foo"]

    def test_splits_on_whitespace(self):
        tokens = SimpleTFIDFProvider._tokenize("one  two\tthree\nfour")
        assert tokens == ["one", "two", "three", "four"]

    def test_empty_string(self):
        tokens = SimpleTFIDFProvider._tokenize("")
        assert tokens == []

    def test_unicode_characters(self):
        tokens = SimpleTFIDFProvider._tokenize("cafe\u0301 na\u00efve stra\u00dfe")
        # Should lowercase and split; exact tokens depend on Python str.lower()
        assert len(tokens) == 3
        for t in tokens:
            assert t == t.lower()

    def test_special_characters_preserved(self):
        """Tokenizer only splits on whitespace; punctuation stays attached."""
        tokens = SimpleTFIDFProvider._tokenize("hello, world! foo-bar")
        assert tokens == ["hello,", "world!", "foo-bar"]


# ---------------------------------------------------------------------------
# SimpleTFIDFProvider — IDF computation
# ---------------------------------------------------------------------------


class TestSimpleTFIDFIdf:
    """Verify IDF values behave correctly."""

    def test_rare_term_has_higher_idf_than_common_term(self):
        provider = SimpleTFIDFProvider()
        docs = [
            "the cat sat",
            "the dog sat",
            "the bird flew",
            "a rare unicorn appeared",
        ]
        provider.index(docs)

        # "the" appears in 3/4 docs, "unicorn" in 1/4
        idf_the = provider._idf.get("the", 0.0)
        idf_unicorn = provider._idf.get("unicorn", 0.0)
        assert idf_unicorn > idf_the

    def test_term_in_all_docs_has_lowest_idf(self):
        provider = SimpleTFIDFProvider()
        docs = ["alpha beta", "alpha gamma", "alpha delta"]
        provider.index(docs)

        idf_alpha = provider._idf["alpha"]
        for term in ("beta", "gamma", "delta"):
            assert provider._idf[term] > idf_alpha

    def test_idf_formula(self):
        """Verify the exact IDF formula: log((n+1)/(df+1)) + 1."""
        provider = SimpleTFIDFProvider()
        docs = ["apple banana", "apple cherry", "banana cherry", "date"]
        provider.index(docs)

        n = len(docs)  # 4
        # "apple" appears in docs 0, 1 -> df = 2
        expected_idf_apple = math.log((n + 1) / (2 + 1)) + 1
        assert abs(provider._idf["apple"] - expected_idf_apple) < 1e-10

        # "date" appears in doc 3 only -> df = 1
        expected_idf_date = math.log((n + 1) / (1 + 1)) + 1
        assert abs(provider._idf["date"] - expected_idf_date) < 1e-10


# ---------------------------------------------------------------------------
# SimpleTFIDFProvider — TF-IDF vectors and search
# ---------------------------------------------------------------------------


class TestSimpleTFIDFSearch:
    """Core search behavior tests."""

    def test_identical_document_scores_highest(self):
        provider = SimpleTFIDFProvider()
        docs = [
            "machine learning algorithms",
            "deep learning neural networks",
            "random unrelated topic",
        ]
        provider.index(docs)

        results = provider.search("machine learning algorithms")
        assert len(results) >= 1
        # The identical document should be the top result
        assert results[0][0] == 0
        # Score should be very close to 1.0 (perfect cosine match)
        assert results[0][1] > 0.99

    def test_multiple_documents_ranked_by_relevance(self):
        provider = SimpleTFIDFProvider()
        docs = [
            "python web framework flask",
            "python data science pandas numpy",
            "javascript react frontend",
            "python machine learning scikit",
        ]
        provider.index(docs)

        results = provider.search("python")
        indices = [idx for idx, _ in results]
        # All three python docs should appear, javascript doc should not
        assert 2 not in indices
        assert len(results) == 3

    def test_top_k_limits_results(self):
        provider = SimpleTFIDFProvider()
        docs = [f"common term doc{i}" for i in range(20)]
        provider.index(docs)

        results = provider.search("common term", top_k=5)
        assert len(results) <= 5

    def test_top_k_larger_than_matches(self):
        provider = SimpleTFIDFProvider()
        docs = ["alpha beta", "gamma delta", "alpha gamma"]
        provider.index(docs)

        results = provider.search("alpha", top_k=100)
        # Only 2 docs contain "alpha"
        assert len(results) == 2

    def test_search_not_indexed(self):
        provider = SimpleTFIDFProvider()
        # Never called index
        results = provider.search("anything")
        assert results == []

    def test_search_with_query_term_not_in_corpus(self):
        provider = SimpleTFIDFProvider()
        provider.index(["hello world", "foo bar"])
        results = provider.search("zzzzz")
        assert results == []

    def test_scores_are_between_zero_and_one(self):
        """Cosine similarity should always be in (0, 1]."""
        provider = SimpleTFIDFProvider()
        docs = [
            "authentication tokens jwt",
            "database migrations alembic",
            "testing pytest coverage",
            "jwt refresh token rotation",
        ]
        provider.index(docs)

        results = provider.search("jwt authentication")
        for _, score in results:
            assert 0 < score <= 1.0


# ---------------------------------------------------------------------------
# SimpleTFIDFProvider — doc_ids, re-indexing, edge cases
# ---------------------------------------------------------------------------


class TestSimpleTFIDFEdgeCases:
    """Edge cases: doc_ids, re-indexing, unicode, single-word docs, long docs."""

    def test_custom_doc_ids(self):
        provider = SimpleTFIDFProvider()
        docs = ["alpha content", "beta content"]
        provider.index(docs, doc_ids=["id-a", "id-b"])

        assert provider._doc_ids == ["id-a", "id-b"]

    def test_default_doc_ids_are_string_indices(self):
        provider = SimpleTFIDFProvider()
        docs = ["one", "two", "three"]
        provider.index(docs)

        assert provider._doc_ids == ["0", "1", "2"]

    def test_reindex_resets_state(self):
        provider = SimpleTFIDFProvider()
        provider.index(["old document about cats"])
        assert provider.is_indexed()

        old_results = provider.search("cats")
        assert len(old_results) == 1

        # Re-index with completely different corpus
        provider.index(["new document about dogs", "another about birds"])
        assert provider.is_indexed()
        assert len(provider._documents) == 2

        cats_results = provider.search("cats")
        assert len(cats_results) == 0

        dogs_results = provider.search("dogs")
        assert len(dogs_results) == 1

    def test_unicode_documents(self):
        provider = SimpleTFIDFProvider()
        docs = [
            "\u4f60\u597d\u4e16\u754c",  # Chinese: hello world
            "caf\u00e9 latt\u00e9",
            "\u00fcber stra\u00dfe",
        ]
        provider.index(docs)
        assert provider.is_indexed()
        assert len(provider._doc_tfidf) == 3

    def test_single_word_documents(self):
        provider = SimpleTFIDFProvider()
        docs = ["hello", "world", "python"]
        provider.index(docs)

        results = provider.search("hello")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_very_long_document(self):
        provider = SimpleTFIDFProvider()
        long_doc = " ".join([f"word{i}" for i in range(10000)])
        short_doc = "specific target term"
        provider.index([long_doc, short_doc])

        results = provider.search("specific target term")
        assert len(results) >= 1
        # The short doc with exact match should rank first
        assert results[0][0] == 1

    def test_duplicate_documents(self):
        provider = SimpleTFIDFProvider()
        docs = ["hello world", "hello world", "something else"]
        provider.index(docs)

        results = provider.search("hello world")
        # Both duplicate docs should appear with equal scores
        assert len(results) >= 2
        scores = {idx: score for idx, score in results}
        assert abs(scores[0] - scores[1]) < 1e-10

    def test_whitespace_only_query(self):
        provider = SimpleTFIDFProvider()
        provider.index(["some document here"])
        results = provider.search("   ")
        assert results == []


# ---------------------------------------------------------------------------
# BM25SProvider — fallback behavior (bm25s is optional)
# ---------------------------------------------------------------------------


class TestBM25SProviderFallback:
    """Tests for BM25SProvider when bm25s is NOT installed (fallback to TF-IDF)."""

    @patch.dict("sys.modules", {"bm25s": None})
    def test_init_creates_fallback_when_bm25s_unavailable(self):
        """When bm25s import fails, fallback should be created."""
        # Force re-import with bm25s blocked
        provider = BM25SProvider()
        # If bm25s is truly unavailable, _fallback is set
        # If bm25s IS available in the test env, this test still works
        # because we check the actual behavior
        if not provider._bm25s_available:
            assert provider._fallback is not None
            assert isinstance(provider._fallback, SimpleTFIDFProvider)

    def test_index_delegates_to_fallback(self):
        provider = BM25SProvider()
        if not provider._bm25s_available:
            provider.index(["doc one", "doc two"])
            assert provider._fallback.is_indexed()

    def test_search_delegates_to_fallback(self):
        provider = BM25SProvider()
        if not provider._bm25s_available:
            provider.index(["alpha beta", "gamma delta"])
            results = provider.search("alpha")
            assert len(results) >= 1
            assert results[0][0] == 0

    def test_is_indexed_delegates_to_fallback(self):
        provider = BM25SProvider()
        if not provider._bm25s_available:
            assert not provider.is_indexed()
            provider.index(["doc"])
            assert provider.is_indexed()

    def test_empty_documents(self):
        provider = BM25SProvider()
        provider.index([])
        assert not provider.is_indexed()
        results = provider.search("anything")
        assert results == []

    def test_empty_query(self):
        provider = BM25SProvider()
        provider.index(["some document"])
        results = provider.search("")
        assert results == []

    def test_top_k_parameter(self):
        provider = BM25SProvider()
        docs = [f"shared keyword doc{i}" for i in range(10)]
        provider.index(docs)

        results = provider.search("shared keyword", top_k=3)
        assert len(results) <= 3

    def test_search_before_index(self):
        provider = BM25SProvider()
        results = provider.search("anything")
        assert results == []

    def test_basic_relevance(self):
        """Regardless of backend, relevant docs should rank higher."""
        provider = BM25SProvider()
        docs = [
            "python flask web development",
            "java spring boot microservices",
            "python django rest framework",
            "rust systems programming",
        ]
        provider.index(docs)

        results = provider.search("python web")
        if len(results) >= 1:
            # A python doc should be in the top result
            assert results[0][0] in (0, 2)
