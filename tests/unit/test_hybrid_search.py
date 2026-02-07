"""Tests for hybrid search (BM25/TFIDF + Vector with RRF fusion)."""

from alma.retrieval.hybrid import HybridResult, HybridSearchConfig, HybridSearchEngine
from alma.retrieval.text_search import SimpleTFIDFProvider


class TestSimpleTFIDFProvider:
    """Tests for the pure-Python TF-IDF fallback."""

    def test_basic_search(self):
        provider = SimpleTFIDFProvider()
        docs = [
            "JWT authentication with refresh tokens",
            "database migration using alembic",
            "REST API endpoint testing",
            "JWT token validation and expiry",
        ]
        provider.index(docs)

        results = provider.search("JWT token", top_k=2)
        assert len(results) >= 1
        # JWT docs should rank highest
        top_indices = [idx for idx, _ in results]
        assert 0 in top_indices or 3 in top_indices

    def test_empty_corpus(self):
        provider = SimpleTFIDFProvider()
        provider.index([])
        assert provider.search("anything") == []

    def test_empty_query(self):
        provider = SimpleTFIDFProvider()
        provider.index(["some document"])
        assert provider.search("") == []

    def test_no_match(self):
        provider = SimpleTFIDFProvider()
        provider.index(["python programming language"])
        results = provider.search("quantum physics entanglement")
        assert len(results) == 0

    def test_is_indexed(self):
        provider = SimpleTFIDFProvider()
        assert not provider.is_indexed()
        provider.index(["doc one", "doc two"])
        assert provider.is_indexed()

    def test_single_document(self):
        provider = SimpleTFIDFProvider()
        provider.index(["hello world"])
        results = provider.search("hello", top_k=5)
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0


class TestHybridSearchConfig:
    """Tests for hybrid search configuration."""

    def test_default_config(self):
        config = HybridSearchConfig()
        assert config.rrf_k == 60
        assert abs(config.vector_weight - 0.5) < 0.01
        assert abs(config.text_weight - 0.5) < 0.01

    def test_weight_normalization(self):
        config = HybridSearchConfig(vector_weight=3.0, text_weight=1.0)
        assert abs(config.vector_weight - 0.75) < 0.01
        assert abs(config.text_weight - 0.25) < 0.01


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_basic_fusion(self):
        engine = HybridSearchEngine()

        # Vector search: doc 0 best, then 1, then 2
        vector_results = [(0, 0.95), (1, 0.80), (2, 0.70)]
        # Text search: doc 2 best, then 0, then 3
        text_results = [(2, 5.0), (0, 3.0), (3, 1.0)]

        fused = engine.fuse(vector_results, text_results)

        # All 4 unique docs should appear
        indices = {r.index for r in fused}
        assert indices == {0, 1, 2, 3}

        # Doc 0 and doc 2 appear in both -- should score highest
        top_2_indices = {fused[0].index, fused[1].index}
        assert 0 in top_2_indices
        assert 2 in top_2_indices

    def test_rrf_score_formula(self):
        """Verify exact RRF score computation."""
        config = HybridSearchConfig(rrf_k=60, vector_weight=0.5, text_weight=0.5)
        engine = HybridSearchEngine(config=config)

        # Doc at rank 1 in vector, rank 2 in text
        vector_results = [(0, 0.9)]
        text_results = [(1, 3.0), (0, 2.0)]

        fused = engine.fuse(vector_results, text_results)
        doc0 = next(r for r in fused if r.index == 0)

        # Expected: 0.5 * 1/(60+1) + 0.5 * 1/(60+2) = 0.5/61 + 0.5/62
        expected = 0.5 / 61 + 0.5 / 62
        assert abs(doc0.rrf_score - expected) < 1e-10

    def test_vector_only_results(self):
        engine = HybridSearchEngine()
        vector_results = [(0, 0.9), (1, 0.7)]
        fused = engine.fuse(vector_results, text_results=[])

        assert len(fused) == 2
        assert fused[0].text_rank is None
        assert fused[0].vector_rank == 1

    def test_text_only_results(self):
        engine = HybridSearchEngine()
        text_results = [(2, 5.0), (3, 3.0)]
        fused = engine.fuse(vector_results=[], text_results=text_results)

        assert len(fused) == 2
        assert fused[0].vector_rank is None
        assert fused[0].text_rank == 1

    def test_empty_inputs(self):
        engine = HybridSearchEngine()
        fused = engine.fuse([], [])
        assert fused == []

    def test_items_passthrough(self):
        engine = HybridSearchEngine()
        items = ["doc_a", "doc_b", "doc_c"]
        vector_results = [(0, 0.9), (2, 0.7)]
        text_results = [(1, 3.0)]

        fused = engine.fuse(vector_results, text_results, items=items)
        for r in fused:
            assert r.item == items[r.index]

    def test_provenance_tracking(self):
        engine = HybridSearchEngine()
        vector_results = [(0, 0.95)]
        text_results = [(0, 5.0), (1, 3.0)]

        fused = engine.fuse(vector_results, text_results)
        doc0 = next(r for r in fused if r.index == 0)

        assert doc0.vector_rank == 1
        assert doc0.text_rank == 1
        assert doc0.vector_score == 0.95
        assert doc0.text_score == 5.0


class TestHybridSearchEngineIntegration:
    """Integration tests for the full hybrid search flow."""

    def test_index_and_search(self):
        engine = HybridSearchEngine(
            config=HybridSearchConfig(text_provider="tfidf")
        )
        corpus = [
            "deploy auth service using kubernetes",
            "database backup and restore procedures",
            "JWT token rotation strategy",
            "kubernetes pod scaling configuration",
        ]
        engine.index_corpus(corpus)

        text_results = engine.text_search("kubernetes deploy", top_k=3)
        assert len(text_results) >= 1
        # kubernetes docs should match
        top_indices = [idx for idx, _ in text_results]
        assert 0 in top_indices or 3 in top_indices

    def test_full_hybrid_flow(self):
        engine = HybridSearchEngine(
            config=HybridSearchConfig(text_provider="tfidf")
        )
        corpus = [
            "use retry with exponential backoff for API calls",
            "validate form inputs before submission",
            "API rate limiting with token bucket",
        ]
        engine.index_corpus(corpus)

        # Simulate vector search (indices into corpus)
        vector_results = [(1, 0.9), (0, 0.7), (2, 0.5)]
        text_results = engine.text_search("API rate limiting", top_k=3)

        fused = engine.fuse(vector_results, text_results, items=corpus)
        assert len(fused) >= 1
        # Doc about API rate limiting should be boosted
        assert any(r.item and "rate limiting" in r.item for r in fused[:2])
