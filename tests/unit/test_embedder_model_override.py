"""
Unit tests for the ``model_name`` override on :class:`LocalEmbedder` and
its plumbing through :class:`benchmarks.locomo.runner.LoCoMoRunner`.

Background
----------
LoCoMo adversarial refusal experiments needed to swap the default
``all-MiniLM-L6-v2`` (384-dim) embedder for a stronger model
(``BAAI/bge-large-en-v1.5``, 1024-dim) to test whether OOD calibration
separates adversarial from real queries. This file pins three guarantees:

1. The default ``LocalEmbedder()`` continues to use ``all-MiniLM-L6-v2``
   (backward compat — default path must never change).
2. Explicit ``LocalEmbedder(model_name=...)`` stores the requested name
   without attempting to load the model (happy path, no network).
3. ``LoCoMoRunner(embedder_model=...)`` threads the override through to
   :class:`LocalEmbedder` when it lazily instantiates the shared
   embedder (integration happy path + default path + mock short-circuit
   edge case).

These tests do NOT download any models — the sentence-transformers
import is never triggered because we never call ``encode()``.
"""

from __future__ import annotations

import pytest

from alma.retrieval.embeddings import LocalEmbedder
from benchmarks.locomo.runner import LoCoMoRunner


class TestLocalEmbedderModelName:
    """LocalEmbedder model_name override behavior."""

    def test_default_model_is_minilm(self):
        """Happy path: no args => ALMA's default MiniLM-L6-v2."""
        embedder = LocalEmbedder()
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_explicit_bge_model_name(self):
        """Happy path: opt-in to BGE — name stored, no load yet."""
        embedder = LocalEmbedder(model_name="BAAI/bge-large-en-v1.5")
        assert embedder.model_name == "BAAI/bge-large-en-v1.5"
        # Lazy-load contract: _model must remain None until first encode().
        assert embedder._model is None

    def test_empty_model_name_is_stored_verbatim(self):
        """Edge case: empty string is stored as-is (fails later at load
        time with a clear sentence-transformers error, not silently
        here). This pins the current contract so a future refactor
        that adds eager validation is a conscious choice.
        """
        embedder = LocalEmbedder(model_name="")
        assert embedder.model_name == ""


class TestLoCoMoRunnerEmbedderModel:
    """Runner plumbs embedder_model through to LocalEmbedder."""

    def test_default_runner_uses_minilm(self, tmp_path):
        """Happy path: no override => default MiniLM."""
        runner = LoCoMoRunner(
            embedding_provider="local",
            tmp_dir=str(tmp_path),
        )
        embedder = runner._get_embedder()
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model_name == "all-MiniLM-L6-v2"

    def test_runner_honors_embedder_model_override(self, tmp_path):
        """Happy path: explicit BGE model propagates into LocalEmbedder."""
        runner = LoCoMoRunner(
            embedding_provider="local",
            tmp_dir=str(tmp_path),
            embedder_model="BAAI/bge-large-en-v1.5",
        )
        embedder = runner._get_embedder()
        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model_name == "BAAI/bge-large-en-v1.5"

    def test_runner_ignores_embedder_model_for_mock_provider(self, tmp_path):
        """Edge case: ``mock`` provider ignores the override entirely
        (MockEmbedder has no model concept). We pin this so a future
        refactor does not silently route BGE loads through the mock path.
        """
        runner = LoCoMoRunner(
            embedding_provider="mock",
            tmp_dir=str(tmp_path),
            embedder_model="BAAI/bge-large-en-v1.5",  # should be ignored
        )
        embedder = runner._get_embedder()
        # MockEmbedder does not expose .model_name — assert via type.
        assert not isinstance(embedder, LocalEmbedder)
        assert embedder.dimension == 384


class TestRunnerConstructorSurface:
    """Error paths on the new constructor surface."""

    def test_embedder_model_defaults_to_none(self):
        """Happy path: omitted => None (triggers default LocalEmbedder)."""
        runner = LoCoMoRunner()
        assert runner.embedder_model is None

    def test_embedder_model_accepts_none_explicitly(self):
        """Edge case: explicit None is valid (same as omitted)."""
        runner = LoCoMoRunner(embedder_model=None)
        assert runner.embedder_model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
