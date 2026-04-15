"""
Unit tests for the LoCoMo ``--min-score`` threshold plumbing.

Covers:
- ``LoCoMoRunner`` stores ``min_score`` and forwards it to the engine.
- The CLI parses ``--min-score`` correctly.
- Edge cases: default is 0.0, high thresholds pass through unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.locomo.runner import LoCoMoRunner  # noqa: E402

# ---------------------------------------------------------------------------
# [UNIT] LoCoMoRunner.__init__ -- should store min_score
# ---------------------------------------------------------------------------


def test_runner_stores_min_score_when_provided():
    """Happy path: explicit min_score is stored on self."""
    runner = LoCoMoRunner(embedding_provider="mock", top_k=5, min_score=0.5)
    assert runner.min_score == 0.5
    assert runner.top_k == 5
    runner.cleanup()


def test_runner_default_min_score_is_zero():
    """Edge: default keeps historical behavior (threshold=0.0)."""
    runner = LoCoMoRunner(embedding_provider="mock")
    assert runner.min_score == 0.0
    runner.cleanup()


def test_runner_accepts_high_min_score():
    """Edge: very high thresholds are stored verbatim (no clamping)."""
    runner = LoCoMoRunner(embedding_provider="mock", min_score=0.99)
    assert runner.min_score == 0.99
    runner.cleanup()


# ---------------------------------------------------------------------------
# [UNIT] RetrievalEngine construction -- should forward min_score_threshold
# ---------------------------------------------------------------------------


def test_run_conversation_forwards_min_score_to_engine():
    """Verify the engine is constructed with our min_score_threshold.

    We patch the heavy dependencies inside ``_run_conversation`` and
    assert the ``min_score_threshold`` kwarg equals the runner's
    ``min_score``. No real embedding or storage work is done.
    """
    runner = LoCoMoRunner(embedding_provider="mock", min_score=0.42)

    fake_conv = MagicMock()
    fake_conv.conv_id = "test-conv"
    fake_conv.qa_pairs = []  # no QA -> skip the query loop

    with (
        patch("alma.retrieval.engine.RetrievalEngine") as mock_engine_cls,
        patch("alma.storage.sqlite_local.SQLiteStorage") as mock_storage_cls,
        patch(
            "benchmarks.locomo.evidence_mapping.ingest_conversation",
            return_value={},
        ),
    ):
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_storage_cls.return_value = MagicMock()

        runner._run_conversation(fake_conv)

        assert mock_engine_cls.called, "RetrievalEngine was not constructed"
        kwargs = mock_engine_cls.call_args.kwargs
        assert kwargs["min_score_threshold"] == 0.42

    runner.cleanup()


# ---------------------------------------------------------------------------
# [UNIT] CLI argparse -- should parse --min-score
# ---------------------------------------------------------------------------


def test_cli_parses_min_score_flag():
    """Happy path: --min-score 0.3 lands on args.min_score as a float."""
    from benchmarks.locomo.runner import main as runner_main  # noqa: F401

    # Re-build the parser by introspecting the module source is brittle;
    # instead, we patch sys.argv and intercept the runner construction
    # to capture the parsed value without running the benchmark.
    captured: dict = {}

    class _Sentinel(Exception):
        pass

    def _fake_runner_init(self, **kwargs):
        captured.update(kwargs)
        raise _Sentinel("stop before run()")

    with patch.object(LoCoMoRunner, "__init__", _fake_runner_init):
        with patch(
            "benchmarks.locomo.dataset.LoCoMoDataset.load",
            return_value=MagicMock(
                conversations=[], summary=lambda: "empty", __len__=lambda s: 0
            ),
        ):
            old_argv = sys.argv
            sys.argv = [
                "runner",
                "--mode",
                "retrieval",
                "--embedding",
                "mock",
                "--min-score",
                "0.3",
                "--limit",
                "1",
            ]
            try:
                with pytest.raises(_Sentinel):
                    runner_main()
            finally:
                sys.argv = old_argv

    assert captured.get("min_score") == 0.3
    assert isinstance(captured.get("min_score"), float)


def test_cli_min_score_defaults_to_zero_when_omitted():
    """Edge: omitting --min-score yields the 0.0 default."""
    from benchmarks.locomo.runner import main as runner_main

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    def _fake_runner_init(self, **kwargs):
        captured.update(kwargs)
        raise _Sentinel("stop")

    with patch.object(LoCoMoRunner, "__init__", _fake_runner_init):
        with patch(
            "benchmarks.locomo.dataset.LoCoMoDataset.load",
            return_value=MagicMock(
                conversations=[], summary=lambda: "empty", __len__=lambda s: 0
            ),
        ):
            old_argv = sys.argv
            sys.argv = ["runner", "--mode", "retrieval", "--embedding", "mock"]
            try:
                with pytest.raises(_Sentinel):
                    runner_main()
            finally:
                sys.argv = old_argv

    assert captured.get("min_score") == 0.0
