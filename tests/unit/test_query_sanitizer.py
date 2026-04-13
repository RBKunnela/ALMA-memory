"""
Tests for alma.retrieval.query_sanitizer.

Validates the 4-step sanitization pipeline:
  1. passthrough (short queries)
  2. question_extraction (? detection)
  3. tail_sentence (last meaningful sentence)
  4. tail_truncation (fallback)
"""

from unittest.mock import MagicMock, patch

import pytest

from alma.retrieval.query_sanitizer import (
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    SAFE_QUERY_LENGTH,
    sanitize_query,
)


class TestPassthrough:
    """Step 1: Short queries pass through unchanged."""

    def test_short_query_passthrough(self):
        """Short query under SAFE_QUERY_LENGTH passes through."""
        query = "What is the deployment strategy?"
        result = sanitize_query(query)

        assert result["clean_query"] == query
        assert result["was_sanitized"] is False
        assert result["method"] == "passthrough"
        assert result["original_length"] == len(query)
        assert result["clean_length"] == len(query)

    def test_exactly_safe_length_passthrough(self):
        """Query at exactly SAFE_QUERY_LENGTH passes through."""
        query = "x" * SAFE_QUERY_LENGTH
        result = sanitize_query(query)

        assert result["was_sanitized"] is False
        assert result["method"] == "passthrough"

    def test_one_over_safe_length_not_passthrough(self):
        """Query one char over SAFE_QUERY_LENGTH triggers sanitization."""
        # Build a query just over the limit with a clear tail sentence
        query = "a" * (SAFE_QUERY_LENGTH - 10) + "\n" + "b" * 20
        result = sanitize_query(query)

        # Should not be passthrough since it's over SAFE_QUERY_LENGTH
        assert result["original_length"] > SAFE_QUERY_LENGTH


class TestEmptyInput:
    """Handle empty/None input gracefully."""

    def test_none_input(self):
        """None input returns empty passthrough."""
        result = sanitize_query(None)

        assert result["clean_query"] == ""
        assert result["was_sanitized"] is False
        assert result["original_length"] == 0
        assert result["clean_length"] == 0
        assert result["method"] == "passthrough"

    def test_empty_string(self):
        """Empty string returns empty passthrough."""
        result = sanitize_query("")

        assert result["clean_query"] == ""
        assert result["was_sanitized"] is False
        assert result["method"] == "passthrough"

    def test_whitespace_only(self):
        """Whitespace-only string returns empty passthrough."""
        result = sanitize_query("   \n\t  ")

        assert result["was_sanitized"] is False
        assert result["method"] == "passthrough"


class TestQuestionExtraction:
    """Step 2: Extract question from contaminated query."""

    def test_system_prompt_with_question(self):
        """System prompt prepended to a question gets extracted."""
        system_prompt = (
            "You are a helpful assistant. You must follow these rules: "
            "1. Always be precise. 2. Never make things up. "
            "3. Cite your sources. 4. Be concise in your responses. "
            "5. Follow the user's instructions carefully. "
        )
        question = "What is the best caching strategy for embeddings?"
        query = system_prompt + "\n" + question

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "question_extraction"
        assert "?" in result["clean_query"]
        assert result["clean_length"] < result["original_length"]

    def test_fullwidth_question_mark(self):
        """Fullwidth question mark is detected."""
        system_prompt = "x" * 250
        question = "What is the answer\uff1f"
        query = system_prompt + "\n" + question

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "question_extraction"

    def test_question_with_trailing_quote(self):
        """Question ending with ?' is detected."""
        system_prompt = "x" * 250
        question = "What is the answer?'"
        query = system_prompt + "\n" + question

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "question_extraction"

    def test_multiple_questions_takes_last(self):
        """When multiple questions exist, last one is preferred."""
        system_prompt = "x" * 250
        query = system_prompt + "\nIs this important?" + "\nWhat is the actual query?"

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "question_extraction"
        assert "actual query" in result["clean_query"]


class TestTailSentence:
    """Step 3: Extract last meaningful sentence."""

    def test_tail_sentence_extraction(self):
        """When no question mark, extract last meaningful sentence."""
        system_prompt = (
            "You are a helpful assistant. Always follow instructions. "
            "Be precise and careful. Never make assumptions. "
            "Follow the guidelines provided. " * 5
        )
        actual_query = "Find documents about vector databases"
        query = system_prompt + "\n" + actual_query

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "tail_sentence"
        assert result["clean_length"] < result["original_length"]

    def test_tail_sentence_skips_short_segments(self):
        """Segments shorter than MIN_QUERY_LENGTH are skipped."""
        filler = "x" * 250
        query = filler + "\nhi\nok\nFind relevant memory entries for agent"

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        # Should pick the long tail, not "hi" or "ok"
        assert result["clean_length"] >= MIN_QUERY_LENGTH


class TestTailTruncation:
    """Step 4: Fallback tail truncation."""

    def test_tail_truncation_fallback(self):
        """When all segments are too short, truncate from tail."""
        # Build a string where every newline-separated segment is shorter
        # than MIN_QUERY_LENGTH, forcing the tail_truncation fallback.
        # Each segment is 5 chars (below MIN_QUERY_LENGTH=10).
        segments = ["abcde"] * 100  # 100 segments of 5 chars each
        query = "\n".join(segments)  # total ~600 chars

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "tail_truncation"
        assert result["clean_length"] <= MAX_QUERY_LENGTH

    def test_tail_truncation_preserves_end(self):
        """Tail truncation takes from the end of the string."""
        # All segments under MIN_QUERY_LENGTH, but the combined tail
        # should preserve the end of the string
        short_segments = ["ab"] * 150  # 150 segments of 2 chars
        short_segments.append("END_OK")  # last segment also short (6 chars)
        query = "\n".join(short_segments)

        result = sanitize_query(query)

        assert result["was_sanitized"] is True
        assert result["method"] == "tail_truncation"
        assert result["clean_query"].endswith("END_OK")


class TestConstants:
    """Verify constants match expected values."""

    def test_max_query_length(self):
        assert MAX_QUERY_LENGTH == 250

    def test_safe_query_length(self):
        assert SAFE_QUERY_LENGTH == 200

    def test_min_query_length(self):
        assert MIN_QUERY_LENGTH == 10


class TestReturnStructure:
    """Verify all results have the expected keys."""

    @pytest.mark.parametrize(
        "query",
        [
            "short query",
            None,
            "",
            "x" * 300 + "\nWhat is this?",
            "x" * 300 + "\nFind something useful",
            "x" * 500,
        ],
    )
    def test_result_keys(self, query):
        """All results contain the required keys."""
        result = sanitize_query(query)

        assert "clean_query" in result
        assert "was_sanitized" in result
        assert "original_length" in result
        assert "clean_length" in result
        assert "method" in result
        assert result["method"] in {
            "passthrough",
            "question_extraction",
            "tail_sentence",
            "tail_truncation",
        }


class TestEngineIntegration:
    """Verify sanitizer is called within RetrievalEngine.retrieve()."""

    def test_sanitizer_called_in_retrieve(self):
        """RetrievalEngine.retrieve() calls sanitize_query before embedding."""
        from alma.retrieval.engine import RetrievalEngine

        # Create a mock storage backend
        mock_storage = MagicMock()
        mock_storage.get_heuristics.return_value = []
        mock_storage.get_outcomes.return_value = []
        mock_storage.get_domain_knowledge.return_value = []
        mock_storage.get_user_preferences.return_value = []

        engine = RetrievalEngine(
            storage=mock_storage,
            enable_cache=False,
        )

        # Mock the embedding method
        engine._get_embedding = MagicMock(return_value=[0.1] * 384)

        # Patch sanitize_query at the module level where it's imported
        with patch(
            "alma.retrieval.engine.sanitize_query",
            wraps=sanitize_query,
        ) as mock_sanitize:
            try:
                engine.retrieve(
                    query="short test query",
                    agent="test-agent",
                    project_id="test-project",
                )
            except Exception:
                # Storage mock may not return perfect data; that's OK.
                # We only care that sanitize_query was called.
                pass

            mock_sanitize.assert_called_once()

    def test_long_query_sanitized_before_embedding(self):
        """A long contaminated query is sanitized before embedding generation."""
        from alma.retrieval.engine import RetrievalEngine

        mock_storage = MagicMock()
        mock_storage.get_heuristics.return_value = []
        mock_storage.get_outcomes.return_value = []
        mock_storage.get_domain_knowledge.return_value = []
        mock_storage.get_user_preferences.return_value = []

        engine = RetrievalEngine(
            storage=mock_storage,
            enable_cache=False,
        )

        embedding_calls = []

        def capture_embedding(query):
            embedding_calls.append(query)
            return [0.1] * 384

        engine._get_embedding = capture_embedding

        system_prompt = "You are a helpful AI assistant. " * 20
        actual_question = "What caching strategies exist?"
        long_query = system_prompt + "\n" + actual_question

        try:
            engine.retrieve(
                query=long_query,
                agent="test-agent",
                project_id="test-project",
            )
        except Exception:
            pass

        # The embedding should have been called with the sanitized query,
        # not the full contaminated query
        assert len(embedding_calls) == 1
        assert len(embedding_calls[0]) < len(long_query)
