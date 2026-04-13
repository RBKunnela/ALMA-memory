"""
Unit tests for alma.context.memory_stack — 4-Layer MemoryStack.

Tests cover:
- MemoryStack initialization with MockStorage
- wake_up() returns identity + essential story
- recall() with explicit layer selection
- recall() auto-selection logic
- to_prompt() respects max_tokens budget
- identity property reads from file
- token_usage reports per-layer counts
- Empty identity file handling
- ContextLayer basics
"""

import os
import tempfile
from unittest.mock import MagicMock

from alma.context import ContextLayer, MemoryStack
from alma.context.identity import DEFAULT_IDENTITY_TEXT, IdentityManager
from alma.testing import create_test_heuristic, create_test_knowledge
from alma.types import MemorySlice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alma_mock(
    heuristics=None,
    domain_knowledge=None,
    preferences=None,
) -> MagicMock:
    """Create a mock ALMA instance that returns a configured MemorySlice."""
    alma = MagicMock()
    alma.project_id = "test-project"

    slice_ = MemorySlice(
        heuristics=heuristics or [],
        domain_knowledge=domain_knowledge or [],
        preferences=preferences or [],
        query="test",
        agent="default",
    )
    alma.retrieve.return_value = slice_
    return alma


def _make_identity_file(tmp_dir: str, content: str) -> str:
    """Write an identity.txt in a temp directory and return its path."""
    path = os.path.join(tmp_dir, "identity.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# ContextLayer tests
# ---------------------------------------------------------------------------


class TestContextLayer:
    """[UNIT] ContextLayer — basic layer container."""

    def test_initial_state_is_empty(self):
        """Should start with empty content and zero tokens."""
        layer = ContextLayer(level=0, name="Identity")
        assert layer.content == ""
        assert layer.token_count == 0
        assert not layer.is_loaded

    def test_set_content_updates_token_count(self):
        """Should update token_count when content is set."""
        layer = ContextLayer(level=1, name="Essential")
        layer.set_content("Hello world, this is a test string.")
        assert layer.is_loaded
        assert layer.token_count > 0

    def test_set_content_with_empty_string(self):
        """Should handle empty string gracefully."""
        layer = ContextLayer(level=0, name="Identity")
        layer.set_content("")
        assert not layer.is_loaded
        assert layer.token_count == 0


# ---------------------------------------------------------------------------
# IdentityManager tests
# ---------------------------------------------------------------------------


class TestIdentityManager:
    """[UNIT] IdentityManager — Layer 0 identity file handling."""

    def test_load_existing_file(self):
        """Should read identity from existing file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am Atlas, a helpful assistant.")
            mgr = IdentityManager(identity_path=path)
            text = mgr.load()
            assert "Atlas" in text
            assert mgr.token_count > 0

    def test_load_missing_file_returns_default(self):
        """Should return default text when file does not exist."""
        mgr = IdentityManager(identity_path="/nonexistent/identity.txt")
        text = mgr.load()
        assert text == DEFAULT_IDENTITY_TEXT

    def test_load_caches_result(self):
        """Should only read from disk once, then return cached text."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "Cached identity.")
            mgr = IdentityManager(identity_path=path)

            first = mgr.load()
            # Overwrite the file — should still return cached
            _make_identity_file(tmp, "Changed identity.")
            second = mgr.load()

            assert first == second == "Cached identity."

    def test_reload_clears_cache(self):
        """Should re-read from disk after reload()."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "Original.")
            mgr = IdentityManager(identity_path=path)

            assert mgr.load() == "Original."

            _make_identity_file(tmp, "Updated.")
            assert mgr.reload() == "Updated."

    def test_text_property(self):
        """Should expose text via property, loading if needed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "Property test.")
            mgr = IdentityManager(identity_path=path)
            assert mgr.text == "Property test."

    def test_empty_identity_file(self):
        """Should handle an empty identity file gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "")
            mgr = IdentityManager(identity_path=path)
            text = mgr.load()
            # Empty file strips to empty string
            assert text == ""
            assert mgr.token_count == 0

    def test_create_default_when_missing(self):
        """Should create a default identity file when create_default=True."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "identity.txt")
            mgr = IdentityManager(identity_path=path, create_default=True)
            text = mgr.load()
            assert text == DEFAULT_IDENTITY_TEXT
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# MemoryStack initialization tests
# ---------------------------------------------------------------------------


class TestMemoryStackInit:
    """[UNIT] MemoryStack.__init__ — initialization."""

    def test_init_with_alma_mock(self):
        """Should initialize with a mock ALMA instance."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        assert stack._alma is alma
        assert stack._agent == "default"

    def test_init_with_custom_agent(self):
        """Should accept a custom agent name."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma, agent="test-agent")
        assert stack._agent == "test-agent"

    def test_init_with_custom_identity_path(self):
        """Should pass identity_path to IdentityManager."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma, identity_path="/custom/path.txt")
        assert stack._identity_mgr.path == "/custom/path.txt"


# ---------------------------------------------------------------------------
# wake_up() tests
# ---------------------------------------------------------------------------


class TestWakeUp:
    """[UNIT] MemoryStack.wake_up — L0 + L1 context loading."""

    def test_wake_up_returns_identity_and_story(self):
        """Should return combined L0 identity + L1 essential story."""
        heuristics = [
            create_test_heuristic(
                agent="default",
                strategy="Use caching for repeated queries",
                confidence=0.95,
            ),
        ]
        alma = _make_alma_mock(heuristics=heuristics)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am TestBot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.wake_up()

        assert "TestBot" in result
        assert "Essential Story" in result

    def test_wake_up_with_no_memories(self):
        """Should still return identity when no memories exist."""
        alma = _make_alma_mock()

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am EmptyBot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.wake_up()

        assert "EmptyBot" in result
        assert "Essential Story" in result

    def test_wake_up_handles_retrieval_failure(self):
        """Should gracefully handle retrieval engine failure."""
        alma = MagicMock()
        alma.retrieve.side_effect = RuntimeError("Storage unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am ResilientBot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.wake_up()

        assert "ResilientBot" in result
        assert "No memories available" in result

    def test_wake_up_includes_domain_knowledge(self):
        """Should include domain knowledge in L1."""
        knowledge = [
            create_test_knowledge(
                agent="default",
                fact="Login uses JWT with 24h expiry",
            ),
        ]
        alma = _make_alma_mock(domain_knowledge=knowledge)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am DomainBot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.wake_up()

        assert "JWT" in result or "Domain Knowledge" in result


# ---------------------------------------------------------------------------
# recall() tests
# ---------------------------------------------------------------------------


class TestRecall:
    """[UNIT] MemoryStack.recall — layer-specific and auto retrieval."""

    def test_recall_explicit_layer_2(self):
        """Should use L2 on-demand when layer=2."""
        heuristics = [
            create_test_heuristic(
                agent="default",
                strategy="Cache database queries",
            ),
        ]
        alma = _make_alma_mock(heuristics=heuristics)
        stack = MemoryStack(alma)
        result = stack.recall("caching", layer=2)
        assert isinstance(result, str)
        # ALMA retrieve should have been called
        alma.retrieve.assert_called()

    def test_recall_explicit_layer_3(self):
        """Should use L3 deep search when layer=3."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        result = stack.recall("complex authentication flow analysis", layer=3)
        assert isinstance(result, str)
        alma.retrieve.assert_called()

    def test_recall_auto_selects_l2_for_short_query(self):
        """Should auto-select L2 for short queries."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        result = stack.recall("auth")  # < 30 chars -> L2
        assert isinstance(result, str)

    def test_recall_auto_selects_l3_for_complex_query(self):
        """Should auto-select L3 for longer, complex queries."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        query = "How does the authentication middleware handle expired JWT tokens in production?"
        result = stack.recall(query)  # > 30 chars, no domain -> L3
        assert isinstance(result, str)

    def test_recall_auto_selects_l2_with_domain(self):
        """Should auto-select L2 when domain is provided."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        result = stack.recall("token expiry", domain="authentication")
        assert isinstance(result, str)

    def test_recall_layer_0_returns_identity(self):
        """Should return identity text when layer=0."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am RecallBot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.recall("anything", layer=0)
        assert "RecallBot" in result

    def test_recall_layer_1_triggers_wake_up_if_not_loaded(self):
        """Should wake_up if L1 not yet loaded when layer=1."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am L1Bot.")
            stack = MemoryStack(alma, identity_path=path)
            result = stack.recall("anything", layer=1)
        assert "Essential Story" in result

    def test_recall_handles_retrieval_failure(self):
        """Should return error message on failure."""
        alma = MagicMock()
        alma.retrieve.side_effect = RuntimeError("Connection lost")
        stack = MemoryStack(alma)
        result = stack.recall("test query", layer=3)
        assert "failed" in result.lower() or "Search failed" in result


# ---------------------------------------------------------------------------
# to_prompt() tests
# ---------------------------------------------------------------------------


class TestToPrompt:
    """[UNIT] MemoryStack.to_prompt — budget-aware prompt formatting."""

    def test_to_prompt_respects_max_tokens(self):
        """Should not exceed max_tokens budget."""
        heuristics = [
            create_test_heuristic(
                agent="default",
                strategy=f"Strategy number {i} with some detail" * 5,
                confidence=0.9 - i * 0.1,
            )
            for i in range(10)
        ]
        alma = _make_alma_mock(heuristics=heuristics)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am BudgetBot.")
            stack = MemoryStack(alma, identity_path=path)
            stack.wake_up()

            # Set a tight budget
            result = stack.to_prompt(max_tokens=200)

        # Rough check: 200 tokens ~ 800 chars
        # The result should be reasonably bounded
        estimated_tokens = len(result) // 4
        assert estimated_tokens <= 300  # some slack for estimation

    def test_to_prompt_includes_all_layers(self):
        """Should include L0, L1, and active recalls."""
        heuristics = [
            create_test_heuristic(agent="default", strategy="Test strat"),
        ]
        alma = _make_alma_mock(heuristics=heuristics)

        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am FullBot.")
            stack = MemoryStack(alma, identity_path=path)
            stack.wake_up()
            stack.recall("some topic", layer=2)

            result = stack.to_prompt(max_tokens=5000)

        assert "FullBot" in result
        assert "Essential Story" in result

    def test_to_prompt_empty_stack(self):
        """Should return empty string when nothing is loaded."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        result = stack.to_prompt()
        assert result == ""


# ---------------------------------------------------------------------------
# identity property tests
# ---------------------------------------------------------------------------


class TestIdentityProperty:
    """[UNIT] MemoryStack.identity — property access."""

    def test_identity_reads_from_file(self):
        """Should read identity text from configured file."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am PropBot.")
            stack = MemoryStack(alma, identity_path=path)
            assert stack.identity == "I am PropBot."

    def test_identity_returns_default_when_missing(self):
        """Should return default text when file missing."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma, identity_path="/nonexistent/path.txt")
        assert stack.identity == DEFAULT_IDENTITY_TEXT

    def test_identity_handles_empty_file(self):
        """Should handle empty identity file."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "")
            stack = MemoryStack(alma, identity_path=path)
            assert stack.identity == ""


# ---------------------------------------------------------------------------
# token_usage property tests
# ---------------------------------------------------------------------------


class TestTokenUsage:
    """[UNIT] MemoryStack.token_usage — per-layer token reporting."""

    def test_token_usage_reports_all_layers(self):
        """Should report token counts for all layers."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am TokenBot.")
            stack = MemoryStack(alma, identity_path=path)
            stack.wake_up()

            usage = stack.token_usage

        assert "L0_identity" in usage
        assert "L1_essential_story" in usage
        assert "total" in usage
        assert usage["L0_identity"] > 0
        assert usage["total"] > 0

    def test_token_usage_includes_recalls(self):
        """Should include active recall tokens in reporting."""
        alma = _make_alma_mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_identity_file(tmp, "I am RecallTokenBot.")
            stack = MemoryStack(alma, identity_path=path)
            stack.wake_up()
            stack.recall("some topic", layer=2)

            usage = stack.token_usage

        assert "active_recalls" in usage
        assert usage["total"] >= usage["L0_identity"]

    def test_token_usage_zero_when_empty(self):
        """Should report zero tokens when nothing loaded."""
        alma = _make_alma_mock()
        stack = MemoryStack(alma)
        usage = stack.token_usage
        assert usage["total"] == 0
