"""
Unit tests for MemorySlice token handling.

Tests the to_prompt() method with the new tokenizer integration.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    MemorySlice,
    UserPreference,
)


class TestMemorySliceToPrompt:
    """Tests for MemorySlice.to_prompt() with token handling."""

    @pytest.fixture
    def sample_heuristics(self):
        """Create sample heuristics."""
        now = datetime.now(timezone.utc)
        return [
            Heuristic(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="test-project",
                condition="form testing with multiple inputs",
                strategy="validate each input individually before full form submit",
                confidence=0.85,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now - timedelta(days=30),
            ),
            Heuristic(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="test-project",
                condition="modal dialog testing",
                strategy="wait for animation before interacting",
                confidence=0.90,
                occurrence_count=15,
                success_count=14,
                last_validated=now,
                created_at=now - timedelta(days=20),
            ),
        ]

    @pytest.fixture
    def sample_anti_patterns(self):
        """Create sample anti-patterns."""
        now = datetime.now(timezone.utc)
        return [
            AntiPattern(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="test-project",
                pattern="Using fixed sleep() for async waits",
                why_bad="Causes flaky tests, doesn't adapt to load",
                better_alternative="Use explicit waits with conditions",
                occurrence_count=5,
                last_seen=now,
                created_at=now - timedelta(days=10),
            ),
        ]

    @pytest.fixture
    def sample_preferences(self):
        """Create sample user preferences."""
        now = datetime.now(timezone.utc)
        return [
            UserPreference(
                id=str(uuid.uuid4()),
                user_id="test-user",
                category="code_style",
                preference="No emojis in documentation",
                source="explicit_instruction",
                confidence=1.0,
                timestamp=now,
            ),
        ]

    @pytest.fixture
    def sample_domain_knowledge(self):
        """Create sample domain knowledge."""
        now = datetime.now(timezone.utc)
        return [
            DomainKnowledge(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="test-project",
                domain="selector_patterns",
                fact="data-testid selectors are most stable for testing",
                source="test_run:stability=0.95",
                confidence=0.95,
                last_verified=now,
            ),
        ]

    @pytest.fixture
    def populated_memory_slice(
        self,
        sample_heuristics,
        sample_anti_patterns,
        sample_preferences,
        sample_domain_knowledge,
    ):
        """Create a populated memory slice."""
        return MemorySlice(
            heuristics=sample_heuristics,
            anti_patterns=sample_anti_patterns,
            preferences=sample_preferences,
            domain_knowledge=sample_domain_knowledge,
            query="test query",
            agent="test",
            retrieval_time_ms=10,
        )

    def test_to_prompt_returns_string(self, populated_memory_slice):
        """Test that to_prompt returns a string."""
        result = populated_memory_slice.to_prompt()
        assert isinstance(result, str)

    def test_to_prompt_contains_sections(self, populated_memory_slice):
        """Test that to_prompt includes all section headers."""
        result = populated_memory_slice.to_prompt()
        assert "## Relevant Strategies" in result
        assert "## Avoid These Patterns" in result
        assert "## User Preferences" in result
        assert "## Domain Context" in result

    def test_to_prompt_contains_heuristic_data(self, populated_memory_slice):
        """Test that to_prompt includes heuristic content."""
        result = populated_memory_slice.to_prompt()
        # Check for some expected content
        assert "form testing with multiple inputs" in result
        assert "validate each input individually" in result

    def test_to_prompt_contains_anti_pattern_data(self, populated_memory_slice):
        """Test that to_prompt includes anti-pattern content."""
        result = populated_memory_slice.to_prompt()
        assert "Using fixed sleep()" in result
        assert "flaky tests" in result
        assert "explicit waits" in result

    def test_to_prompt_contains_preference_data(self, populated_memory_slice):
        """Test that to_prompt includes preference content."""
        result = populated_memory_slice.to_prompt()
        assert "No emojis in documentation" in result

    def test_to_prompt_contains_domain_knowledge_data(self, populated_memory_slice):
        """Test that to_prompt includes domain knowledge content."""
        result = populated_memory_slice.to_prompt()
        assert "data-testid selectors" in result

    def test_to_prompt_with_default_max_tokens(self, populated_memory_slice):
        """Test to_prompt with default max_tokens."""
        result = populated_memory_slice.to_prompt()
        # Default is 2000 tokens, which should be plenty for our test data
        assert "[truncated]" not in result

    def test_to_prompt_with_small_max_tokens(self, populated_memory_slice):
        """Test to_prompt with small max_tokens triggers truncation."""
        result = populated_memory_slice.to_prompt(max_tokens=10)
        assert "[truncated]" in result
        # Should be shorter than the untruncated version
        full_result = populated_memory_slice.to_prompt(max_tokens=10000)
        assert len(result) < len(full_result)

    def test_to_prompt_with_model_specification(self, populated_memory_slice):
        """Test to_prompt with specific model for tokenization."""
        # Should work with various models
        result_gpt4 = populated_memory_slice.to_prompt(model="gpt-4")
        result_claude = populated_memory_slice.to_prompt(model="claude-3-sonnet")
        result_llama = populated_memory_slice.to_prompt(model="llama-3-70b")

        # All should return valid strings
        assert isinstance(result_gpt4, str)
        assert isinstance(result_claude, str)
        assert isinstance(result_llama, str)

        # Content should be similar (no truncation with default budget)
        assert "Relevant Strategies" in result_gpt4
        assert "Relevant Strategies" in result_claude
        assert "Relevant Strategies" in result_llama

    def test_to_prompt_empty_slice(self):
        """Test to_prompt with empty memory slice."""
        empty_slice = MemorySlice()
        result = empty_slice.to_prompt()
        assert result == ""

    def test_to_prompt_only_heuristics(self, sample_heuristics):
        """Test to_prompt with only heuristics."""
        slice = MemorySlice(heuristics=sample_heuristics)
        result = slice.to_prompt()
        assert "## Relevant Strategies" in result
        assert "## Avoid These Patterns" not in result

    def test_to_prompt_only_preferences(self, sample_preferences):
        """Test to_prompt with only preferences."""
        slice = MemorySlice(preferences=sample_preferences)
        result = slice.to_prompt()
        assert "## User Preferences" in result
        assert "## Relevant Strategies" not in result

    def test_to_prompt_heuristics_sorted_by_confidence(self):
        """Test that heuristics are sorted by confidence (highest first)."""
        now = datetime.now(timezone.utc)
        heuristics = [
            Heuristic(
                id="low",
                agent="test",
                project_id="test",
                condition="low confidence",
                strategy="low strategy",
                confidence=0.3,
                occurrence_count=5,
                success_count=2,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="high",
                agent="test",
                project_id="test",
                condition="high confidence",
                strategy="high strategy",
                confidence=0.95,
                occurrence_count=20,
                success_count=19,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="medium",
                agent="test",
                project_id="test",
                condition="medium confidence",
                strategy="medium strategy",
                confidence=0.6,
                occurrence_count=10,
                success_count=6,
                last_validated=now,
                created_at=now,
            ),
        ]

        slice = MemorySlice(heuristics=heuristics)
        result = slice.to_prompt()

        # High confidence should appear before low confidence
        high_pos = result.find("high confidence")
        medium_pos = result.find("medium confidence")
        low_pos = result.find("low confidence")

        assert high_pos < medium_pos < low_pos

    def test_to_prompt_truncation_preserves_structure(self):
        """Test that truncation preserves prompt structure when possible."""
        now = datetime.now(timezone.utc)
        # Create a lot of content that will need truncation
        many_heuristics = [
            Heuristic(
                id=str(i),
                agent="test",
                project_id="test",
                condition=f"condition number {i} with some additional text to pad",
                strategy=f"strategy number {i} with more text for padding purposes",
                confidence=0.8,
                occurrence_count=10,
                success_count=8,
                last_validated=now,
                created_at=now,
            )
            for i in range(50)
        ]

        slice = MemorySlice(heuristics=many_heuristics)
        result = slice.to_prompt(max_tokens=100)

        # Should still start with the section header
        assert result.startswith("## Relevant Strategies")
        # Should end with truncation marker
        assert result.endswith("[truncated]")


class TestMemorySliceTokenBudget:
    """Tests for token budget enforcement in MemorySlice."""

    def test_large_content_respects_token_limit(self):
        """Test that large content is properly truncated."""
        now = datetime.now(timezone.utc)

        # Create very long heuristic content
        long_heuristics = [
            Heuristic(
                id=str(i),
                agent="test",
                project_id="test",
                condition="A" * 500,  # Very long condition
                strategy="B" * 500,  # Very long strategy
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
            )
            for i in range(10)
        ]

        slice = MemorySlice(heuristics=long_heuristics)

        # With small token limit, should be truncated
        result_small = slice.to_prompt(max_tokens=50)
        result_large = slice.to_prompt(max_tokens=5000)

        assert len(result_small) < len(result_large)
        assert "[truncated]" in result_small
