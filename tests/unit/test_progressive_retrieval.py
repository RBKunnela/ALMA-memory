"""
Unit tests for ALMA Progressive Disclosure.

Tests DisclosureLevel, MemorySummary, ProgressiveSlice,
SummaryExtractor, and ProgressiveRetrieval.
"""

import pytest

from alma.retrieval.progressive import (
    DisclosureLevel,
    MemorySummary,
    ProgressiveRetrieval,
    ProgressiveSlice,
    SummaryExtractor,
)
from alma.testing.factories import (
    create_test_anti_pattern,
    create_test_heuristic,
    create_test_knowledge,
    create_test_outcome,
    create_test_preference,
)


class TestDisclosureLevel:
    """Tests for DisclosureLevel enum."""

    def test_ordering(self):
        """Levels should increase from REFERENCE to FULL."""
        assert DisclosureLevel.REFERENCE.value < DisclosureLevel.SUMMARY.value
        assert DisclosureLevel.SUMMARY.value < DisclosureLevel.KEY_DETAILS.value
        assert DisclosureLevel.KEY_DETAILS.value < DisclosureLevel.FULL.value

    def test_all_four_levels(self):
        assert len(DisclosureLevel) == 4


class TestMemorySummary:
    """Tests for MemorySummary dataclass."""

    def test_get_full_returns_item(self):
        item = {"key": "value"}
        summary = MemorySummary(
            id="s1",
            memory_type="heuristic",
            summary="test",
            relevance_hint="hint",
            estimated_tokens=10,
            _full_item=item,
        )
        assert summary.get_full() == item

    def test_get_full_returns_none_when_absent(self):
        summary = MemorySummary(
            id="s1",
            memory_type="heuristic",
            summary="test",
            relevance_hint="hint",
            estimated_tokens=10,
        )
        assert summary.get_full() is None

    def test_default_disclosure_level(self):
        summary = MemorySummary(
            id="s1",
            memory_type="heuristic",
            summary="test",
            relevance_hint="hint",
            estimated_tokens=10,
        )
        assert summary.disclosure_level == DisclosureLevel.SUMMARY

    def test_key_fields_default_empty(self):
        summary = MemorySummary(
            id="s1",
            memory_type="heuristic",
            summary="test",
            relevance_hint="hint",
            estimated_tokens=10,
        )
        assert summary.key_fields == {}


class TestProgressiveSlice:
    """Tests for ProgressiveSlice."""

    def test_all_summaries_combines_types(self):
        s1 = MemorySummary(
            id="h1", memory_type="heuristic", summary="h",
            relevance_hint="", estimated_tokens=5,
        )
        s2 = MemorySummary(
            id="o1", memory_type="outcome", summary="o",
            relevance_hint="", estimated_tokens=5,
        )
        s3 = MemorySummary(
            id="ap1", memory_type="anti_pattern", summary="ap",
            relevance_hint="", estimated_tokens=5,
        )
        pslice = ProgressiveSlice(
            heuristic_summaries=[s1],
            outcome_summaries=[s2],
            anti_pattern_summaries=[s3],
        )
        all_s = pslice.all_summaries
        assert len(all_s) == 3

    def test_all_summaries_empty(self):
        pslice = ProgressiveSlice()
        assert pslice.all_summaries == []

    def test_get_full_item_from_cache(self):
        pslice = ProgressiveSlice()
        pslice._full_items["abc"] = {"data": "test"}
        assert pslice.get_full_item("abc") == {"data": "test"}

    def test_get_full_item_from_summary_attachment(self):
        item = create_test_heuristic(id="h-full")
        summary = MemorySummary(
            id="h-full",
            memory_type="heuristic",
            summary="test",
            relevance_hint="hint",
            estimated_tokens=10,
            _full_item=item,
        )
        pslice = ProgressiveSlice(heuristic_summaries=[summary])
        result = pslice.get_full_item("h-full")
        assert result == item
        # Should also be cached now
        assert "h-full" in pslice._full_items

    def test_get_full_item_not_found(self):
        pslice = ProgressiveSlice()
        assert pslice.get_full_item("nonexistent") is None

    def test_get_ids_by_type_heuristic(self):
        s1 = MemorySummary(
            id="h1", memory_type="heuristic", summary="",
            relevance_hint="", estimated_tokens=0,
        )
        s2 = MemorySummary(
            id="h2", memory_type="heuristic", summary="",
            relevance_hint="", estimated_tokens=0,
        )
        pslice = ProgressiveSlice(heuristic_summaries=[s1, s2])
        ids = pslice.get_ids_by_type("heuristic")
        assert ids == ["h1", "h2"]

    def test_get_ids_by_type_outcome(self):
        s = MemorySummary(
            id="o1", memory_type="outcome", summary="",
            relevance_hint="", estimated_tokens=0,
        )
        pslice = ProgressiveSlice(outcome_summaries=[s])
        assert pslice.get_ids_by_type("outcome") == ["o1"]

    def test_get_ids_by_type_unknown(self):
        pslice = ProgressiveSlice()
        assert pslice.get_ids_by_type("nonexistent") == []


class TestSummaryExtractor:
    """Tests for SummaryExtractor."""

    @pytest.fixture
    def extractor(self):
        return SummaryExtractor(max_summary_length=100, chars_per_token=4)

    # --- Heuristic extraction ---

    def test_heuristic_reference_level(self, extractor):
        h = create_test_heuristic(condition="when testing patterns")
        summary = extractor.extract_heuristic_summary(h, DisclosureLevel.REFERENCE)
        assert summary.disclosure_level == DisclosureLevel.REFERENCE
        assert "Heuristic:" in summary.summary
        assert summary.key_fields == {}
        assert summary._full_item is None

    def test_heuristic_summary_level(self, extractor):
        h = create_test_heuristic(
            condition="API fails", strategy="retry with backoff"
        )
        summary = extractor.extract_heuristic_summary(h, DisclosureLevel.SUMMARY)
        assert "When" in summary.summary
        assert summary.disclosure_level == DisclosureLevel.SUMMARY
        assert "Success rate:" in summary.relevance_hint
        assert summary._full_item is None

    def test_heuristic_key_details_level(self, extractor):
        h = create_test_heuristic(confidence=0.85, occurrence_count=10)
        summary = extractor.extract_heuristic_summary(h, DisclosureLevel.KEY_DETAILS)
        assert "confidence" in summary.key_fields
        assert "success_rate" in summary.key_fields
        assert "occurrence_count" in summary.key_fields
        assert summary._full_item is None

    def test_heuristic_full_level(self, extractor):
        h = create_test_heuristic()
        summary = extractor.extract_heuristic_summary(h, DisclosureLevel.FULL)
        assert summary._full_item is h
        assert "confidence" in summary.key_fields

    # --- Outcome extraction ---

    def test_outcome_reference_level(self, extractor):
        o = create_test_outcome(task_type="api_test", success=True)
        summary = extractor.extract_outcome_summary(o, DisclosureLevel.REFERENCE)
        assert "Outcome:" in summary.summary
        assert "Success" in summary.summary
        assert summary._full_item is None

    def test_outcome_summary_level(self, extractor):
        o = create_test_outcome(
            task_description="validate auth endpoint",
            strategy_used="token refresh",
            success=False,
        )
        summary = extractor.extract_outcome_summary(o, DisclosureLevel.SUMMARY)
        assert "Failed" in summary.summary
        assert summary._full_item is None

    def test_outcome_key_details_level(self, extractor):
        o = create_test_outcome(success=True, task_type="test_task")
        summary = extractor.extract_outcome_summary(o, DisclosureLevel.KEY_DETAILS)
        assert "success" in summary.key_fields
        assert "task_type" in summary.key_fields

    def test_outcome_key_details_with_error(self, extractor):
        o = create_test_outcome(success=False, error_message="Connection refused")
        summary = extractor.extract_outcome_summary(o, DisclosureLevel.KEY_DETAILS)
        assert "error" in summary.key_fields

    def test_outcome_full_level(self, extractor):
        o = create_test_outcome()
        summary = extractor.extract_outcome_summary(o, DisclosureLevel.FULL)
        assert summary._full_item is o

    # --- Knowledge extraction ---

    def test_knowledge_reference_level(self, extractor):
        k = create_test_knowledge(domain="authentication")
        summary = extractor.extract_knowledge_summary(k, DisclosureLevel.REFERENCE)
        assert "Knowledge:" in summary.summary
        assert summary._full_item is None

    def test_knowledge_summary_level(self, extractor):
        k = create_test_knowledge(
            domain="auth", fact="JWT expires in 24h", confidence=0.95
        )
        summary = extractor.extract_knowledge_summary(k, DisclosureLevel.SUMMARY)
        assert "[auth]" in summary.summary
        assert "Confidence:" in summary.relevance_hint

    def test_knowledge_key_details_level(self, extractor):
        k = create_test_knowledge(domain="db", confidence=0.8, source="manual")
        summary = extractor.extract_knowledge_summary(k, DisclosureLevel.KEY_DETAILS)
        assert summary.key_fields["domain"] == "db"
        assert summary.key_fields["confidence"] == 0.8
        assert summary.key_fields["source"] == "manual"

    def test_knowledge_full_level(self, extractor):
        k = create_test_knowledge()
        summary = extractor.extract_knowledge_summary(k, DisclosureLevel.FULL)
        assert summary._full_item is k

    # --- Anti-pattern extraction ---

    def test_anti_pattern_reference_level(self, extractor):
        ap = create_test_anti_pattern(pattern="Using sleep() for waits")
        summary = extractor.extract_anti_pattern_summary(
            ap, DisclosureLevel.REFERENCE
        )
        assert "Warning:" in summary.summary
        assert summary._full_item is None

    def test_anti_pattern_summary_level(self, extractor):
        ap = create_test_anti_pattern(
            pattern="bare except",
            why_bad="hides errors",
            occurrence_count=5,
        )
        summary = extractor.extract_anti_pattern_summary(
            ap, DisclosureLevel.SUMMARY
        )
        assert "Avoid:" in summary.summary
        assert "5x" in summary.relevance_hint

    def test_anti_pattern_key_details_level(self, extractor):
        ap = create_test_anti_pattern(
            pattern="global state",
            occurrence_count=7,
            better_alternative="dependency injection",
        )
        summary = extractor.extract_anti_pattern_summary(
            ap, DisclosureLevel.KEY_DETAILS
        )
        assert summary.key_fields["pattern"] == "global state"
        assert summary.key_fields["occurrence_count"] == 7
        assert summary.key_fields["alternative"] == "dependency injection"

    def test_anti_pattern_full_level(self, extractor):
        ap = create_test_anti_pattern()
        summary = extractor.extract_anti_pattern_summary(ap, DisclosureLevel.FULL)
        assert summary._full_item is ap

    # --- Preference extraction ---

    def test_preference_reference_level(self, extractor):
        p = create_test_preference(category="code_style")
        summary = extractor.extract_preference_summary(p, DisclosureLevel.REFERENCE)
        assert "Preference:" in summary.summary
        assert summary._full_item is None

    def test_preference_summary_level(self, extractor):
        p = create_test_preference(
            category="communication", preference="no emojis"
        )
        summary = extractor.extract_preference_summary(p, DisclosureLevel.SUMMARY)
        assert "[communication]" in summary.summary
        assert "Confidence:" in summary.relevance_hint

    def test_preference_full_level(self, extractor):
        p = create_test_preference()
        summary = extractor.extract_preference_summary(p, DisclosureLevel.FULL)
        assert summary._full_item is p

    # --- Truncation ---

    def test_truncate_short_text(self, extractor):
        result = extractor._truncate("short", 100)
        assert result == "short"

    def test_truncate_long_text(self, extractor):
        long_text = "a" * 200
        result = extractor._truncate(long_text, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_truncate_exact_length(self, extractor):
        text = "a" * 100
        result = extractor._truncate(text, 100)
        assert result == text

    # --- Token estimation ---

    def test_estimated_tokens_positive(self, extractor):
        h = create_test_heuristic()
        summary = extractor.extract_heuristic_summary(h, DisclosureLevel.SUMMARY)
        assert summary.estimated_tokens > 0

    def test_memory_type_set_correctly(self, extractor):
        h = create_test_heuristic()
        o = create_test_outcome()
        k = create_test_knowledge()
        ap = create_test_anti_pattern()
        p = create_test_preference()

        assert extractor.extract_heuristic_summary(h).memory_type == "heuristic"
        assert extractor.extract_outcome_summary(o).memory_type == "outcome"
        assert extractor.extract_knowledge_summary(k).memory_type == "domain_knowledge"
        assert extractor.extract_anti_pattern_summary(ap).memory_type == "anti_pattern"
        assert extractor.extract_preference_summary(p).memory_type == "preference"


class TestProgressiveRetrieval:
    """Tests for ProgressiveRetrieval (non-engine methods)."""

    def test_format_summaries_empty(self):
        """Formatting empty slice should return empty or minimal text."""
        pr = ProgressiveRetrieval(
            retrieval_engine=None, storage=None
        )
        pslice = ProgressiveSlice()
        result = pr.format_summaries_for_context(pslice)
        assert isinstance(result, str)

    def test_format_summaries_with_heuristics(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        s = MemorySummary(
            id="h1",
            memory_type="heuristic",
            summary="When X, do Y",
            relevance_hint="90% success",
            estimated_tokens=10,
        )
        pslice = ProgressiveSlice(
            heuristic_summaries=[s],
            summaries_included=1,
            estimated_summary_tokens=10,
        )
        result = pr.format_summaries_for_context(pslice)
        assert "## Relevant Patterns" in result
        assert "[h1]" in result
        assert "When X, do Y" in result

    def test_format_summaries_with_anti_patterns(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        s = MemorySummary(
            id="ap1",
            memory_type="anti_pattern",
            summary="Avoid: bare except",
            relevance_hint="Seen 3x",
            estimated_tokens=10,
        )
        pslice = ProgressiveSlice(
            anti_pattern_summaries=[s],
            summaries_included=1,
            estimated_summary_tokens=10,
        )
        result = pr.format_summaries_for_context(pslice)
        assert "## Warnings" in result
        assert "Avoid: bare except" in result

    def test_format_summaries_with_outcomes(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        s = MemorySummary(
            id="o1",
            memory_type="outcome",
            summary="Success: deployed API",
            relevance_hint="deploy task",
            estimated_tokens=10,
        )
        pslice = ProgressiveSlice(
            outcome_summaries=[s],
            summaries_included=1,
            estimated_summary_tokens=10,
        )
        result = pr.format_summaries_for_context(pslice)
        assert "## Recent Outcomes" in result

    def test_format_summaries_with_preferences(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        s = MemorySummary(
            id="p1",
            memory_type="preference",
            summary="[style] No emojis",
            relevance_hint="User constraint",
            estimated_tokens=10,
        )
        pslice = ProgressiveSlice(
            preference_summaries=[s],
            summaries_included=1,
            estimated_summary_tokens=10,
        )
        result = pr.format_summaries_for_context(pslice)
        assert "## User Preferences" in result

    def test_format_summaries_with_knowledge(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        s = MemorySummary(
            id="k1",
            memory_type="domain_knowledge",
            summary="[auth] JWT expires in 24h",
            relevance_hint="95% confidence",
            estimated_tokens=10,
        )
        pslice = ProgressiveSlice(
            knowledge_summaries=[s],
            summaries_included=1,
            estimated_summary_tokens=10,
        )
        result = pr.format_summaries_for_context(pslice)
        assert "## Domain Knowledge" in result

    def test_format_summaries_fetch_hint(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        pslice = ProgressiveSlice(
            summaries_included=5,
            estimated_summary_tokens=100,
        )
        result = pr.format_summaries_for_context(pslice, include_fetch_hint=True)
        assert "5 summaries shown" in result
        assert "~100 tokens" in result

    def test_format_summaries_no_fetch_hint(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        pslice = ProgressiveSlice(
            summaries_included=5, estimated_summary_tokens=100,
        )
        result = pr.format_summaries_for_context(pslice, include_fetch_hint=False)
        assert "summaries shown" not in result

    def test_get_full_item_from_cache(self):
        """get_full_item should return cached items without touching storage."""
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        h = create_test_heuristic(id="cached-h")
        pr._item_cache["cached-h"] = h

        result = pr.get_full_item("cached-h", "heuristic")
        assert result is h

    def test_clear_cache(self):
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        pr._item_cache["a"] = "test"
        pr.clear_cache()
        assert pr._item_cache == {}

    def test_format_all_types_combined(self):
        """All section types should appear in correct order."""
        pr = ProgressiveRetrieval(retrieval_engine=None, storage=None)
        pslice = ProgressiveSlice(
            heuristic_summaries=[
                MemorySummary(
                    id="h1", memory_type="heuristic", summary="h",
                    relevance_hint="", estimated_tokens=5,
                )
            ],
            anti_pattern_summaries=[
                MemorySummary(
                    id="ap1", memory_type="anti_pattern", summary="ap",
                    relevance_hint="", estimated_tokens=5,
                )
            ],
            outcome_summaries=[
                MemorySummary(
                    id="o1", memory_type="outcome", summary="o",
                    relevance_hint="", estimated_tokens=5,
                )
            ],
            preference_summaries=[
                MemorySummary(
                    id="p1", memory_type="preference", summary="p",
                    relevance_hint="", estimated_tokens=5,
                )
            ],
            knowledge_summaries=[
                MemorySummary(
                    id="k1", memory_type="domain_knowledge", summary="k",
                    relevance_hint="", estimated_tokens=5,
                )
            ],
            summaries_included=5,
            estimated_summary_tokens=25,
        )
        result = pr.format_summaries_for_context(pslice)

        # Check all sections present
        assert "## Relevant Patterns" in result
        assert "## Warnings" in result
        assert "## Recent Outcomes" in result
        assert "## User Preferences" in result
        assert "## Domain Knowledge" in result

        # Check ordering (patterns before warnings before outcomes etc.)
        patterns_pos = result.index("## Relevant Patterns")
        warnings_pos = result.index("## Warnings")
        outcomes_pos = result.index("## Recent Outcomes")
        assert patterns_pos < warnings_pos < outcomes_pos
