"""
Tests for extraction module top-level exports.

Verifies that AutoLearner and related classes are properly exported from alma package.
"""

import pytest


class TestExtractionExports:
    """Tests for extraction module exports from top-level alma package."""

    def test_autolearner_importable_from_alma(self):
        """AutoLearner should be importable from alma package."""
        from alma import AutoLearner

        assert AutoLearner is not None
        assert hasattr(AutoLearner, "learn_from_conversation")

    def test_fact_extractor_importable_from_alma(self):
        """FactExtractor should be importable from alma package."""
        from alma import FactExtractor

        assert FactExtractor is not None
        # It's an abstract base class
        assert hasattr(FactExtractor, "extract")

    def test_llm_fact_extractor_importable_from_alma(self):
        """LLMFactExtractor should be importable from alma package."""
        from alma import LLMFactExtractor

        assert LLMFactExtractor is not None
        assert hasattr(LLMFactExtractor, "extract")

    def test_rule_based_extractor_importable_from_alma(self):
        """RuleBasedExtractor should be importable from alma package."""
        from alma import RuleBasedExtractor

        assert RuleBasedExtractor is not None
        assert hasattr(RuleBasedExtractor, "extract")

    def test_extracted_fact_importable_from_alma(self):
        """ExtractedFact should be importable from alma package."""
        from alma import ExtractedFact

        assert ExtractedFact is not None
        # It's a dataclass
        assert hasattr(ExtractedFact, "__dataclass_fields__")

    def test_extraction_result_importable_from_alma(self):
        """ExtractionResult should be importable from alma package."""
        from alma import ExtractionResult

        assert ExtractionResult is not None
        assert hasattr(ExtractionResult, "__dataclass_fields__")

    def test_fact_type_importable_from_alma(self):
        """FactType enum should be importable from alma package."""
        from alma import FactType

        assert FactType is not None
        # Verify enum values exist
        assert FactType.HEURISTIC
        assert FactType.ANTI_PATTERN
        assert FactType.PREFERENCE
        assert FactType.DOMAIN_KNOWLEDGE
        assert FactType.OUTCOME

    def test_create_extractor_importable_from_alma(self):
        """create_extractor factory should be importable from alma package."""
        from alma import create_extractor

        assert create_extractor is not None
        assert callable(create_extractor)

    def test_add_auto_learning_to_alma_importable(self):
        """add_auto_learning_to_alma helper should be importable from alma package."""
        from alma import add_auto_learning_to_alma

        assert add_auto_learning_to_alma is not None
        assert callable(add_auto_learning_to_alma)

    def test_all_extraction_exports_in_alma_all(self):
        """All extraction exports should be listed in alma.__all__."""
        import alma

        expected_exports = [
            "AutoLearner",
            "FactExtractor",
            "LLMFactExtractor",
            "RuleBasedExtractor",
            "ExtractedFact",
            "ExtractionResult",
            "FactType",
            "create_extractor",
            "add_auto_learning_to_alma",
        ]

        for export in expected_exports:
            assert export in alma.__all__, f"{export} not found in alma.__all__"

    def test_batch_import_all_extraction_classes(self):
        """Should be able to import all extraction classes in one statement."""
        from alma import (
            AutoLearner,
            ExtractedFact,
            ExtractionResult,
            FactExtractor,
            FactType,
            LLMFactExtractor,
            RuleBasedExtractor,
            add_auto_learning_to_alma,
            create_extractor,
        )

        # All imports should be successful
        assert AutoLearner is not None
        assert ExtractedFact is not None
        assert ExtractionResult is not None
        assert FactExtractor is not None
        assert FactType is not None
        assert LLMFactExtractor is not None
        assert RuleBasedExtractor is not None
        assert add_auto_learning_to_alma is not None
        assert create_extractor is not None


class TestExtractionFunctionality:
    """Tests for basic functionality of extraction classes."""

    def test_create_extractor_returns_rule_based_by_default(self):
        """create_extractor should return RuleBasedExtractor when no API keys."""
        import os

        from alma import RuleBasedExtractor, create_extractor

        # Temporarily clear any API keys
        openai_key = os.environ.pop("OPENAI_API_KEY", None)
        anthropic_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            extractor = create_extractor()
            assert isinstance(extractor, RuleBasedExtractor)
        finally:
            # Restore keys if they existed
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    def test_create_extractor_rule_based_explicit(self):
        """create_extractor with provider='rule-based' returns RuleBasedExtractor."""
        from alma import RuleBasedExtractor, create_extractor

        extractor = create_extractor(provider="rule-based")
        assert isinstance(extractor, RuleBasedExtractor)

    def test_rule_based_extractor_extract_basic(self):
        """RuleBasedExtractor should extract facts from conversations."""
        from alma import ExtractionResult, RuleBasedExtractor

        extractor = RuleBasedExtractor()
        messages = [
            {"role": "user", "content": "How do I fix this bug?"},
            {
                "role": "assistant",
                "content": "I fixed it by using a try-catch block. It worked well.",
            },
        ]

        result = extractor.extract(messages)

        assert isinstance(result, ExtractionResult)
        assert isinstance(result.facts, list)
        assert result.tokens_used == 0  # Rule-based doesn't use tokens
        assert result.extraction_time_ms >= 0

    def test_extracted_fact_dataclass_creation(self):
        """ExtractedFact dataclass should be creatable."""
        from alma import ExtractedFact, FactType

        fact = ExtractedFact(
            fact_type=FactType.HEURISTIC,
            content="Use incremental validation for forms",
            confidence=0.85,
            source_text="The assistant mentioned using incremental validation",
            condition="When testing forms",
            strategy="Validate each field as user types",
        )

        assert fact.fact_type == FactType.HEURISTIC
        assert fact.confidence == 0.85
        assert fact.condition == "When testing forms"

    def test_extraction_result_dataclass_creation(self):
        """ExtractionResult dataclass should be creatable."""
        from alma import ExtractedFact, ExtractionResult, FactType

        fact = ExtractedFact(
            fact_type=FactType.DOMAIN_KNOWLEDGE,
            content="React uses a virtual DOM",
            confidence=0.9,
            source_text="User mentioned React's virtual DOM",
            domain="frontend",
        )

        result = ExtractionResult(
            facts=[fact],
            raw_response="test response",
            tokens_used=100,
            extraction_time_ms=50,
        )

        assert len(result.facts) == 1
        assert result.tokens_used == 100
        assert result.extraction_time_ms == 50

    def test_fact_type_enum_values(self):
        """FactType enum should have correct string values."""
        from alma import FactType

        assert FactType.HEURISTIC.value == "heuristic"
        assert FactType.ANTI_PATTERN.value == "anti_pattern"
        assert FactType.PREFERENCE.value == "preference"
        assert FactType.DOMAIN_KNOWLEDGE.value == "domain_knowledge"
        assert FactType.OUTCOME.value == "outcome"
