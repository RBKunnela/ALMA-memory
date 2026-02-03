"""
Unit tests for the Memory Compression Pipeline.

Tests CompressionLevel, CompressedMemory, CompressionConfig,
MemoryCompressor, and all compression methods.
"""

from unittest.mock import MagicMock

from alma.compression import (
    CompressedMemory,
    CompressionConfig,
    CompressionLevel,
    CompressionResult,
    MemoryCompressor,
    create_compressor,
)


class TestCompressionLevel:
    """Tests for CompressionLevel enum."""

    def test_none_level(self):
        """Should have none level."""
        assert CompressionLevel.NONE.value == "none"

    def test_light_level(self):
        """Should have light level."""
        assert CompressionLevel.LIGHT.value == "light"

    def test_medium_level(self):
        """Should have medium level."""
        assert CompressionLevel.MEDIUM.value == "medium"

    def test_aggressive_level(self):
        """Should have aggressive level."""
        assert CompressionLevel.AGGRESSIVE.value == "aggressive"


class TestCompressedMemory:
    """Tests for CompressedMemory dataclass."""

    def test_create_compressed_memory(self):
        """Should create compressed memory with all fields."""
        cm = CompressedMemory(
            original_length=1000,
            compressed_length=300,
            compression_ratio=3.33,
            key_facts=["Fact 1", "Fact 2"],
            constraints=["Constraint 1"],
            patterns=["Pattern 1"],
            summary="Compressed summary",
            full_content="Original long content...",
        )

        assert cm.original_length == 1000
        assert cm.compressed_length == 300
        assert cm.compression_ratio == 3.33
        assert len(cm.key_facts) == 2
        assert len(cm.constraints) == 1
        assert len(cm.patterns) == 1
        assert cm.summary == "Compressed summary"

    def test_to_metadata(self):
        """Should generate metadata dictionary."""
        cm = CompressedMemory(
            original_length=500,
            compressed_length=100,
            compression_ratio=5.0,
            key_facts=["Important fact"],
            constraints=["Must do X"],
            patterns=["When Y then Z"],
            summary="Short summary",
            full_content="Long content",
        )

        metadata = cm.to_metadata()

        assert metadata["compressed"] is True
        assert metadata["compression_ratio"] == 5.0
        assert metadata["original_length"] == 500
        assert metadata["compressed_length"] == 100
        assert "Important fact" in metadata["key_facts"]
        assert "Must do X" in metadata["constraints"]

    def test_to_dict(self):
        """Should convert to full dictionary."""
        cm = CompressedMemory(
            original_length=200,
            compressed_length=50,
            compression_ratio=4.0,
            key_facts=["Fact"],
            constraints=[],
            patterns=[],
            summary="Summary",
            full_content="Full content here",
        )

        d = cm.to_dict()

        assert d["original_length"] == 200
        assert d["compressed_length"] == 50
        assert d["summary"] == "Summary"
        assert d["full_content"] == "Full content here"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "original_length": 300,
            "compressed_length": 100,
            "compression_ratio": 3.0,
            "key_facts": ["Fact 1"],
            "constraints": ["Constraint"],
            "patterns": [],
            "summary": "Test summary",
            "full_content": "Test content",
        }

        cm = CompressedMemory.from_dict(data)

        assert cm.original_length == 300
        assert cm.compression_ratio == 3.0
        assert cm.key_facts == ["Fact 1"]


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_create_result(self):
        """Should create compression result."""
        cm = CompressedMemory(
            original_length=100,
            compressed_length=50,
            compression_ratio=2.0,
            key_facts=[],
            constraints=[],
            patterns=[],
            summary="Summary",
            full_content="Content",
        )

        result = CompressionResult(
            compressed=cm,
            level=CompressionLevel.MEDIUM,
            method="llm",
            compression_time_ms=150,
        )

        assert result.compressed == cm
        assert result.level == CompressionLevel.MEDIUM
        assert result.method == "llm"
        assert result.compression_time_ms == 150
        assert result.success is True
        assert result.error is None

    def test_failed_result(self):
        """Should handle failed compression."""
        result = CompressionResult(
            compressed=None,
            level=CompressionLevel.MEDIUM,
            method="error",
            success=False,
            error="LLM timeout",
        )

        assert result.compressed is None
        assert result.success is False
        assert result.error == "LLM timeout"

    def test_to_dict(self):
        """Should convert to dictionary."""
        cm = CompressedMemory(
            original_length=100,
            compressed_length=50,
            compression_ratio=2.0,
            key_facts=[],
            constraints=[],
            patterns=[],
            summary="Summary",
            full_content="Content",
        )

        result = CompressionResult(
            compressed=cm,
            level=CompressionLevel.LIGHT,
            method="rule_based",
        )

        d = result.to_dict()

        assert d["level"] == "light"
        assert d["method"] == "rule_based"
        assert d["compressed"] is not None


class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = CompressionConfig()

        assert config.default_level == CompressionLevel.MEDIUM
        assert config.min_length_for_compression == 200
        assert config.max_key_facts == 5
        assert config.max_constraints == 3
        assert config.preserve_full_content is True

    def test_custom_config(self):
        """Should accept custom values."""
        config = CompressionConfig(
            default_level=CompressionLevel.AGGRESSIVE,
            min_length_for_compression=100,
            max_key_facts=3,
        )

        assert config.default_level == CompressionLevel.AGGRESSIVE
        assert config.min_length_for_compression == 100
        assert config.max_key_facts == 3

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "default_level": "light",
            "min_length_for_compression": 50,
            "max_key_facts": 10,
        }

        config = CompressionConfig.from_dict(data)

        assert config.default_level == CompressionLevel.LIGHT
        assert config.min_length_for_compression == 50
        assert config.max_key_facts == 10

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = CompressionConfig(default_level=CompressionLevel.AGGRESSIVE)

        d = config.to_dict()

        assert d["default_level"] == "aggressive"
        assert "min_length_for_compression" in d


class TestMemoryCompressorBasic:
    """Basic tests for MemoryCompressor."""

    def test_create_compressor(self):
        """Should create compressor without LLM."""
        compressor = MemoryCompressor()

        assert compressor.llm is None
        assert compressor.config is not None

    def test_create_with_llm(self):
        """Should create compressor with LLM."""
        llm = MagicMock()
        compressor = MemoryCompressor(llm_client=llm)

        assert compressor.llm == llm

    def test_create_with_config(self):
        """Should create with custom config."""
        config = CompressionConfig(default_level=CompressionLevel.LIGHT)
        compressor = MemoryCompressor(config=config)

        assert compressor.config.default_level == CompressionLevel.LIGHT


class TestNoCompression:
    """Tests for NONE compression level."""

    def test_no_compression_preserves_content(self):
        """Should preserve original content."""
        compressor = MemoryCompressor()
        content = "This is the original content that should not be changed."

        result = compressor.compress(content, CompressionLevel.NONE)

        assert result.compressed is not None
        assert result.compressed.summary == content
        assert result.compressed.compression_ratio == 1.0

    def test_short_content_skips_compression(self):
        """Should skip compression for short content."""
        config = CompressionConfig(min_length_for_compression=100)
        compressor = MemoryCompressor(config=config)
        content = "Short content."

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.level == CompressionLevel.NONE
        assert result.method == "skip"


class TestRuleBasedCompression:
    """Tests for rule-based compression fallback."""

    def test_light_compression_removes_duplicates(self):
        """Light compression should remove duplicate sentences."""
        config = CompressionConfig(min_length_for_compression=50)
        compressor = MemoryCompressor(config=config)
        content = (
            "This is a test sentence that will be duplicated. "
            "This is a test sentence that will be duplicated. "
            "This is different content. This is another unique sentence."
        )

        result = compressor.compress(content, CompressionLevel.LIGHT)

        assert result.compressed is not None
        assert result.method == "rule_based"
        # Should have fewer occurrences of duplicate
        assert result.compressed.summary.count("This is a test sentence") == 1

    def test_medium_compression_extracts_key_sentences(self):
        """Medium compression should extract key sentences."""
        config = CompressionConfig(min_length_for_compression=50)
        compressor = MemoryCompressor(config=config)
        content = (
            "The project started in January. "
            "Many meetings were held. "
            "The key requirement is performance. "
            "Coffee was served. "
            "The system must handle 1000 requests per second. "
            "The weather was nice. "
            "The conclusion is that we need more testing."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert result.method == "rule_based"
        # Should have compression
        assert result.compressed.compression_ratio > 1.0
        # Should keep first sentence (boosted)
        assert "January" in result.compressed.summary
        # Constraint with "must" should be extracted to constraints
        assert len(result.compressed.constraints) > 0
        assert any("must" in c.lower() for c in result.compressed.constraints)

    def test_aggressive_compression_minimal_content(self):
        """Aggressive compression should be maximal."""
        compressor = MemoryCompressor()
        content = (
            "The project started in January. "
            "There were many stakeholders involved. "
            "Requirements gathering took several weeks. "
            "The key outcome is the new architecture. "
            "Implementation will begin next month. "
            "The team is ready to proceed."
        )

        result = compressor.compress(content, CompressionLevel.AGGRESSIVE)

        assert result.compressed is not None
        # Should have high compression
        assert result.compressed.compression_ratio > 1.5
        # Should keep first sentence (context)
        assert "January" in result.compressed.summary

    def test_extracts_key_facts(self):
        """Should extract sentences that appear to be facts."""
        config = CompressionConfig(min_length_for_compression=50)
        compressor = MemoryCompressor(config=config)
        content = (
            "The system is written in Python. "
            "Users have reported issues. "
            "The database is PostgreSQL. "
            "We discussed options for the architecture. "
            "The API has 50 endpoints. "
            "This is additional filler content to make the text long enough for compression."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert result.method == "rule_based"
        # Should extract facts with "is" indicator
        facts = result.compressed.key_facts
        assert len(facts) > 0
        # At least one fact should mention something concrete
        all_facts = " ".join(facts).lower()
        assert "python" in all_facts or "postgresql" in all_facts or "api" in all_facts

    def test_extracts_constraints(self):
        """Should extract sentences that describe constraints."""
        config = CompressionConfig(min_length_for_compression=50)
        compressor = MemoryCompressor(config=config)
        content = (
            "The system is flexible and extensible. "
            "You must authenticate before accessing data. "
            "The API cannot handle files larger than 10MB. "
            "The team is distributed across multiple locations. "
            "Users should not share credentials with others. "
            "This is additional content to make text longer."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert result.method == "rule_based"
        constraints = result.compressed.constraints
        # Should extract sentences with must/cannot/should not
        assert len(constraints) > 0


class TestLLMBasedCompression:
    """Tests for LLM-based compression."""

    def test_llm_compression_called(self):
        """Should call LLM for compression."""
        llm = MagicMock()
        llm.complete.return_value = """SUMMARY: Key outcome achieved.
KEY_FACTS:
- Fact one
- Fact two
CONSTRAINTS:
- None
PATTERNS:
- Pattern found"""

        compressor = MemoryCompressor(llm_client=llm)
        content = "A" * 300  # Long enough to compress

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert llm.complete.called
        assert result.method == "llm"
        assert result.compressed is not None
        assert "Key outcome" in result.compressed.summary

    def test_llm_response_parsing(self):
        """Should parse LLM response correctly."""
        llm = MagicMock()
        llm.complete.return_value = """SUMMARY: The project was successful.
KEY_FACTS:
- Deployed on time
- Under budget
- High user satisfaction
CONSTRAINTS:
- Must maintain backwards compatibility
PATTERNS:
- When deploying, always run tests first"""

        compressor = MemoryCompressor(llm_client=llm)
        content = "A" * 300

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert len(result.compressed.key_facts) == 3
        assert len(result.compressed.constraints) == 1
        assert len(result.compressed.patterns) == 1

    def test_llm_failure_falls_back_to_rule_based(self):
        """Should fall back to rule-based on LLM failure."""
        llm = MagicMock()
        llm.complete.side_effect = Exception("LLM error")

        compressor = MemoryCompressor(llm_client=llm)
        # Content must be long enough to trigger compression (>= 200 chars)
        content = (
            "This is a test sentence with more content to make it long enough. "
            "The system is reliable and well-tested. It must work properly. "
            "We have verified this through extensive testing and validation. "
            "The team is confident in the implementation quality."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.success is True
        assert result.method == "rule_based"
        assert result.compressed is not None


class TestCompressOutcome:
    """Tests for compress_outcome method."""

    def test_compress_outcome_basic(self):
        """Should compress task outcome."""
        compressor = MemoryCompressor()
        outcome = (
            "The task was completed successfully. "
            "The implementation required careful consideration of edge cases. "
            "The main challenge was handling concurrent access. "
            "We used a mutex to solve this. "
            "Performance tests showed 100ms response time."
        )

        compressed = compressor.compress_outcome(outcome, CompressionLevel.MEDIUM)

        assert compressed is not None
        assert compressed.original_length > compressed.compressed_length

    def test_compress_outcome_with_llm(self):
        """Should use LLM for outcome compression."""
        llm = MagicMock()
        llm.complete.return_value = """SUMMARY: Task completed with mutex solution.
KEY_FACTS:
- Used mutex for concurrency
- 100ms response time
CONSTRAINTS:
- None
PATTERNS:
- Use mutex for concurrent access"""

        compressor = MemoryCompressor(llm_client=llm)
        outcome = "A" * 300

        compressed = compressor.compress_outcome(outcome)

        assert "mutex" in compressed.summary.lower()


class TestCompressConversation:
    """Tests for compress_conversation method."""

    def test_compress_conversation_basic(self):
        """Should extract knowledge from conversation."""
        compressor = MemoryCompressor()
        conversation = (
            "User: How do I deploy this?\n"
            "Agent: First, run the build script.\n"
            "User: What about the database?\n"
            "Agent: The database migration must be run first.\n"
            "User: Thanks!\n"
            "Agent: You're welcome."
        )

        compressed = compressor.compress_conversation(conversation)

        assert compressed is not None
        assert compressed.compression_ratio >= 1.0

    def test_compress_conversation_with_focus(self):
        """Should focus on specified topic."""
        llm = MagicMock()
        llm.complete.return_value = """SUMMARY: Database setup requires migration.
KEY_FACTS:
- Run migration first
CONSTRAINTS:
- Database must be accessible
PATTERNS:
- None"""

        compressor = MemoryCompressor(llm_client=llm)
        conversation = "A" * 300

        compressor.compress_conversation(conversation, focus="database setup")

        # Verify focus was included in prompt
        call_args = llm.complete.call_args[0][0]
        assert "database setup" in call_args.lower()


class TestExtractHeuristic:
    """Tests for extract_heuristic method."""

    def test_returns_none_if_too_few_experiences(self):
        """Should return None with fewer than min experiences."""
        compressor = MemoryCompressor()
        experiences = ["Experience 1", "Experience 2"]

        result = compressor.extract_heuristic(experiences, min_experiences=3)

        assert result is None

    def test_extracts_heuristic_with_llm(self):
        """Should extract heuristic using LLM."""
        llm = MagicMock()
        llm.complete.return_value = (
            "When deploying to production, then run all tests first "
            "because untested code causes incidents."
        )

        compressor = MemoryCompressor(llm_client=llm)
        experiences = [
            "Deployed without tests, caused incident",
            "Ran tests before deploy, successful",
            "Skipped tests for quick fix, broke production",
            "Full test suite before deploy, no issues",
        ]

        result = compressor.extract_heuristic(experiences)

        assert result is not None
        assert "test" in result.lower()

    def test_returns_none_for_no_pattern(self):
        """Should return None when no pattern found."""
        llm = MagicMock()
        llm.complete.return_value = "NO_PATTERN"

        compressor = MemoryCompressor(llm_client=llm)
        experiences = ["Random event 1", "Different thing 2", "Unrelated 3"]

        result = compressor.extract_heuristic(experiences)

        assert result is None

    def test_rule_based_heuristic_fallback(self):
        """Should attempt rule-based extraction without LLM."""
        compressor = MemoryCompressor()
        experiences = [
            "Used caching to improve performance",
            "Added caching layer for speed",
            "Implemented cache for faster response",
            "Cache helped reduce latency",
        ]

        result = compressor.extract_heuristic(experiences)

        # May or may not find pattern, but should not error
        # If pattern found, should mention common word
        if result:
            assert "cach" in result.lower() or "experience" in result.lower()


class TestDeduplicateKnowledge:
    """Tests for deduplicate_knowledge method."""

    def test_returns_new_if_no_existing(self):
        """Should return new knowledge if no existing."""
        compressor = MemoryCompressor()
        new_knowledge = "The API uses REST conventions."

        result = compressor.deduplicate_knowledge(new_knowledge, [])

        assert result == new_knowledge

    def test_returns_new_if_no_similar(self):
        """Should return new if nothing similar exists."""
        compressor = MemoryCompressor()
        new_knowledge = "The API uses REST conventions."
        existing = ["The database is PostgreSQL.", "Authentication uses OAuth."]

        result = compressor.deduplicate_knowledge(new_knowledge, existing)

        assert result == new_knowledge

    def test_detects_duplicate_without_llm(self):
        """Should detect obvious duplicate without LLM."""
        compressor = MemoryCompressor()
        new_knowledge = "The API uses REST conventions."
        existing = ["The API uses REST conventions.", "Other knowledge."]

        result = compressor.deduplicate_knowledge(new_knowledge, existing)

        # Without LLM, exact duplicates detected by word overlap
        assert result is None

    def test_merges_with_llm(self):
        """Should merge overlapping knowledge with LLM."""
        llm = MagicMock()
        llm.complete.return_value = (
            "The API uses REST conventions with JSON responses and "
            "supports pagination."
        )

        compressor = MemoryCompressor(llm_client=llm)
        new_knowledge = "The API supports pagination."
        existing = ["The API uses REST conventions with JSON responses."]

        result = compressor.deduplicate_knowledge(new_knowledge, existing)

        assert result is not None
        assert "pagination" in result.lower()

    def test_detects_duplicate_with_llm(self):
        """Should detect semantic duplicate with LLM."""
        llm = MagicMock()
        llm.complete.return_value = "DUPLICATE"

        compressor = MemoryCompressor(llm_client=llm)
        # Use content with enough word overlap to pass similarity threshold (0.3)
        new_knowledge = "The API uses REST conventions for data access."
        existing = ["The API uses REST conventions with JSON responses."]

        result = compressor.deduplicate_knowledge(new_knowledge, existing)

        assert result is None


class TestBatchCompression:
    """Tests for batch_compress method."""

    def test_compress_multiple_items(self):
        """Should compress multiple content items."""
        compressor = MemoryCompressor()
        contents = [
            "First item with some content that is long enough to compress. " * 5,
            "Second item with different content that is also long enough. " * 5,
            "Third item with yet more content for compression testing. " * 5,
        ]

        results = compressor.batch_compress(contents, CompressionLevel.MEDIUM)

        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert result.compressed is not None


class TestCompressionRatios:
    """Tests for compression ratio targets."""

    def test_light_compression_ratio(self):
        """Light compression should achieve ~1.5x ratio."""
        compressor = MemoryCompressor()
        # Create content with some redundancy
        content = (
            "This is a repeated sentence. " * 10 +
            "This is unique content. " +
            "More unique content here."
        )

        result = compressor.compress(content, CompressionLevel.LIGHT)

        assert result.compressed is not None
        # Light compression removes duplicates
        assert result.compressed.compression_ratio >= 1.2

    def test_medium_compression_ratio(self):
        """Medium compression should achieve ~3x ratio."""
        compressor = MemoryCompressor()
        # Create verbose content
        content = (
            "The project began with extensive planning. "
            "We held many meetings to discuss requirements. "
            "The key requirement is high availability. "
            "Stakeholders were consulted throughout. "
            "The system must handle 10000 concurrent users. "
            "Various options were evaluated. "
            "The final decision was to use microservices. "
            "Implementation started in Q2. "
            "Testing took several weeks. "
            "The result is a robust system."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        # Medium should achieve at least 1.5x, ideally 3x
        assert result.compressed.compression_ratio >= 1.5

    def test_aggressive_compression_ratio(self):
        """Aggressive compression should achieve ~5x ratio."""
        compressor = MemoryCompressor()
        # Create very verbose content
        content = " ".join([
            f"Sentence number {i} with some filler content and details. "
            f"This adds more words to the sentence {i}."
            for i in range(20)
        ])

        result = compressor.compress(content, CompressionLevel.AGGRESSIVE)

        assert result.compressed is not None
        # Aggressive should achieve at least 2x, ideally 5x
        assert result.compressed.compression_ratio >= 2.0


class TestCreateCompressor:
    """Tests for factory function."""

    def test_create_without_args(self):
        """Should create compressor without arguments."""
        compressor = create_compressor()

        assert compressor is not None
        assert compressor.llm is None

    def test_create_with_llm(self):
        """Should create with LLM client."""
        llm = MagicMock()
        compressor = create_compressor(llm_client=llm)

        assert compressor.llm == llm

    def test_create_with_config(self):
        """Should create with config."""
        config = CompressionConfig(default_level=CompressionLevel.AGGRESSIVE)
        compressor = create_compressor(config=config)

        assert compressor.config.default_level == CompressionLevel.AGGRESSIVE


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_content(self):
        """Should handle empty content."""
        compressor = MemoryCompressor()

        result = compressor.compress("", CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert result.compressed.summary == ""

    def test_single_sentence(self):
        """Should handle single sentence."""
        compressor = MemoryCompressor()
        content = "This is a single sentence with enough length to process."

        result = compressor.compress(content, CompressionLevel.AGGRESSIVE)

        assert result.compressed is not None
        # Should not over-compress single sentence
        assert len(result.compressed.summary) > 0

    def test_very_long_content(self):
        """Should handle very long content."""
        compressor = MemoryCompressor()
        content = "This is a test sentence. " * 1000  # Very long

        result = compressor.compress(content, CompressionLevel.AGGRESSIVE)

        assert result.compressed is not None
        assert result.compressed.compression_ratio > 1.0

    def test_special_characters(self):
        """Should handle special characters."""
        compressor = MemoryCompressor()
        content = (
            "The API endpoint is https://api.example.com/v1/users. "
            "Use the header: X-API-Key: abc123. "
            "Response format: {'data': [...]}. "
            "Rate limit: 100 req/min."
        )

        result = compressor.compress(content, CompressionLevel.MEDIUM)

        assert result.compressed is not None
        assert result.success is True

    def test_unicode_content(self):
        """Should handle unicode content."""
        compressor = MemoryCompressor()
        content = (
            "日本語のテキストです。This is mixed content. "
            "Ещё немного текста на русском. "
            "The system handles multiple languages."
        )

        result = compressor.compress(content, CompressionLevel.LIGHT)

        assert result.compressed is not None
        assert result.success is True
