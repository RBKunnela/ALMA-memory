"""
Integration tests for Memory Compression Pipeline.

Tests compression with real storage and full learning flow.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from alma.compression.pipeline import (
    CompressionLevel,
    MemoryCompressor,
)
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import Heuristic, Outcome


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response or """SUMMARY: Task completed successfully with good results.
KEY_FACTS:
- Implementation was straightforward
- Performance meets requirements
CONSTRAINTS:
- Must maintain backwards compatibility
PATTERNS:
- Test before deploying"""
        self.call_count = 0
        self.last_prompt = None

    def complete(self, prompt: str, timeout: Optional[float] = None) -> str:
        """Return mock response."""
        self.call_count += 1
        self.last_prompt = prompt
        return self.default_response


class TestCompressionWithStorage:
    """Integration tests for compression with storage."""

    @pytest.fixture
    def storage(self):
        """Create a temporary SQLite storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_compression.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_compress_and_store_outcome(self, storage):
        """Should compress outcome and store with metadata."""
        compressor = MemoryCompressor()
        verbose_outcome = (
            "The task involved implementing a new feature for user authentication. "
            "We started by analyzing the existing codebase and identified several "
            "areas that needed modification. The main challenge was ensuring backwards "
            "compatibility with the existing session management. We implemented a new "
            "token-based system that works alongside the old cookie-based approach. "
            "Testing revealed some edge cases that needed handling. The final solution "
            "includes rate limiting to prevent abuse. Performance testing showed response "
            "times under 100ms for authentication requests."
        )

        # Compress
        compressed = compressor.compress_outcome(verbose_outcome, CompressionLevel.MEDIUM)

        # Store as outcome with compression metadata
        now = datetime.now(timezone.utc)
        metadata = compressed.to_metadata()

        outcome = Outcome(
            id="out-compressed-1",
            agent="test-agent",
            project_id="test-project",
            task_type="feature",
            task_description=compressed.summary,  # Use compressed summary
            success=True,
            strategy_used="token-based auth",
            duration_ms=5000,
            timestamp=now,
            metadata=metadata,
        )

        storage.save_outcome(outcome)

        # Retrieve and verify
        outcomes = storage.get_outcomes(project_id="test-project")
        assert len(outcomes) == 1
        retrieved = outcomes[0]

        # Verify compression metadata preserved
        assert retrieved.metadata["compressed"] is True
        assert retrieved.metadata["compression_ratio"] > 1.0
        assert len(retrieved.metadata["key_facts"]) > 0

    def test_compress_and_store_heuristic(self, storage):
        """Should extract and store heuristic from compressed content."""
        llm = MockLLMClient(
            default_response="""SUMMARY: When implementing auth, use tokens with rate limiting.
KEY_FACTS:
- Token-based auth is preferred
- Rate limiting prevents abuse
CONSTRAINTS:
- Must support legacy sessions
PATTERNS:
- Always add rate limiting to auth endpoints"""
        )
        compressor = MemoryCompressor(llm_client=llm)

        content = (
            "Implemented authentication system. Used JWT tokens for security. "
            "Added rate limiting at 100 requests per minute. "
            "Legacy cookie sessions still supported for backwards compatibility."
        )

        compressed = compressor.compress_outcome(content, CompressionLevel.MEDIUM)

        # Create heuristic from pattern
        if compressed.patterns:
            now = datetime.now(timezone.utc)
            heuristic = Heuristic(
                id="heur-from-compression",
                agent="test-agent",
                project_id="test-project",
                condition="implementing authentication",
                strategy=compressed.patterns[0],
                confidence=0.7,
                occurrence_count=1,
                success_count=1,
                last_validated=now,
                created_at=now,
                metadata=compressed.to_metadata(),
            )

            storage.save_heuristic(heuristic)

            # Retrieve and verify
            heuristics = storage.get_heuristics(project_id="test-project")
            assert len(heuristics) == 1
            assert "rate limiting" in heuristics[0].strategy.lower()


class TestHeuristicExtraction:
    """Integration tests for heuristic extraction from multiple experiences."""

    def test_extract_heuristic_from_outcomes(self):
        """Should extract heuristic from multiple similar outcomes."""
        llm = MockLLMClient(
            default_response=(
                "When deploying database changes, then run migrations before "
                "code deployment because database schema must match code expectations."
            )
        )
        compressor = MemoryCompressor(llm_client=llm)

        experiences = [
            "Deployed code first, migration failed, had to rollback",
            "Ran migration before deploy, everything worked smoothly",
            "Tried simultaneous deploy and migrate, caused race condition",
            "Migration-first approach successful again",
            "Forgot migration, app crashed on new column access",
        ]

        heuristic = compressor.extract_heuristic(experiences)

        assert heuristic is not None
        assert "migration" in heuristic.lower()

    def test_no_heuristic_from_dissimilar(self):
        """Should not extract heuristic from dissimilar experiences."""
        llm = MockLLMClient(default_response="NO_PATTERN")
        compressor = MemoryCompressor(llm_client=llm)

        experiences = [
            "Fixed CSS styling issue",
            "Optimized database query",
            "Added new API endpoint",
            "Updated documentation",
        ]

        heuristic = compressor.extract_heuristic(experiences)

        assert heuristic is None


class TestDeduplicationWithStorage:
    """Integration tests for knowledge deduplication."""

    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_dedup.db"
            storage = SQLiteStorage(db_path=db_path)
            yield storage

    def test_deduplicate_before_storing(self, storage):
        """Should deduplicate knowledge before storing."""
        llm = MockLLMClient()
        compressor = MemoryCompressor(llm_client=llm)

        # Store some existing knowledge
        from alma.types import DomainKnowledge

        now = datetime.now(timezone.utc)
        existing = DomainKnowledge(
            id="dk-existing",
            agent="test-agent",
            project_id="test-project",
            domain="api",
            fact="The API uses REST conventions with JSON responses",
            source="documentation",
            confidence=0.9,
            last_verified=now,
        )
        storage.save_domain_knowledge(existing)

        # Try to add similar knowledge
        existing_facts = [existing.fact]
        new_fact = "REST API returns JSON data"

        # Set up LLM to detect duplicate
        llm.default_response = "DUPLICATE"
        result = compressor.deduplicate_knowledge(new_fact, existing_facts)

        assert result is None  # Should be detected as duplicate

    def test_merge_overlapping_knowledge(self, storage):
        """Should merge overlapping knowledge."""
        llm = MockLLMClient(
            default_response=(
                "The API uses REST conventions with JSON responses "
                "and supports both authentication methods."
            )
        )
        compressor = MemoryCompressor(llm_client=llm)

        existing = ["The API uses REST conventions with JSON responses"]
        new_knowledge = "The API supports both OAuth and API key authentication"

        result = compressor.deduplicate_knowledge(new_knowledge, existing)

        assert result is not None
        assert "authentication" in result.lower()


class TestCompressionLevelsIntegration:
    """Integration tests for different compression levels."""

    def test_compression_level_impact(self):
        """Different levels should produce different results."""
        compressor = MemoryCompressor()

        verbose_content = (
            "The quarterly report meeting was held on Tuesday afternoon. "
            "All department heads were present. Sarah presented the sales figures. "
            "Revenue increased by 15% compared to last quarter. "
            "The main driver was the new product line launched in September. "
            "Marketing expenses were slightly over budget. "
            "John explained the reasons for the overspend. "
            "It was due to the additional campaign for the product launch. "
            "The board approved the next quarter's budget. "
            "Action items were assigned to team leads. "
            "Next meeting scheduled for January 15th."
        )

        light = compressor.compress(verbose_content, CompressionLevel.LIGHT)
        medium = compressor.compress(verbose_content, CompressionLevel.MEDIUM)
        aggressive = compressor.compress(verbose_content, CompressionLevel.AGGRESSIVE)

        # Light should be longest, aggressive should be shortest
        assert light.compressed is not None
        assert medium.compressed is not None
        assert aggressive.compressed is not None

        assert light.compressed.compressed_length >= medium.compressed.compressed_length
        assert medium.compressed.compressed_length >= aggressive.compressed.compressed_length

        # Compression ratios should increase
        assert light.compressed.compression_ratio <= medium.compressed.compression_ratio
        assert medium.compressed.compression_ratio <= aggressive.compressed.compression_ratio


class TestCompressionWithRealContent:
    """Tests with realistic content scenarios."""

    def test_compress_code_review_notes(self):
        """Should compress code review feedback."""
        compressor = MemoryCompressor()

        review_notes = (
            "Code review for PR #1234: Authentication refactoring. "
            "Overall the changes look good. A few suggestions: "
            "1. The token validation could be extracted to a separate function. "
            "2. Consider adding more detailed logging for failed auth attempts. "
            "3. The rate limiting logic should use a sliding window. "
            "4. Good job on the test coverage, but add edge case tests. "
            "5. Documentation is clear and helpful. "
            "Approved with minor changes requested. "
            "Follow-up meeting scheduled for Thursday."
        )

        compressed = compressor.compress_outcome(review_notes, CompressionLevel.MEDIUM)

        assert compressed is not None
        assert compressed.compression_ratio > 1.0
        # Should preserve key actionable items
        summary_lower = compressed.summary.lower()
        # At least some technical terms should be preserved
        has_technical = any(
            term in summary_lower
            for term in ["token", "logging", "rate", "test", "auth"]
        )
        assert has_technical or len(compressed.key_facts) > 0

    def test_compress_incident_report(self):
        """Should compress incident report preserving key details."""
        compressor = MemoryCompressor()

        incident_report = (
            "Incident Report - Production Outage on 2024-01-15. "
            "Duration: 45 minutes (14:30 - 15:15 UTC). "
            "Impact: All API endpoints returning 503 errors. "
            "Root cause: Database connection pool exhausted due to connection leak. "
            "The leak was introduced in commit abc123 on 2024-01-14. "
            "Resolution: Rolled back to previous version, patched connection handling. "
            "Affected users: Approximately 5000 active sessions. "
            "Preventive measures: Add connection pool monitoring, "
            "improve code review for database operations. "
            "Follow-up: Schedule post-mortem for 2024-01-17."
        )

        compressed = compressor.compress_outcome(
            incident_report, CompressionLevel.MEDIUM
        )

        assert compressed is not None
        # Should identify constraints/requirements
        all_content = (
            compressed.summary +
            " ".join(compressed.constraints) +
            " ".join(compressed.key_facts)
        ).lower()

        # Key incident details should be preserved somewhere
        has_key_info = (
            "database" in all_content or
            "connection" in all_content or
            "outage" in all_content or
            "503" in all_content
        )
        assert has_key_info

    def test_compress_meeting_notes(self):
        """Should compress meeting notes extracting decisions."""
        llm = MockLLMClient(
            default_response="""SUMMARY: Sprint planning decided on auth feature priority and Q2 timeline.
KEY_FACTS:
- Authentication feature is top priority
- Target completion by end of Q2
- Team capacity is 120 story points
CONSTRAINTS:
- Must maintain backwards compatibility
- Cannot exceed budget
PATTERNS:
- Break large features into smaller stories"""
        )
        compressor = MemoryCompressor(llm_client=llm)

        meeting_notes = (
            "Sprint Planning Meeting - January 10, 2024. "
            "Attendees: Alice, Bob, Charlie, Diana. "
            "Discussed upcoming sprint priorities. "
            "Alice proposed focusing on the authentication feature. "
            "Bob raised concerns about timeline. "
            "After discussion, agreed to prioritize auth. "
            "Target completion by end of Q2. "
            "Team capacity estimated at 120 story points. "
            "Backwards compatibility is a must. "
            "Budget constraints discussed. "
            "Action: Break down auth feature into smaller stories."
        )

        compressed = compressor.compress_conversation(meeting_notes)

        assert compressed is not None
        assert len(compressed.key_facts) > 0
        # Should extract key decisions
        all_facts = " ".join(compressed.key_facts).lower()
        assert "auth" in all_facts or "priority" in all_facts


class TestCompressionPerformance:
    """Performance tests for compression."""

    def test_compression_time_reasonable(self):
        """Compression should complete in reasonable time."""
        import time

        compressor = MemoryCompressor()

        # Generate substantial content
        content = " ".join([
            f"Sentence {i} with various content about topic {i % 10}. "
            f"Details include aspect {i % 5} and consideration {i % 3}."
            for i in range(100)
        ])

        start = time.time()
        result = compressor.compress(content, CompressionLevel.MEDIUM)
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert elapsed_ms < 5000  # Should complete in under 5 seconds

    def test_batch_compression_efficient(self):
        """Batch compression should be efficient."""
        import time

        compressor = MemoryCompressor()

        # Create multiple items to compress
        items = [
            f"Content item {i} with details about task {i}. " * 10
            for i in range(10)
        ]

        start = time.time()
        results = compressor.batch_compress(items, CompressionLevel.MEDIUM)
        elapsed_ms = (time.time() - start) * 1000

        assert len(results) == 10
        assert all(r.success for r in results)
        assert elapsed_ms < 10000  # Under 10 seconds for 10 items
