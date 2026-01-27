"""
Memory Growth Performance Tests.

Tests for how the system handles increasing memory volume.
"""

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alma import ALMA, MemoryScope
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage
from alma.types import DomainKnowledge, Outcome


class TestMemoryGrowth:
    """Test system behavior as memories accumulate."""

    @pytest.fixture
    def growth_alma(self, perf_storage_dir: Path) -> ALMA:
        """Create ALMA for growth testing."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing", "patterns"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="growth-test",
        )

    def test_learning_rate_consistent(self, growth_alma: ALMA):
        """Learning operations should maintain consistent speed."""
        times = []

        for i in range(100):
            start = time.perf_counter()
            growth_alma.learn(
                agent="helena",
                task=f"Test task {i}",
                outcome="success" if i % 3 != 0 else "failure",
                strategy_used=f"Strategy {i % 10}",
                task_type="testing",
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Average learning time should be < 50ms
        avg_time = sum(times) / len(times)
        assert (
            avg_time < 50
        ), f"Average learning time: {avg_time:.1f}ms, expected < 50ms"

        # Last 10 should not be significantly slower than first 10
        first_10_avg = sum(times[:10]) / 10
        last_10_avg = sum(times[-10:]) / 10

        # Allow 3x degradation (file I/O can vary)
        assert last_10_avg < first_10_avg * 3, (
            f"Performance degraded: first 10 avg {first_10_avg:.1f}ms, "
            f"last 10 avg {last_10_avg:.1f}ms"
        )

    def test_stats_performance(self, growth_alma: ALMA):
        """Stats retrieval should be fast even with many memories."""
        # Add significant data
        for i in range(200):
            growth_alma.learn(
                agent="helena",
                task=f"Stats test {i}",
                outcome="success",
                strategy_used="standard",
                task_type="testing",
            )

        # Stats should be fast
        start = time.perf_counter()
        stats = growth_alma.get_stats(agent="helena")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Stats took {elapsed:.1f}ms, expected < 100ms"
        assert stats.get("outcomes_count", 0) == 200

    def test_forget_performance(self, growth_alma: ALMA):
        """Forget operation should be efficient."""
        # Add data to forget
        for i in range(100):
            growth_alma.learn(
                agent="helena",
                task=f"Forget test {i}",
                outcome="success",
                strategy_used="standard",
                task_type="testing",
            )

        growth_alma.get_stats(agent="helena")

        # Forget with criteria that won't match anything (recent data)
        start = time.perf_counter()
        growth_alma.forget(
            agent="helena",
            older_than_days=1,  # Nothing is older than 1 day
            below_confidence=0.0,
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Forget should be fast even if nothing is pruned
        assert elapsed < 200, f"Forget took {elapsed:.1f}ms, expected < 200ms"


class TestLargeVolumePerformance:
    """Test with large data volumes."""

    def test_large_outcome_volume(self, perf_storage_dir: Path):
        """Handle 1000+ outcomes efficiently."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        now = datetime.now(timezone.utc)

        # Create 1000 outcomes
        for i in range(1000):
            outcome = Outcome(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="volume-test",
                task_type=f"type_{i % 10}",
                task_description=f"Task {i}",
                success=i % 3 != 0,
                strategy_used=f"Strategy {i % 20}",
                duration_ms=100 + (i % 500),
                timestamp=now - timedelta(hours=i),
            )
            storage.save_outcome(outcome)

        # Retrieval should still be reasonable
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="volume-test",
        )

        start = time.perf_counter()
        alma.retrieve(task="Test pattern", agent="helena", top_k=10)
        elapsed = (time.perf_counter() - start) * 1000

        # Should handle 1000 outcomes in under 1 second
        assert elapsed < 1000, f"Large volume retrieval took {elapsed:.1f}ms"

    def test_knowledge_volume(self, perf_storage_dir: Path):
        """Handle many domain knowledge entries."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        now = datetime.now(timezone.utc)

        # Create 500 domain knowledge items
        for i in range(500):
            dk = DomainKnowledge(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="knowledge-test",
                domain=f"domain_{i % 30}",
                fact=f"Important fact {i}: This is knowledge about topic {i % 50}",
                source="bulk_load",
                confidence=0.8,
                last_verified=now,
            )
            storage.save_domain_knowledge(dk)

        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="knowledge-test",
        )

        # Retrieval should find relevant knowledge quickly
        start = time.perf_counter()
        alma.retrieve(task="domain_5 knowledge", agent="helena", top_k=10)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 500, f"Knowledge retrieval took {elapsed:.1f}ms"


class TestConcurrentLoad:
    """Test performance under simulated concurrent load."""

    def test_rapid_learn_retrieve_cycles(self, perf_storage_dir: Path):
        """Rapid alternation of learn and retrieve operations."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(storage=storage, embedding_provider="mock")
        learning = LearningProtocol(storage=storage, scopes=scopes)

        alma = ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="concurrent-test",
        )

        start = time.perf_counter()

        # Simulate rapid operations
        for i in range(50):
            # Learn
            alma.learn(
                agent="helena",
                task=f"Rapid task {i}",
                outcome="success",
                strategy_used="rapid strategy",
                task_type="testing",
            )

            # Retrieve
            alma.retrieve(task=f"Rapid query {i}", agent="helena", top_k=5)

        total_elapsed = (time.perf_counter() - start) * 1000

        # 50 learn + 50 retrieve should complete in < 10 seconds
        assert total_elapsed < 10000, f"100 operations took {total_elapsed:.1f}ms"

        # Average should be under 100ms per operation
        avg_per_op = total_elapsed / 100
        assert avg_per_op < 100, f"Average per operation: {avg_per_op:.1f}ms"
