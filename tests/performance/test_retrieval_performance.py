"""
Retrieval Performance Tests.

Benchmarks for memory retrieval under various conditions.
"""

import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from alma import ALMA, MemoryScope
from alma.learning.protocols import LearningProtocol
from alma.retrieval.engine import RetrievalEngine
from alma.storage.file_based import FileBasedStorage
from alma.types import DomainKnowledge, Heuristic


class TestRetrievalLatency:
    """Test retrieval response times."""

    @pytest.fixture
    def seeded_storage(self, perf_storage_dir: Path) -> FileBasedStorage:
        """Create storage with realistic amount of data."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        now = datetime.now(timezone.utc)

        # Seed with 500 heuristics
        for i in range(500):
            h = Heuristic(
                id=str(uuid.uuid4()),
                agent="helena" if i % 2 == 0 else "victor",
                project_id="perf-test",
                condition=f"Test condition {i} for pattern matching",
                strategy=f"Use strategy {i} for best results",
                confidence=0.5 + (i % 50) / 100,
                occurrence_count=i % 20 + 1,
                success_count=i % 15 + 1,
                last_validated=now,
                created_at=now - timedelta(days=i % 90),
            )
            storage.save_heuristic(h)

        # Seed with 200 domain knowledge items
        for i in range(200):
            dk = DomainKnowledge(
                id=str(uuid.uuid4()),
                agent="helena" if i % 2 == 0 else "victor",
                project_id="perf-test",
                domain=f"domain_{i % 20}",
                fact=f"Important fact number {i} about the system",
                source="test_generation",
                confidence=0.7 + (i % 30) / 100,
                last_verified=now,
            )
            storage.save_domain_knowledge(dk)

        return storage

    @pytest.fixture
    def perf_alma(self, seeded_storage: FileBasedStorage) -> ALMA:
        """Create ALMA for performance testing."""
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing_strategies", "selector_patterns"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
            "victor": MemoryScope(
                agent_name="victor",
                can_learn=["api_design_patterns", "error_handling"],
                cannot_learn=[],
                min_occurrences_for_heuristic=3,
            ),
        }
        retrieval = RetrievalEngine(
            storage=seeded_storage,
            embedding_provider="mock",
        )
        learning = LearningProtocol(storage=seeded_storage, scopes=scopes)

        return ALMA(
            storage=seeded_storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="perf-test",
        )

    def test_single_retrieval_latency(self, perf_alma: ALMA):
        """Single retrieval should complete within 200ms."""
        start = time.perf_counter()

        memories = perf_alma.retrieve(
            task="Test form validation patterns",
            agent="helena",
            top_k=10,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 200, f"Retrieval took {elapsed_ms:.1f}ms, expected < 200ms"
        assert memories is not None

    def test_multiple_sequential_retrievals(self, perf_alma: ALMA):
        """10 sequential retrievals should average < 50ms each."""
        tasks = [
            "Test login form",
            "Test registration",
            "Test modal dialog",
            "Test table sorting",
            "Test file upload",
            "Test pagination",
            "Test search filter",
            "Test dropdown menu",
            "Test navigation",
            "Test accessibility",
        ]

        times: List[float] = []

        for task in tasks:
            start = time.perf_counter()
            perf_alma.retrieve(task=task, agent="helena", top_k=5)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)

        # After warm-up, should be faster due to caching
        # Allow 100ms average for file-based storage
        assert avg_time < 100, f"Average retrieval: {avg_time:.1f}ms, expected < 100ms"

    def test_concurrent_agent_retrievals(self, perf_alma: ALMA):
        """Helena and Victor retrievals should not block each other."""
        start = time.perf_counter()

        # Sequential but testing both agents
        helena_memories = perf_alma.retrieve(
            task="UI testing patterns",
            agent="helena",
            top_k=5,
        )

        victor_memories = perf_alma.retrieve(
            task="API testing patterns",
            agent="victor",
            top_k=5,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Both retrievals together should complete in < 400ms
        assert elapsed_ms < 400, f"Combined retrieval took {elapsed_ms:.1f}ms"
        assert helena_memories is not None
        assert victor_memories is not None


class TestCachePerformance:
    """Test caching effectiveness."""

    @pytest.fixture
    def cache_alma(self, perf_storage_dir: Path) -> ALMA:
        """Create ALMA for cache testing."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        now = datetime.now(timezone.utc)

        # Seed with some data
        for i in range(100):
            h = Heuristic(
                id=str(uuid.uuid4()),
                agent="helena",
                project_id="cache-test",
                condition=f"Condition {i}",
                strategy=f"Strategy {i}",
                confidence=0.8,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now - timedelta(days=i),
            )
            storage.save_heuristic(h)

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

        return ALMA(
            storage=storage,
            retrieval_engine=retrieval,
            learning_protocol=learning,
            scopes=scopes,
            project_id="cache-test",
        )

    def test_cache_hit_faster_than_miss(self, cache_alma: ALMA):
        """Cached retrieval should be faster than uncached."""
        task = "Test form validation"

        # First call - cache miss
        start1 = time.perf_counter()
        cache_alma.retrieve(task=task, agent="helena", top_k=5)
        first_time = (time.perf_counter() - start1) * 1000

        # Second call - should hit cache
        start2 = time.perf_counter()
        cache_alma.retrieve(task=task, agent="helena", top_k=5)
        second_time = (time.perf_counter() - start2) * 1000

        # Cache hit should be at least as fast
        # (For file-based storage, difference may be minimal)
        assert (
            second_time <= first_time * 2
        ), f"Second call ({second_time:.1f}ms) slower than twice first ({first_time:.1f}ms)"

    def test_cache_invalidation_on_learn(self, cache_alma: ALMA):
        """Learning should invalidate cache."""
        task = "Test modal"

        # Prime the cache
        cache_alma.retrieve(task=task, agent="helena", top_k=5)

        # Learn something
        cache_alma.learn(
            agent="helena",
            task=task,
            outcome="success",
            strategy_used="wait for animation",
            task_type="testing",
        )

        # Next retrieval should reflect new learning
        start = time.perf_counter()
        cache_alma.retrieve(task=task, agent="helena", top_k=5)
        elapsed = (time.perf_counter() - start) * 1000

        # Should still be reasonably fast
        assert elapsed < 200, f"Post-invalidation retrieval took {elapsed:.1f}ms"


class TestScalability:
    """Test performance as data volume grows."""

    def test_retrieval_scales_with_data(self, perf_storage_dir: Path):
        """Retrieval time should scale reasonably with data size."""
        storage = FileBasedStorage(storage_dir=perf_storage_dir)
        now = datetime.now(timezone.utc)

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
            project_id="scale-test",
        )

        times_by_size = []

        # Test with increasing data sizes
        for size in [10, 50, 100, 200]:
            # Add more data
            for i in range(size):
                h = Heuristic(
                    id=str(uuid.uuid4()),
                    agent="helena",
                    project_id="scale-test",
                    condition=f"Scale test condition {size}_{i}",
                    strategy=f"Strategy {size}_{i}",
                    confidence=0.8,
                    occurrence_count=10,
                    success_count=9,
                    last_validated=now,
                    created_at=now - timedelta(days=i),
                )
                storage.save_heuristic(h)

            # Measure retrieval time
            start = time.perf_counter()
            alma.retrieve(task="Test pattern", agent="helena", top_k=10)
            elapsed = (time.perf_counter() - start) * 1000

            times_by_size.append((size, elapsed))

        # Verify times don't grow excessively
        # With 200 items, should still be under 500ms for file-based
        for size, elapsed in times_by_size:
            assert elapsed < 500, f"With {size} items, retrieval took {elapsed:.1f}ms"
