"""
Unit tests for ALMA storage backends.
"""

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alma.storage.file_based import FileBasedStorage
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import (
    AntiPattern,
    DomainKnowledge,
    Heuristic,
    Outcome,
    UserPreference,
)


class TestFileBasedStorage:
    """Tests for FileBasedStorage backend."""

    @pytest.fixture
    def storage(self):
        """Create temporary storage for tests."""
        temp_dir = tempfile.mkdtemp()
        storage = FileBasedStorage(storage_dir=Path(temp_dir))
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_heuristic(self):
        """Create sample heuristic."""
        return Heuristic(
            id="heur_test_001",
            agent="helena",
            project_id="test-project",
            condition="form with multiple inputs",
            strategy="test happy path first",
            confidence=0.85,
            occurrence_count=5,
            success_count=4,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_outcome(self):
        """Create sample outcome."""
        return Outcome(
            id="out_test_001",
            agent="victor",
            project_id="test-project",
            task_type="api_testing",
            task_description="Test login endpoint",
            success=True,
            strategy_used="Test with valid and invalid tokens",
            duration_ms=1500,
            timestamp=datetime.now(timezone.utc),
        )

    def test_save_and_get_heuristic(self, storage, sample_heuristic):
        """Test saving and retrieving a heuristic."""
        # Save
        heuristic_id = storage.save_heuristic(sample_heuristic)
        assert heuristic_id == sample_heuristic.id

        # Retrieve
        heuristics = storage.get_heuristics(
            project_id="test-project",
            agent="helena",
        )

        assert len(heuristics) == 1
        assert heuristics[0].id == sample_heuristic.id
        assert heuristics[0].strategy == sample_heuristic.strategy
        assert heuristics[0].confidence == sample_heuristic.confidence

    def test_save_and_get_outcome(self, storage, sample_outcome):
        """Test saving and retrieving an outcome."""
        # Save
        outcome_id = storage.save_outcome(sample_outcome)
        assert outcome_id == sample_outcome.id

        # Retrieve
        outcomes = storage.get_outcomes(
            project_id="test-project",
            agent="victor",
        )

        assert len(outcomes) == 1
        assert outcomes[0].id == sample_outcome.id
        assert outcomes[0].success
        assert outcomes[0].strategy_used == sample_outcome.strategy_used

    def test_filter_by_agent(self, storage, sample_heuristic):
        """Test filtering by agent."""
        # Save heuristic for helena
        storage.save_heuristic(sample_heuristic)

        # Create and save heuristic for victor
        victor_heuristic = Heuristic(
            id="heur_victor_001",
            agent="victor",
            project_id="test-project",
            condition="API endpoint",
            strategy="test error codes",
            confidence=0.9,
            occurrence_count=10,
            success_count=9,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        storage.save_heuristic(victor_heuristic)

        # Filter by helena
        helena_heuristics = storage.get_heuristics(
            project_id="test-project",
            agent="helena",
        )
        assert len(helena_heuristics) == 1
        assert helena_heuristics[0].agent == "helena"

        # Filter by victor
        victor_heuristics = storage.get_heuristics(
            project_id="test-project",
            agent="victor",
        )
        assert len(victor_heuristics) == 1
        assert victor_heuristics[0].agent == "victor"

    def test_min_confidence_filter(self, storage):
        """Test minimum confidence filtering."""
        # Create heuristics with different confidence levels
        low_confidence = Heuristic(
            id="heur_low",
            agent="helena",
            project_id="test-project",
            condition="condition",
            strategy="strategy",
            confidence=0.2,
            occurrence_count=1,
            success_count=0,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        high_confidence = Heuristic(
            id="heur_high",
            agent="helena",
            project_id="test-project",
            condition="condition",
            strategy="strategy",
            confidence=0.8,
            occurrence_count=10,
            success_count=8,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        storage.save_heuristic(low_confidence)
        storage.save_heuristic(high_confidence)

        # Filter with min_confidence=0.5
        heuristics = storage.get_heuristics(
            project_id="test-project",
            min_confidence=0.5,
        )

        assert len(heuristics) == 1
        assert heuristics[0].id == "heur_high"

    def test_delete_old_outcomes(self, storage):
        """Test deleting outcomes older than a threshold."""
        old_outcome = Outcome(
            id="out_old",
            agent="helena",
            project_id="test-project",
            task_type="testing",
            task_description="Old task",
            success=True,
            strategy_used="strategy",
            timestamp=datetime.now(timezone.utc) - timedelta(days=100),
        )
        new_outcome = Outcome(
            id="out_new",
            agent="helena",
            project_id="test-project",
            task_type="testing",
            task_description="New task",
            success=True,
            strategy_used="strategy",
            timestamp=datetime.now(timezone.utc),
        )

        storage.save_outcome(old_outcome)
        storage.save_outcome(new_outcome)

        # Delete outcomes older than 90 days
        deleted = storage.delete_outcomes_older_than(
            project_id="test-project",
            older_than=datetime.now(timezone.utc) - timedelta(days=90),
        )

        assert deleted == 1

        # Verify only new outcome remains
        outcomes = storage.get_outcomes(project_id="test-project")
        assert len(outcomes) == 1
        assert outcomes[0].id == "out_new"

    def test_user_preferences(self, storage):
        """Test saving and retrieving user preferences."""
        pref = UserPreference(
            id="pref_001",
            user_id="user123",
            category="communication",
            preference="No emojis in code",
            source="explicit_instruction",
            confidence=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        storage.save_user_preference(pref)

        prefs = storage.get_user_preferences(user_id="user123")
        assert len(prefs) == 1
        assert prefs[0].preference == "No emojis in code"

    def test_stats(self, storage, sample_heuristic, sample_outcome):
        """Test getting storage statistics."""
        storage.save_heuristic(sample_heuristic)
        storage.save_outcome(sample_outcome)

        stats = storage.get_stats(project_id="test-project")

        assert stats["heuristics_count"] == 1
        assert stats["outcomes_count"] == 1


class TestSQLiteStorage:
    """Tests for SQLiteStorage backend."""

    @pytest.fixture
    def storage(self):
        """Create temporary SQLite storage for tests."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        storage = SQLiteStorage(db_path=db_path)
        yield storage
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_heuristic(self):
        """Create sample heuristic with embedding."""
        return Heuristic(
            id="heur_sql_001",
            agent="helena",
            project_id="test-project",
            condition="form validation",
            strategy="test required fields first",
            confidence=0.75,
            occurrence_count=3,
            success_count=2,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.1] * 384,  # Fake embedding
        )

    @pytest.fixture
    def sample_outcome(self):
        """Create sample outcome with embedding."""
        return Outcome(
            id="out_sql_001",
            agent="victor",
            project_id="test-project",
            task_type="api_testing",
            task_description="Test authentication endpoint",
            success=True,
            strategy_used="Test with JWT tokens",
            duration_ms=2000,
            timestamp=datetime.now(timezone.utc),
            embedding=[0.2] * 384,  # Fake embedding
        )

    def test_save_and_get_heuristic(self, storage, sample_heuristic):
        """Test saving and retrieving a heuristic."""
        heuristic_id = storage.save_heuristic(sample_heuristic)
        assert heuristic_id == sample_heuristic.id

        heuristics = storage.get_heuristics(
            project_id="test-project",
            agent="helena",
        )

        assert len(heuristics) == 1
        assert heuristics[0].id == sample_heuristic.id
        assert heuristics[0].strategy == sample_heuristic.strategy

    def test_save_and_get_outcome(self, storage, sample_outcome):
        """Test saving and retrieving an outcome."""
        outcome_id = storage.save_outcome(sample_outcome)
        assert outcome_id == sample_outcome.id

        outcomes = storage.get_outcomes(
            project_id="test-project",
            agent="victor",
        )

        assert len(outcomes) == 1
        assert outcomes[0].id == sample_outcome.id
        assert outcomes[0].success

    def test_vector_search_heuristics(self, storage):
        """Test vector similarity search for heuristics."""
        # Create heuristics with different embeddings
        h1 = Heuristic(
            id="heur_v1",
            agent="helena",
            project_id="test-project",
            condition="form testing",
            strategy="test inputs",
            confidence=0.8,
            occurrence_count=5,
            success_count=4,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[1.0] + [0.0] * 383,  # Similar to query
        )
        h2 = Heuristic(
            id="heur_v2",
            agent="helena",
            project_id="test-project",
            condition="api testing",
            strategy="test endpoints",
            confidence=0.7,
            occurrence_count=3,
            success_count=2,
            last_validated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            embedding=[0.0] * 383 + [1.0],  # Different from query
        )

        storage.save_heuristic(h1)
        storage.save_heuristic(h2)

        # Search with embedding similar to h1
        query_embedding = [0.9] + [0.1] * 383
        heuristics = storage.get_heuristics(
            project_id="test-project",
            embedding=query_embedding,
            top_k=2,
        )

        # Both should be returned since we're just filtering
        assert len(heuristics) >= 1

    def test_increment_heuristic_occurrence(self, storage, sample_heuristic):
        """Test incrementing heuristic occurrence count."""
        storage.save_heuristic(sample_heuristic)

        # Increment with success
        storage.increment_heuristic_occurrence(sample_heuristic.id, success=True)

        heuristics = storage.get_heuristics(project_id="test-project")
        assert heuristics[0].occurrence_count == sample_heuristic.occurrence_count + 1
        assert heuristics[0].success_count == sample_heuristic.success_count + 1

    def test_update_heuristic(self, storage, sample_heuristic):
        """Test updating heuristic fields."""
        storage.save_heuristic(sample_heuristic)

        # Update confidence
        storage.update_heuristic(
            sample_heuristic.id,
            {"confidence": 0.95, "strategy": "updated strategy"},
        )

        heuristics = storage.get_heuristics(project_id="test-project")
        assert heuristics[0].confidence == 0.95
        assert heuristics[0].strategy == "updated strategy"

    def test_domain_knowledge(self, storage):
        """Test saving and retrieving domain knowledge."""
        knowledge = DomainKnowledge(
            id="dk_001",
            agent="helena",
            project_id="test-project",
            domain="authentication",
            fact="Login uses JWT with 24h expiry",
            source="code_analysis",
            confidence=0.9,
            last_verified=datetime.now(timezone.utc),
        )

        storage.save_domain_knowledge(knowledge)

        results = storage.get_domain_knowledge(
            project_id="test-project",
            agent="helena",
        )

        assert len(results) == 1
        assert results[0].fact == knowledge.fact

    def test_anti_patterns(self, storage):
        """Test saving and retrieving anti-patterns."""
        anti = AntiPattern(
            id="anti_001",
            agent="victor",
            project_id="test-project",
            pattern="Using sleep() for async waits",
            why_bad="Causes flaky tests",
            better_alternative="Use explicit waits",
            occurrence_count=3,
            last_seen=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        storage.save_anti_pattern(anti)

        results = storage.get_anti_patterns(
            project_id="test-project",
            agent="victor",
        )

        assert len(results) == 1
        assert results[0].pattern == anti.pattern
        assert results[0].better_alternative == anti.better_alternative

    def test_stats(self, storage, sample_heuristic, sample_outcome):
        """Test getting storage statistics."""
        storage.save_heuristic(sample_heuristic)
        storage.save_outcome(sample_outcome)

        stats = storage.get_stats(project_id="test-project")

        assert stats["heuristics_count"] == 1
        assert stats["outcomes_count"] == 1
        assert stats["storage_type"] == "sqlite"


class TestStorageFromConfig:
    """Test storage initialization from config."""

    def test_file_based_from_config(self):
        """Test FileBasedStorage from config."""
        temp_dir = tempfile.mkdtemp()
        try:
            config = {"storage_dir": temp_dir}
            storage = FileBasedStorage.from_config(config)
            assert storage.storage_dir == Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir)

    def test_sqlite_from_config(self):
        """Test SQLiteStorage from config."""
        temp_dir = tempfile.mkdtemp()
        try:
            config = {
                "storage_dir": temp_dir,
                "db_name": "test.db",
                "embedding_dim": 384,
            }
            storage = SQLiteStorage.from_config(config)
            assert storage.db_path == Path(temp_dir) / "test.db"
        finally:
            shutil.rmtree(temp_dir)
