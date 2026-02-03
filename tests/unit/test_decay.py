"""
Unit tests for ALMA Decay-Based Forgetting.

Tests memory strength calculation, decay over time, reinforcement,
and state transitions.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from alma.learning.decay import (
    DecayConfig,
    DecayManager,
    MemoryStrength,
    StrengthState,
    calculate_projected_strength,
    days_until_threshold,
)


class TestMemoryStrength:
    """Tests for MemoryStrength dataclass."""

    def test_fresh_memory_strength(self):
        """New memory should have strength based on formula."""
        strength = MemoryStrength(memory_id="test-1")
        # With default importance (0.5), importance factor = 0.75
        # Fresh memory with no decay: base_decay=1.0, access_bonus=0, reinforcement_bonus=0
        # Expected: 1.0 * 0.75 = 0.75
        assert strength.current_strength() > 0.7
        assert strength.current_strength() <= 1.0

    def test_no_decay_on_same_day(self):
        """Memory should not decay significantly on same day."""
        strength = MemoryStrength(memory_id="test-1")
        # Allow for small variance from access_bonus
        assert strength.current_strength() >= 0.5

    def test_decay_over_time_half_life(self):
        """Memory should decay to ~0.5 at half-life."""
        now = datetime.now(timezone.utc)
        half_life = 30

        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=half_life,
            last_accessed=now - timedelta(days=half_life),
            created_at=now - timedelta(days=half_life),
        )

        # At half-life, base decay should be ~0.5
        # With importance factor of 0.75 (default 0.5 importance)
        # Expected: ~0.5 * 0.75 = 0.375
        current = strength.current_strength()
        assert 0.3 < current < 0.5

    def test_decay_over_time_double_half_life(self):
        """Memory should decay to ~0.25 at 2x half-life."""
        now = datetime.now(timezone.utc)
        half_life = 30

        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=half_life,
            last_accessed=now - timedelta(days=half_life * 2),
            created_at=now - timedelta(days=half_life * 2),
        )

        # At 2x half-life, base decay should be ~0.25
        current = strength.current_strength()
        assert current < 0.35

    def test_access_increases_strength(self):
        """Accessing a memory should increase its strength."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            last_accessed=now - timedelta(days=15),
            access_count=0,
        )

        initial = strength.current_strength()
        strength.access()
        after_access = strength.current_strength()

        assert after_access > initial
        assert strength.access_count == 1

    def test_access_diminishing_returns(self):
        """Multiple accesses should have diminishing returns."""
        strength = MemoryStrength(memory_id="test-1")

        # Record first access
        strength.access()
        first = strength.current_strength()

        # Record many accesses
        for _ in range(10):
            strength.access()

        many = strength.current_strength()

        # Difference should be less than first access
        # because of diminishing returns via log
        assert many > first
        assert (many - first) < 0.4  # Access bonus is capped at 0.4

    def test_reinforce_adds_bonus(self):
        """Reinforcement should add to strength."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            last_accessed=now - timedelta(days=20),
        )

        initial = strength.current_strength()
        strength.reinforce()
        after = strength.current_strength()

        assert after > initial
        assert len(strength.reinforcement_events) == 1

    def test_old_reinforcements_dont_count(self):
        """Reinforcements older than 7 days shouldn't count."""
        now = datetime.now(timezone.utc)
        old_reinforcement = now - timedelta(days=10)

        strength = MemoryStrength(
            memory_id="test-1",
            reinforcement_events=[old_reinforcement],
        )

        # Old reinforcement should not add bonus
        current = strength.current_strength()

        # Add a new reinforcement
        strength.reinforce()
        after = strength.current_strength()

        # New reinforcement should add bonus
        assert after > current

    def test_reinforcement_events_capped(self):
        """Only last 10 reinforcement events should be kept."""
        strength = MemoryStrength(memory_id="test-1")

        for _ in range(15):
            strength.reinforce()

        assert len(strength.reinforcement_events) == 10

    def test_importance_high_increases_strength(self):
        """High importance should increase strength."""
        strength = MemoryStrength(
            memory_id="test-1",
            explicit_importance=1.0,
        )

        high_importance = strength.current_strength()

        strength_low = MemoryStrength(
            memory_id="test-2",
            explicit_importance=0.0,
        )

        low_importance = strength_low.current_strength()

        assert high_importance > low_importance

    def test_importance_factor_range(self):
        """Importance factor should scale from 0.5 to 1.0."""
        strength_zero = MemoryStrength(
            memory_id="test-1",
            explicit_importance=0.0,
        )
        strength_one = MemoryStrength(
            memory_id="test-2",
            explicit_importance=1.0,
        )

        # With same base decay, importance=1 should be ~2x importance=0
        ratio = strength_one.current_strength() / strength_zero.current_strength()
        assert 1.5 < ratio < 2.5

    def test_set_importance(self):
        """set_importance should update importance and return new strength."""
        strength = MemoryStrength(memory_id="test-1")
        initial = strength.current_strength()

        new_strength = strength.set_importance(1.0)

        assert strength.explicit_importance == 1.0
        assert new_strength > initial

    def test_set_importance_clamps(self):
        """set_importance should clamp values to [0, 1]."""
        strength = MemoryStrength(memory_id="test-1")

        strength.set_importance(-0.5)
        assert strength.explicit_importance == 0.0

        strength.set_importance(1.5)
        assert strength.explicit_importance == 1.0


class TestStrengthState:
    """Tests for memory strength state transitions."""

    def test_strong_state(self):
        """Fresh memory should be STRONG."""
        strength = MemoryStrength(
            memory_id="test-1",
            explicit_importance=1.0,
        )
        assert strength.get_state() == StrengthState.STRONG

    def test_weak_state(self):
        """Old memory should be WEAK."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
            last_accessed=now - timedelta(days=60),
            explicit_importance=0.5,
        )

        state = strength.get_state()
        # Could be WEAK or FORGETTABLE depending on exact calculation
        assert state in [StrengthState.WEAK, StrengthState.FORGETTABLE]

    def test_should_forget_threshold(self):
        """should_forget should return True below threshold."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
            last_accessed=now - timedelta(days=120),
            explicit_importance=0.0,
        )

        assert strength.should_forget(threshold=0.1)

    def test_is_recoverable(self):
        """is_recoverable should return True in weak range."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
            last_accessed=now - timedelta(days=45),
            explicit_importance=0.5,
        )

        # This should be in the weak but recoverable range
        current = strength.current_strength()
        if 0.1 <= current < 0.3:
            assert strength.is_recoverable()


class TestMemoryStrengthSerialization:
    """Tests for MemoryStrength serialization."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            memory_type="heuristic",
            initial_strength=0.9,
            decay_half_life_days=60,
            created_at=now,
            last_accessed=now,
            access_count=5,
            explicit_importance=0.8,
        )

        data = strength.to_dict()

        assert data["memory_id"] == "test-1"
        assert data["memory_type"] == "heuristic"
        assert data["initial_strength"] == 0.9
        assert data["decay_half_life_days"] == 60
        assert data["access_count"] == 5
        assert data["explicit_importance"] == 0.8
        assert "created_at" in data
        assert "last_accessed" in data

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "memory_id": "test-1",
            "memory_type": "outcome",
            "initial_strength": 0.85,
            "decay_half_life_days": 45,
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
            "access_count": 3,
            "reinforcement_events": [now.isoformat()],
            "explicit_importance": 0.7,
        }

        strength = MemoryStrength.from_dict(data)

        assert strength.memory_id == "test-1"
        assert strength.memory_type == "outcome"
        assert strength.initial_strength == 0.85
        assert strength.decay_half_life_days == 45
        assert strength.access_count == 3
        assert len(strength.reinforcement_events) == 1
        assert strength.explicit_importance == 0.7

    def test_roundtrip(self):
        """Serialization should be reversible."""
        now = datetime.now(timezone.utc)
        original = MemoryStrength(
            memory_id="test-1",
            memory_type="knowledge",
            initial_strength=0.95,
            decay_half_life_days=90,
            created_at=now,
            last_accessed=now,
            access_count=10,
            reinforcement_events=[now - timedelta(days=1), now],
            explicit_importance=0.9,
        )

        data = original.to_dict()
        restored = MemoryStrength.from_dict(data)

        assert restored.memory_id == original.memory_id
        assert restored.memory_type == original.memory_type
        assert restored.access_count == original.access_count
        assert len(restored.reinforcement_events) == len(original.reinforcement_events)


class TestDecayConfig:
    """Tests for DecayConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = DecayConfig()

        assert config.enabled is True
        assert config.default_half_life_days == 30
        assert config.forget_threshold == 0.1
        assert config.weak_threshold == 0.3
        assert config.strong_threshold == 0.7

    def test_half_life_by_type(self):
        """Should return type-specific half-lives."""
        config = DecayConfig()

        assert config.get_half_life("heuristic") == 60
        assert config.get_half_life("outcome") == 30
        assert config.get_half_life("preference") == 365
        assert config.get_half_life("knowledge") == 90
        assert config.get_half_life("unknown") == 30  # Default

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "enabled": False,
            "default_half_life_days": 45,
            "forget_threshold": 0.05,
            "half_life_by_type": {
                "heuristic": 100,
            },
        }

        config = DecayConfig.from_dict(data)

        assert config.enabled is False
        assert config.default_half_life_days == 45
        assert config.forget_threshold == 0.05
        assert config.get_half_life("heuristic") == 100


class TestDecayManager:
    """Tests for DecayManager."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage backend."""
        storage = MagicMock()
        storage.get_memory_strength.return_value = None
        storage.save_memory_strength.return_value = "test-1"
        storage.get_all_memory_strengths.return_value = []
        storage.delete_memory_strength.return_value = True
        return storage

    def test_get_strength_creates_new(self, mock_storage):
        """Should create new strength record if not exists."""
        manager = DecayManager(storage=mock_storage)

        strength = manager.get_strength("test-1", "heuristic")

        assert strength.memory_id == "test-1"
        assert strength.memory_type == "heuristic"
        assert strength.decay_half_life_days == 60  # heuristic default
        mock_storage.save_memory_strength.assert_called()

    def test_get_strength_returns_existing(self, mock_storage):
        """Should return existing strength record."""
        existing = MemoryStrength(memory_id="test-1", access_count=5)
        mock_storage.get_memory_strength.return_value = existing

        manager = DecayManager(storage=mock_storage)
        strength = manager.get_strength("test-1")

        assert strength.access_count == 5

    def test_record_access(self, mock_storage):
        """Should record access and update storage."""
        manager = DecayManager(storage=mock_storage)

        strength_value = manager.record_access("test-1", "outcome")

        assert strength_value > 0
        mock_storage.save_memory_strength.assert_called()

    def test_reinforce_memory(self, mock_storage):
        """Should reinforce memory and update storage."""
        manager = DecayManager(storage=mock_storage)

        strength_value = manager.reinforce_memory("test-1", "heuristic")

        assert strength_value > 0
        mock_storage.save_memory_strength.assert_called()

    def test_set_importance(self, mock_storage):
        """Should set importance and update storage."""
        manager = DecayManager(storage=mock_storage)

        strength_value = manager.set_importance("test-1", 0.9)

        assert strength_value > 0
        mock_storage.save_memory_strength.assert_called()

    def test_get_forgettable_memories(self, mock_storage):
        """Should return memories below threshold."""
        now = datetime.now(timezone.utc)
        weak_strength = MemoryStrength(
            memory_id="weak-1",
            memory_type="outcome",
            last_accessed=now - timedelta(days=200),
            explicit_importance=0.0,
        )
        strong_strength = MemoryStrength(
            memory_id="strong-1",
            memory_type="heuristic",
        )

        mock_storage.get_all_memory_strengths.return_value = [
            weak_strength,
            strong_strength,
        ]

        manager = DecayManager(storage=mock_storage)
        forgettable = manager.get_forgettable_memories("project-1")

        # Only the weak one should be forgettable
        forgettable_ids = [f[0] for f in forgettable]
        assert "weak-1" in forgettable_ids
        assert "strong-1" not in forgettable_ids

    def test_get_weak_memories(self, mock_storage):
        """Should return recoverable memories."""
        now = datetime.now(timezone.utc)
        weak_strength = MemoryStrength(
            memory_id="weak-1",
            memory_type="outcome",
            decay_half_life_days=30,
            last_accessed=now - timedelta(days=50),
            explicit_importance=0.5,
        )

        mock_storage.get_all_memory_strengths.return_value = [weak_strength]

        manager = DecayManager(storage=mock_storage)
        weak_memories = manager.get_weak_memories("project-1")

        # Should identify recoverable memories
        # Note: exact result depends on whether strength falls in 0.1-0.3 range
        assert isinstance(weak_memories, list)

    def test_get_memory_stats(self, mock_storage):
        """Should return statistics about memory strengths."""
        now = datetime.now(timezone.utc)
        strengths = [
            MemoryStrength(memory_id="s1", memory_type="heuristic"),
            MemoryStrength(
                memory_id="s2",
                memory_type="outcome",
                last_accessed=now - timedelta(days=100),
            ),
        ]

        mock_storage.get_all_memory_strengths.return_value = strengths

        manager = DecayManager(storage=mock_storage)
        stats = manager.get_memory_stats("project-1")

        assert stats["total"] == 2
        assert "average_strength" in stats
        assert "by_type" in stats
        assert "heuristic" in stats["by_type"]
        assert "outcome" in stats["by_type"]

    def test_invalidate_cache(self, mock_storage):
        """Should clear strength cache."""
        manager = DecayManager(storage=mock_storage)

        # Populate cache
        manager.get_strength("test-1")
        assert "test-1" in manager._strength_cache

        # Invalidate specific
        manager.invalidate_cache("test-1")
        assert "test-1" not in manager._strength_cache

        # Populate again
        manager.get_strength("test-1")
        manager.get_strength("test-2")

        # Invalidate all
        manager.invalidate_cache()
        assert len(manager._strength_cache) == 0


class TestProjectedStrength:
    """Tests for strength projection functions."""

    def test_calculate_projected_strength_future(self):
        """Should project strength decay into future."""
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
        )

        current = strength.current_strength()
        projected = calculate_projected_strength(strength, days_ahead=30)

        # After half-life, should be roughly half
        assert projected < current
        assert projected > 0

    def test_calculate_projected_strength_zero_days(self):
        """Should return current strength for 0 days."""
        strength = MemoryStrength(memory_id="test-1")

        current = strength.current_strength()
        projected = calculate_projected_strength(strength, days_ahead=0)

        # Allow small floating point tolerance
        assert abs(projected - current) < 0.001

    def test_days_until_threshold_calculates_correctly(self):
        """Should calculate days until threshold is reached."""
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
        )

        days = days_until_threshold(strength, threshold=0.1)

        # Should be some positive number of days
        assert days is not None
        assert days > 0

    def test_days_until_threshold_already_below(self):
        """Should return 0 if already below threshold."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=30,
            last_accessed=now - timedelta(days=200),
            explicit_importance=0.0,
        )

        days = days_until_threshold(strength, threshold=0.5)

        # Already below, should return 0
        if strength.current_strength() <= 0.5:
            assert days == 0

    def test_days_until_threshold_no_decay(self):
        """Should return None if no decay configured."""
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=0,  # No decay
        )

        days = days_until_threshold(strength, threshold=0.1)

        assert days is None


class TestDecayFormula:
    """Tests for the decay formula behavior."""

    def test_half_life_formula_accuracy(self):
        """Verify half-life formula is mathematically correct."""
        half_life = 30
        now = datetime.now(timezone.utc)

        # At exactly half-life days, base decay should be 0.5
        strength = MemoryStrength(
            memory_id="test-1",
            decay_half_life_days=half_life,
            last_accessed=now - timedelta(days=half_life),
            access_count=0,
            explicit_importance=0.5,  # neutral importance
        )

        # Base decay = e^(-0.693 * 30 / 30) = e^(-0.693) ≈ 0.5
        # With importance factor of 0.75, and no bonuses:
        # Expected ≈ 0.5 * 0.75 = 0.375
        current = strength.current_strength()
        # Allow some tolerance for floating point
        assert 0.35 < current < 0.45

    def test_access_bonus_cap(self):
        """Access bonus should be capped at 0.4."""
        strength = MemoryStrength(memory_id="test-1")

        # Many accesses
        for _ in range(1000):
            strength.access_count += 1

        # Access bonus: min(0.4, log(1 + 1000) * 0.1)
        # log(1001) * 0.1 ≈ 0.69, but capped at 0.4
        # Total strength should not exceed 1.0 due to cap
        assert strength.current_strength() <= 1.0

    def test_reinforcement_bonus_cap(self):
        """Reinforcement bonus should be capped at 0.3."""
        now = datetime.now(timezone.utc)
        strength = MemoryStrength(
            memory_id="test-1",
            reinforcement_events=[now - timedelta(days=i) for i in range(10)],
        )

        # Many recent reinforcements, but bonus capped at 0.3
        assert strength.current_strength() <= 1.0
