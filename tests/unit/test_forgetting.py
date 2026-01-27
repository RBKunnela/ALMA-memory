"""
Unit tests for ALMA Forgetting Mechanism.

Tests cover:
- Decay Functions (Exponential, Linear, Step, NoDecay)
- Confidence Decayer
- Memory Health Monitor
- Cleanup Scheduler
- Forgetting Engine
- Prune Policy
"""

import pytest
import time
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

from alma.learning.forgetting import (
    # Forgetting Engine
    ForgettingEngine,
    PrunePolicy,
    PruneResult,
    PruneSummary,
    PruneReason,
    # Decay Functions
    DecayFunction,
    ExponentialDecay,
    LinearDecay,
    StepDecay,
    NoDecay,
    # Confidence Decay
    ConfidenceDecayer,
    DecayResult,
    # Memory Health Monitoring
    MemoryHealthMonitor,
    MemoryHealthMetrics,
    HealthAlert,
    HealthThresholds,
    # Cleanup Scheduling
    CleanupScheduler,
    CleanupJob,
    CleanupResult,
)
from alma.types import Heuristic, Outcome, DomainKnowledge, AntiPattern


class TestPrunePolicy:
    """Tests for PrunePolicy configuration."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        policy = PrunePolicy()

        assert policy.outcome_max_age_days == 90
        assert policy.heuristic_min_confidence == 0.3
        assert policy.max_heuristics_per_agent == 100

    def test_custom_values(self):
        """Test custom policy values."""
        policy = PrunePolicy(
            outcome_max_age_days=30,
            heuristic_min_confidence=0.5,
            max_heuristics_per_agent=50,
        )

        assert policy.outcome_max_age_days == 30
        assert policy.heuristic_min_confidence == 0.5
        assert policy.max_heuristics_per_agent == 50


class TestPruneSummary:
    """Tests for PruneSummary."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        summary = PruneSummary(
            outcomes_pruned=10,
            heuristics_pruned=5,
            knowledge_pruned=2,
            total_pruned=17,
        )

        d = summary.to_dict()

        assert d["outcomes_pruned"] == 10
        assert d["heuristics_pruned"] == 5
        assert d["total_pruned"] == 17


class TestForgettingEngine:
    """Tests for ForgettingEngine."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage with sample data."""
        storage = MagicMock()
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=120)  # Older than default 90 days

        storage.get_outcomes.return_value = [
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Recent task",
                success=True,
                strategy_used="strategy1",
                timestamp=now,
            ),
            Outcome(
                id="o2",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Old task",
                success=True,
                strategy_used="strategy2",
                timestamp=old,
            ),
        ]

        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="helena",
                project_id="test",
                condition="condition1",
                strategy="strategy1",
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="h2",
                agent="helena",
                project_id="test",
                condition="condition2",
                strategy="strategy2",
                confidence=0.2,  # Below threshold
                occurrence_count=10,
                success_count=2,
                last_validated=old,
                created_at=old,
            ),
        ]

        storage.get_domain_knowledge.return_value = []
        storage.get_anti_patterns.return_value = []
        storage.get_stats.return_value = {"agents": ["helena"]}
        storage.delete_outcomes_older_than.return_value = 1
        storage.delete_heuristic.return_value = True

        return storage

    def test_prune_identifies_stale_outcomes(self, mock_storage):
        """Test that stale outcomes are identified for pruning."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", dry_run=True)

        # One old outcome should be identified
        assert summary.outcomes_pruned >= 1

    def test_prune_identifies_low_confidence_heuristics(self, mock_storage):
        """Test that low confidence heuristics are identified."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", dry_run=True)

        # h2 has confidence 0.2, below threshold 0.3
        low_conf = [
            r for r in summary.pruned_items
            if r.reason == PruneReason.LOW_CONFIDENCE
        ]
        assert len(low_conf) >= 1

    def test_dry_run_no_deletion(self, mock_storage):
        """Test that dry run doesn't delete anything."""
        engine = ForgettingEngine(mock_storage)

        engine.prune("test", dry_run=True)

        # delete_heuristic should not be called in dry run
        mock_storage.delete_heuristic.assert_not_called()

    def test_actual_prune_deletes(self, mock_storage):
        """Test that actual prune deletes items."""
        engine = ForgettingEngine(mock_storage)

        engine.prune("test", dry_run=False)

        # delete methods should be called
        mock_storage.delete_outcomes_older_than.assert_called()

    def test_compute_decay_score(self, mock_storage):
        """Test decay score calculation."""
        engine = ForgettingEngine(mock_storage)

        # Perfect item
        score_perfect = engine.compute_decay_score(
            item_age_days=0,
            confidence=1.0,
            success_rate=1.0,
            occurrence_count=20,
        )

        # Old, low confidence item
        score_poor = engine.compute_decay_score(
            item_age_days=90,
            confidence=0.2,
            success_rate=0.3,
            occurrence_count=2,
        )

        assert score_perfect > score_poor
        assert score_perfect > 0.8
        assert score_poor < 0.4

    def test_identify_candidates(self, mock_storage):
        """Test candidate identification."""
        engine = ForgettingEngine(mock_storage)

        candidates = engine.identify_candidates("test", max_candidates=10)

        assert len(candidates) > 0
        # Candidates should be sorted by score (lowest first)
        if len(candidates) > 1:
            assert candidates[0]["score"] <= candidates[-1]["score"]

    def test_custom_policy(self, mock_storage):
        """Test using custom prune policy."""
        policy = PrunePolicy(
            outcome_max_age_days=30,
            heuristic_min_confidence=0.5,
        )
        engine = ForgettingEngine(mock_storage, policy=policy)

        # With stricter policy, more items should be identified
        summary = engine.prune("test", dry_run=True)

        # With higher confidence threshold, both heuristics might be flagged
        assert summary.heuristics_pruned >= 1

    def test_agent_specific_prune(self, mock_storage):
        """Test pruning for specific agent."""
        engine = ForgettingEngine(mock_storage)

        summary = engine.prune("test", agent="helena", dry_run=True)

        # Should only affect helena's items
        for item in summary.pruned_items:
            assert item.agent == "helena"


class TestPruneReason:
    """Tests for PruneReason enum."""

    def test_enum_values(self):
        """Test that all expected values exist."""
        assert PruneReason.STALE.value == "stale"
        assert PruneReason.LOW_CONFIDENCE.value == "low_confidence"
        assert PruneReason.LOW_SUCCESS_RATE.value == "low_success"
        assert PruneReason.QUOTA_EXCEEDED.value == "quota"


class TestPruneResult:
    """Tests for PruneResult dataclass."""

    def test_creation(self):
        """Test creating a prune result."""
        result = PruneResult(
            reason=PruneReason.STALE,
            item_type="outcome",
            item_id="o1",
            agent="helena",
            project_id="test",
            details="Older than 90 days",
        )

        assert result.reason == PruneReason.STALE
        assert result.item_type == "outcome"
        assert result.agent == "helena"


# ==================== DECAY FUNCTION TESTS ====================


class TestExponentialDecay:
    """Tests for ExponentialDecay function."""

    def test_no_decay_at_zero_days(self):
        """Test that there's no decay at day 0."""
        decay = ExponentialDecay(half_life_days=30.0)
        result = decay.compute_decay(0)
        assert result == 1.0

    def test_half_decay_at_half_life(self):
        """Test that decay is 0.5 at half-life."""
        decay = ExponentialDecay(half_life_days=30.0)
        result = decay.compute_decay(30.0)
        assert abs(result - 0.5) < 0.001

    def test_quarter_decay_at_two_half_lives(self):
        """Test that decay is 0.25 at two half-lives."""
        decay = ExponentialDecay(half_life_days=30.0)
        result = decay.compute_decay(60.0)
        assert abs(result - 0.25) < 0.001

    def test_custom_half_life(self):
        """Test custom half-life values."""
        decay = ExponentialDecay(half_life_days=7.0)
        result = decay.compute_decay(7.0)
        assert abs(result - 0.5) < 0.001

    def test_get_name(self):
        """Test decay function name."""
        decay = ExponentialDecay(half_life_days=30.0)
        assert "exponential" in decay.get_name()


class TestLinearDecay:
    """Tests for LinearDecay function."""

    def test_no_decay_at_zero_days(self):
        """Test that there's no decay at day 0."""
        decay = LinearDecay(decay_period_days=90.0, min_value=0.1)
        result = decay.compute_decay(0)
        assert result == 1.0

    def test_linear_decay_at_midpoint(self):
        """Test linear decay at midpoint."""
        decay = LinearDecay(decay_period_days=90.0, min_value=0.1)
        result = decay.compute_decay(45.0)
        # At midpoint: 1.0 - (45/90) * (1.0 - 0.1) = 1.0 - 0.5 * 0.9 = 0.55
        assert abs(result - 0.55) < 0.01

    def test_min_value_at_full_period(self):
        """Test that min value is reached at decay period."""
        decay = LinearDecay(decay_period_days=90.0, min_value=0.1)
        result = decay.compute_decay(90.0)
        assert result == 0.1

    def test_min_value_beyond_period(self):
        """Test that min value is maintained beyond decay period."""
        decay = LinearDecay(decay_period_days=90.0, min_value=0.1)
        result = decay.compute_decay(180.0)
        assert result == 0.1

    def test_get_name(self):
        """Test decay function name."""
        decay = LinearDecay()
        assert "linear" in decay.get_name()


class TestStepDecay:
    """Tests for StepDecay function."""

    def test_no_decay_before_first_step(self):
        """Test no decay before first step threshold."""
        decay = StepDecay()  # Default: [(30, 0.9), (60, 0.7), (90, 0.5), (180, 0.3)]
        result = decay.compute_decay(15)
        assert result == 1.0

    def test_decay_at_first_step(self):
        """Test decay at first step threshold."""
        decay = StepDecay()
        result = decay.compute_decay(30)
        assert result == 0.9

    def test_decay_between_steps(self):
        """Test decay between step thresholds."""
        decay = StepDecay()
        result = decay.compute_decay(45)
        assert result == 0.9  # Still at step 1 value

    def test_decay_at_second_step(self):
        """Test decay at second step threshold."""
        decay = StepDecay()
        result = decay.compute_decay(60)
        assert result == 0.7

    def test_decay_at_final_step(self):
        """Test decay at final step."""
        decay = StepDecay()
        result = decay.compute_decay(180)
        assert result == 0.3

    def test_decay_beyond_final_step(self):
        """Test decay beyond final step."""
        decay = StepDecay()
        result = decay.compute_decay(365)
        assert result == 0.3  # Stays at final step value

    def test_custom_steps(self):
        """Test custom step configuration."""
        custom_steps = [(7, 0.8), (14, 0.5), (21, 0.2)]
        decay = StepDecay(steps=custom_steps)

        assert decay.compute_decay(0) == 1.0
        assert decay.compute_decay(7) == 0.8
        assert decay.compute_decay(14) == 0.5
        assert decay.compute_decay(21) == 0.2

    def test_get_name(self):
        """Test decay function name."""
        decay = StepDecay()
        assert "step" in decay.get_name()


class TestNoDecay:
    """Tests for NoDecay function."""

    def test_no_decay_at_any_time(self):
        """Test that NoDecay always returns 1.0."""
        decay = NoDecay()

        assert decay.compute_decay(0) == 1.0
        assert decay.compute_decay(30) == 1.0
        assert decay.compute_decay(365) == 1.0
        assert decay.compute_decay(1000) == 1.0

    def test_get_name(self):
        """Test decay function name."""
        decay = NoDecay()
        assert decay.get_name() == "none"


# ==================== CONFIDENCE DECAYER TESTS ====================


class TestDecayResult:
    """Tests for DecayResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = DecayResult()
        assert result.items_processed == 0
        assert result.items_updated == 0
        assert result.items_pruned == 0
        assert result.avg_decay_applied == 0.0
        assert result.execution_time_ms == 0


class TestConfidenceDecayer:
    """Tests for ConfidenceDecayer."""

    @pytest.fixture
    def mock_storage_for_decayer(self):
        """Create mock storage with heuristics for decay testing."""
        storage = MagicMock()
        now = datetime.now(timezone.utc)
        old_30 = now - timedelta(days=30)
        old_60 = now - timedelta(days=60)

        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="helena",
                project_id="test",
                condition="condition1",
                strategy="strategy1",
                confidence=0.9,
                occurrence_count=10,
                success_count=9,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="h2",
                agent="helena",
                project_id="test",
                condition="condition2",
                strategy="strategy2",
                confidence=0.8,
                occurrence_count=10,
                success_count=8,
                last_validated=old_30,  # 30 days old
                created_at=old_30,
            ),
            Heuristic(
                id="h3",
                agent="helena",
                project_id="test",
                condition="condition3",
                strategy="strategy3",
                confidence=0.6,
                occurrence_count=10,
                success_count=6,
                last_validated=old_60,  # 60 days old
                created_at=old_60,
            ),
        ]

        storage.get_domain_knowledge.return_value = []
        storage.update_heuristic_confidence.return_value = True
        storage.update_knowledge_confidence.return_value = True
        storage.delete_heuristic.return_value = True
        storage.delete_domain_knowledge.return_value = True

        return storage

    def test_dry_run_no_changes(self, mock_storage_for_decayer):
        """Test that dry run doesn't modify storage."""
        decayer = ConfidenceDecayer(
            mock_storage_for_decayer,
            decay_function=ExponentialDecay(half_life_days=30.0),
        )

        result = decayer.apply_decay("test", dry_run=True)

        mock_storage_for_decayer.update_heuristic_confidence.assert_not_called()
        mock_storage_for_decayer.delete_heuristic.assert_not_called()
        assert result.items_processed > 0

    def test_apply_decay_updates_confidence(self, mock_storage_for_decayer):
        """Test that decay updates confidence values."""
        decayer = ConfidenceDecayer(
            mock_storage_for_decayer,
            decay_function=ExponentialDecay(half_life_days=30.0),
        )

        result = decayer.apply_decay("test", dry_run=False)

        # Should have processed items
        assert result.items_processed > 0
        # h2 is 30 days old, should be updated
        mock_storage_for_decayer.update_heuristic_confidence.assert_called()

    def test_prune_below_threshold(self, mock_storage_for_decayer):
        """Test that items below threshold are pruned."""
        decayer = ConfidenceDecayer(
            mock_storage_for_decayer,
            decay_function=ExponentialDecay(half_life_days=30.0),
            prune_below_confidence=0.5,
        )

        # Modify h3 to have low decayed confidence
        mock_storage_for_decayer.get_heuristics.return_value[2].confidence = 0.3

        result = decayer.apply_decay("test", dry_run=False)

        # At least the h3 with decayed confidence < 0.5 should be pruned
        assert result.items_processed > 0

    def test_agent_filter(self, mock_storage_for_decayer):
        """Test filtering by agent."""
        decayer = ConfidenceDecayer(
            mock_storage_for_decayer,
            decay_function=ExponentialDecay(half_life_days=30.0),
        )

        result = decayer.apply_decay("test", agent="helena", dry_run=True)

        # Should only query helena's data
        mock_storage_for_decayer.get_heuristics.assert_called_with(
            project_id="test", agent="helena", top_k=10000, min_confidence=0.0
        )


# ==================== MEMORY HEALTH MONITOR TESTS ====================


class TestHealthThresholds:
    """Tests for HealthThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = HealthThresholds()

        assert thresholds.max_total_items_warning == 5000
        assert thresholds.max_total_items_critical == 10000
        assert thresholds.max_stale_percentage_warning == 0.3
        assert thresholds.min_avg_confidence_warning == 0.5


class TestHealthAlert:
    """Tests for HealthAlert dataclass."""

    def test_creation(self):
        """Test creating a health alert."""
        alert = HealthAlert(
            level="warning",
            category="growth_rate",
            message="High growth rate detected",
            current_value=600,
            threshold=500,
        )

        assert alert.level == "warning"
        assert alert.category == "growth_rate"
        assert alert.current_value == 600


class TestMemoryHealthMetrics:
    """Tests for MemoryHealthMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = MemoryHealthMetrics()

        assert metrics.total_items == 0
        assert metrics.heuristic_count == 0
        assert metrics.outcome_count == 0
        assert metrics.avg_heuristic_confidence == 0.0


class TestMemoryHealthMonitor:
    """Tests for MemoryHealthMonitor."""

    @pytest.fixture
    def mock_storage_for_monitor(self):
        """Create mock storage for health monitoring."""
        storage = MagicMock()
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=120)

        storage.get_heuristics.return_value = [
            Heuristic(
                id="h1",
                agent="helena",
                project_id="test",
                condition="c1",
                strategy="s1",
                confidence=0.8,
                occurrence_count=10,
                success_count=8,
                last_validated=now,
                created_at=now,
            ),
            Heuristic(
                id="h2",
                agent="helena",
                project_id="test",
                condition="c2",
                strategy="s2",
                confidence=0.3,
                occurrence_count=5,
                success_count=2,
                last_validated=old,  # Stale
                created_at=old,
            ),
        ]

        storage.get_outcomes.return_value = [
            Outcome(
                id="o1",
                agent="helena",
                project_id="test",
                task_type="testing",
                task_description="Task 1",
                success=True,
                strategy_used="strategy1",
                timestamp=now,
            ),
        ]

        storage.get_domain_knowledge.return_value = []
        storage.get_anti_patterns.return_value = []

        return storage

    def test_collect_metrics(self, mock_storage_for_monitor):
        """Test collecting health metrics."""
        monitor = MemoryHealthMonitor(mock_storage_for_monitor)

        metrics = monitor.collect_metrics("test")

        assert metrics.heuristic_count == 2
        assert metrics.outcome_count == 1
        assert metrics.total_items == 3
        # Average confidence: (0.8 + 0.3) / 2 = 0.55
        assert abs(metrics.avg_heuristic_confidence - 0.55) < 0.01

    def test_check_health_no_alerts(self, mock_storage_for_monitor):
        """Test health check with no alerts."""
        thresholds = HealthThresholds(
            max_total_items_warning=1000,
            max_total_items_critical=2000,
            max_stale_percentage_warning=0.8,  # High threshold
            max_stale_percentage_critical=0.9,
            min_avg_confidence_warning=0.1,  # Low threshold
            min_avg_confidence_critical=0.05,
        )
        monitor = MemoryHealthMonitor(mock_storage_for_monitor, thresholds)

        alerts = monitor.check_health("test")

        # With lenient thresholds, should have no alerts
        assert len(alerts) == 0

    def test_check_health_generates_alerts(self, mock_storage_for_monitor):
        """Test health check generates alerts for violations."""
        thresholds = HealthThresholds(
            max_total_items_warning=1,  # Very low - will trigger alert
            max_total_items_critical=2,
            max_stale_percentage_warning=0.1,  # Very low - will trigger alert
            max_stale_percentage_critical=0.2,
            min_avg_confidence_warning=0.9,  # Very high - will trigger alert
            min_avg_confidence_critical=0.8,
        )
        monitor = MemoryHealthMonitor(mock_storage_for_monitor, thresholds)

        alerts = monitor.check_health("test")

        # Should have multiple alerts
        assert len(alerts) > 0
        alert_categories = [a.category for a in alerts]
        assert "total_items" in alert_categories or "staleness" in alert_categories

    def test_alert_handler(self, mock_storage_for_monitor):
        """Test alert handler callback."""
        thresholds = HealthThresholds(max_total_items_warning=1, max_total_items_critical=2)  # Will trigger alert
        monitor = MemoryHealthMonitor(mock_storage_for_monitor, thresholds)

        received_alerts = []
        monitor.add_alert_handler(lambda a: received_alerts.append(a))

        monitor.check_health("test")

        assert len(received_alerts) > 0


# ==================== CLEANUP SCHEDULER TESTS ====================


class TestCleanupJob:
    """Tests for CleanupJob dataclass."""

    def test_creation(self):
        """Test creating a cleanup job."""
        job = CleanupJob(
            name="daily_cleanup",
            project_id="test",
            interval_hours=24.0,
            agent="helena",
            apply_decay=True,
            enabled=True,
        )

        assert job.name == "daily_cleanup"
        assert job.interval_hours == 24.0
        assert job.apply_decay is True


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_creation(self):
        """Test creating a cleanup result."""
        now = datetime.now(timezone.utc)
        result = CleanupResult(
            job_name="test_job",
            project_id="test",
            started_at=now,
            completed_at=now,
            prune_summary=PruneSummary(total_pruned=5),
            decay_result=DecayResult(items_processed=10),
            success=True,
        )

        assert result.success is True
        assert result.prune_summary.total_pruned == 5
        assert result.decay_result.items_processed == 10


class TestCleanupScheduler:
    """Tests for CleanupScheduler."""

    @pytest.fixture
    def mock_storage_for_scheduler(self):
        """Create mock storage for scheduler testing."""
        storage = MagicMock()
        now = datetime.now(timezone.utc)

        storage.get_heuristics.return_value = []
        storage.get_outcomes.return_value = []
        storage.get_domain_knowledge.return_value = []
        storage.get_anti_patterns.return_value = []
        storage.get_stats.return_value = {"agents": []}
        storage.delete_outcomes_older_than.return_value = 0
        storage.delete_low_confidence_heuristics.return_value = 0

        return storage

    def test_register_job(self, mock_storage_for_scheduler):
        """Test registering a cleanup job."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(
            name="test_job",
            project_id="test",
            interval_hours=1.0,
        )
        scheduler.register_job(job)

        assert "test_job" in scheduler._jobs

    def test_unregister_job(self, mock_storage_for_scheduler):
        """Test unregistering a cleanup job."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(name="test_job", project_id="test", interval_hours=1.0)
        scheduler.register_job(job)
        scheduler.unregister_job("test_job")

        assert "test_job" not in scheduler._jobs

    def test_run_job(self, mock_storage_for_scheduler):
        """Test running a cleanup job."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(
            name="test_job",
            project_id="test",
            interval_hours=1.0,
            apply_decay=False,  # Skip decay for simpler test
        )
        scheduler.register_job(job)

        result = scheduler.run_job("test_job")

        assert result.job_name == "test_job"
        assert result.success is True

    def test_run_nonexistent_job(self, mock_storage_for_scheduler):
        """Test running a non-existent job raises error."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        with pytest.raises(ValueError) as exc_info:
            scheduler.run_job("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_dry_run(self, mock_storage_for_scheduler):
        """Test dry run mode."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(name="test_job", project_id="test", interval_hours=1.0)
        scheduler.register_job(job)

        result = scheduler.run_job("test_job", dry_run=True)

        assert result.success is True
        # In dry run, no actual deletions should happen
        mock_storage_for_scheduler.delete_heuristic.assert_not_called()

    def test_disabled_job_not_run(self, mock_storage_for_scheduler):
        """Test that disabled jobs are not run in run_all_due."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(
            name="disabled_job",
            project_id="test",
            interval_hours=0.0001,  # Very short interval
            enabled=False,
        )
        scheduler.register_job(job)

        # Wait a bit and run
        time.sleep(0.01)
        results = scheduler.run_all_due()

        # Disabled job should not produce results
        assert len([r for r in results if r.job_name == "disabled_job"]) == 0

    def test_get_jobs(self, mock_storage_for_scheduler):
        """Test getting all jobs."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        job = CleanupJob(name="test_job", project_id="test", interval_hours=24.0)
        scheduler.register_job(job)

        jobs = scheduler.get_jobs()

        assert len(jobs) == 1
        assert jobs[0]["name"] == "test_job"
        assert jobs[0]["enabled"] is True
        assert jobs[0]["interval_hours"] == 24.0

    def test_get_jobs_multiple(self, mock_storage_for_scheduler):
        """Test getting multiple jobs."""
        scheduler = CleanupScheduler(mock_storage_for_scheduler)

        scheduler.register_job(CleanupJob(name="job1", project_id="test1", interval_hours=1.0))
        scheduler.register_job(CleanupJob(name="job2", project_id="test2", interval_hours=2.0))

        jobs = scheduler.get_jobs()

        assert len(jobs) == 2
        job_names = [j["name"] for j in jobs]
        assert "job1" in job_names
        assert "job2" in job_names
