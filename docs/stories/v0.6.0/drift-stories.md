# Drift Detection - User Stories

**Epic:** Drift Detection
**Target Version:** 0.6.0
**Sprint:** 2

---

## DRIFT-001: Baseline Manager

**Story Points:** 5
**Priority:** P0 - Critical
**Dependencies:** Existing Observability

### User Story

**As a** system operator,
**I want** ALMA to maintain behavioral baselines for each agent,
**So that** we can detect when behavior deviates from expected patterns.

### Description

Implement a baseline management system that captures and maintains behavioral profiles for agents. Baselines should include strategy distributions, success rates, domain engagement patterns, and performance metrics.

### Acceptance Criteria

- [ ] Create baseline from historical data with configurable lookback period
- [ ] Capture strategy selection frequencies and success rates
- [ ] Capture task type distributions and success rates
- [ ] Capture domain engagement patterns
- [ ] Capture performance metrics (confidence, duration, success rate)
- [ ] Update baseline incrementally with exponential decay
- [ ] Store baselines in persistent storage
- [ ] Load baseline on agent initialization
- [ ] Support manual baseline reset
- [ ] Unit tests with >90% coverage

### Technical Design

```python
# alma/drift/baseline.py

@dataclass
class AgentBaseline:
    agent: str
    created_at: datetime
    updated_at: datetime

    # Strategy distributions
    strategy_frequencies: Dict[str, float]  # strategy -> frequency
    strategy_success_rates: Dict[str, float]  # strategy -> success_rate

    # Task type distributions
    task_type_frequencies: Dict[str, float]
    task_type_success_rates: Dict[str, float]

    # Domain engagement
    domain_frequencies: Dict[str, float]

    # Performance metrics
    avg_confidence: float
    avg_duration_ms: float
    overall_success_rate: float

    # Sample size for statistical significance
    total_observations: int

class BaselineManager:
    def __init__(
        self,
        storage: StorageBackend,
        decay_factor: float = 0.95,  # For exponential moving average
        min_observations: int = 100,  # Minimum for statistical significance
    ):
        pass

    def create_baseline(
        self,
        agent: str,
        lookback_period: timedelta = timedelta(days=30),
    ) -> AgentBaseline:
        """Create baseline from historical outcomes."""
        pass

    def update_baseline(
        self,
        agent: str,
        observation: Observation,
    ) -> AgentBaseline:
        """Incrementally update baseline with new observation."""
        pass

    def get_baseline(self, agent: str) -> Optional[AgentBaseline]:
        """Retrieve stored baseline."""
        pass

    def reset_baseline(self, agent: str) -> AgentBaseline:
        """Reset and rebuild baseline from scratch."""
        pass
```

### Storage Schema

```sql
CREATE TABLE agent_baselines (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL UNIQUE,
    strategy_frequencies JSONB,
    strategy_success_rates JSONB,
    task_type_frequencies JSONB,
    task_type_success_rates JSONB,
    domain_frequencies JSONB,
    avg_confidence FLOAT,
    avg_duration_ms FLOAT,
    overall_success_rate FLOAT,
    total_observations INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

---

## DRIFT-002: Statistical Analyzer

**Story Points:** 5
**Priority:** P0 - Critical
**Dependencies:** DRIFT-001

### User Story

**As a** system operator,
**I want** statistical tests to detect significant drift,
**So that** we distinguish real drift from normal variance.

### Description

Implement statistical analysis methods to detect significant behavioral changes. The analyzer should use multiple detection methods appropriate for different data types.

### Acceptance Criteria

- [ ] Implement Kullback-Leibler divergence for distribution comparison
- [ ] Implement Mann-Whitney U test for performance shifts
- [ ] Implement CUSUM for change point detection
- [ ] Implement Chi-squared test for categorical changes
- [ ] Return significance levels and confidence intervals
- [ ] Handle small sample sizes gracefully
- [ ] Provide interpretable drift metrics
- [ ] Unit tests with >90% coverage

### Technical Design

```python
# alma/drift/statistics.py

@dataclass
class DriftMetric:
    metric_name: str
    baseline_value: float
    current_value: float
    divergence: float  # Measure of difference
    p_value: float  # Statistical significance
    is_significant: bool
    direction: Literal["increase", "decrease", "unchanged"]
    confidence_interval: Tuple[float, float]

class StatisticalAnalyzer:
    def __init__(
        self,
        significance_level: float = 0.05,
        min_sample_size: int = 30,
    ):
        pass

    def analyze_distribution_drift(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
    ) -> DriftMetric:
        """
        Compare distributions using KL divergence.

        Returns metric with divergence score and significance.
        """
        pass

    def analyze_performance_drift(
        self,
        baseline_samples: List[float],
        current_samples: List[float],
    ) -> DriftMetric:
        """
        Detect performance shifts using Mann-Whitney U test.

        Non-parametric test that doesn't assume normal distribution.
        """
        pass

    def detect_change_point(
        self,
        time_series: List[Tuple[datetime, float]],
    ) -> Optional[datetime]:
        """
        Find when drift likely started using CUSUM algorithm.

        Returns estimated change point datetime.
        """
        pass

    def analyze_categorical_drift(
        self,
        baseline_counts: Dict[str, int],
        current_counts: Dict[str, int],
    ) -> DriftMetric:
        """
        Detect changes in categorical distributions using Chi-squared.
        """
        pass
```

### Thresholds

| Test | Threshold | Interpretation |
|------|-----------|----------------|
| KL Divergence | > 0.1 | Significant distribution change |
| Mann-Whitney | p < 0.05 | Significant performance shift |
| Chi-squared | p < 0.05 | Significant categorical change |

---

## DRIFT-003: Drift Classifier

**Story Points:** 3
**Priority:** P1 - High
**Dependencies:** DRIFT-002

### User Story

**As a** system operator,
**I want** drift to be classified by type and severity,
**So that** I can prioritize responses appropriately.

### Description

Implement a classifier that categorizes detected drift by type (behavioral, performance, domain, preference) and severity level.

### Acceptance Criteria

- [ ] Classify drift into types: behavioral, performance, domain, preference
- [ ] Assign severity levels: low, medium, high, critical
- [ ] Detect expected drift patterns (e.g., seasonal)
- [ ] Provide root cause hypothesis
- [ ] Recommend appropriate action
- [ ] Consider multiple drift signals for classification
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/drift/classifier.py

class DriftType(Enum):
    BEHAVIORAL = "behavioral"  # Strategy selection changes
    PERFORMANCE = "performance"  # Success rate/duration changes
    DOMAIN = "domain"  # Working outside expected domains
    PREFERENCE = "preference"  # Ignoring user preferences

class DriftSeverity(Enum):
    LOW = "low"  # Monitor only
    MEDIUM = "medium"  # Alert + investigate
    HIGH = "high"  # Alert + auto-correct if possible
    CRITICAL = "critical"  # Immediate human intervention

@dataclass
class DriftClassification:
    drift_type: DriftType
    severity: DriftSeverity
    is_expected: bool  # e.g., seasonal patterns
    root_cause_hypothesis: str
    recommended_action: str
    confidence: float
    supporting_metrics: List[DriftMetric]

class DriftClassifier:
    def __init__(
        self,
        severity_thresholds: Optional[Dict[DriftType, Dict[str, float]]] = None,
    ):
        pass

    def classify(
        self,
        drift_metrics: Dict[str, DriftMetric],
        context: Optional[DriftContext] = None,
    ) -> DriftClassification:
        pass
```

### Severity Thresholds (Default)

| Drift Type | Low | Medium | High | Critical |
|------------|-----|--------|------|----------|
| Behavioral | 0.1-0.2 KL | 0.2-0.4 KL | 0.4-0.6 KL | >0.6 KL |
| Performance | 5-10% | 10-20% | 20-40% | >40% |
| Domain | 1-2 violations | 3-5 violations | 6-10 | >10 |

---

## DRIFT-004: Drift Event Integration

**Story Points:** 2
**Priority:** P1 - High
**Dependencies:** DRIFT-003

### User Story

**As a** system integrator,
**I want** drift events emitted via the event system,
**So that** I can build alerting and monitoring dashboards.

### Description

Integrate drift detection with the existing event system for webhooks and monitoring.

### Acceptance Criteria

- [ ] Define `DriftEventType` enum
- [ ] Emit events for drift detection, escalation, correction
- [ ] Include full classification and metrics in payload
- [ ] Include baseline snapshot in payload
- [ ] Integrate with existing `WebhookManager`
- [ ] Unit tests for event emission

### Event Types

```python
class DriftEventType(Enum):
    DRIFT_DETECTED = "drift.detected"
    DRIFT_SEVERITY_ESCALATED = "drift.severity_escalated"
    DRIFT_AUTO_CORRECTED = "drift.auto_corrected"
    DRIFT_HUMAN_REVIEW_REQUIRED = "drift.human_review_required"
    BASELINE_UPDATED = "drift.baseline_updated"
    BASELINE_RESET = "drift.baseline_reset"
```

---

## DRIFT-005: Auto-Correction Engine

**Story Points:** 5
**Priority:** P1 - High
**Dependencies:** DRIFT-004, CONV-004 (Intervention Engine)

### User Story

**As a** system operator,
**I want** ALMA to automatically correct certain drift patterns,
**So that** minor issues are resolved without human intervention.

### Description

Implement an auto-correction system that can respond to detected drift with appropriate remediation actions.

### Acceptance Criteria

- [ ] Define correction strategies per drift type
- [ ] Integrate with Intervention Engine from convergence module
- [ ] Support configurable auto-correction policies
- [ ] Record all corrections in memory for learning
- [ ] Implement correction strategies:
  - Reset agent context
  - Refresh relevant memories
  - Adjust confidence thresholds
  - Escalate to supervisor
- [ ] Rate-limit auto-corrections
- [ ] Emit events for all corrections
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/drift/correction.py

class CorrectionStrategy(Enum):
    REFRESH_MEMORIES = "refresh_memories"
    RESET_CONTEXT = "reset_context"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    ESCALATE = "escalate"
    REBUILD_BASELINE = "rebuild_baseline"

@dataclass
class CorrectionPolicy:
    drift_type: DriftType
    max_severity_for_auto: DriftSeverity  # Auto-correct up to this
    strategy: CorrectionStrategy
    cooldown: timedelta  # Time between corrections

class AutoCorrectionEngine:
    def __init__(
        self,
        intervention_engine: InterventionEngine,
        policies: List[CorrectionPolicy],
        rate_limit: int = 3,  # corrections per hour per agent
    ):
        pass

    def evaluate_and_correct(
        self,
        classification: DriftClassification,
        agent: str,
    ) -> Optional[InterventionResult]:
        """Evaluate if auto-correction should apply and execute."""
        pass
```

---

## DRIFT-006: Drift Dashboard Metrics

**Story Points:** 3
**Priority:** P2 - Medium
**Dependencies:** DRIFT-001

### User Story

**As a** system operator,
**I want** drift metrics exposed for dashboards,
**So that** I can visualize agent health over time.

### Description

Expose drift-related metrics through the observability system for external dashboards.

### Acceptance Criteria

- [ ] Expose baseline age metric (time since last update)
- [ ] Expose current drift score per agent
- [ ] Expose drift event count by type
- [ ] Expose correction count and success rate
- [ ] Integrate with existing `ALMAMetrics`
- [ ] Provide Prometheus-compatible format
- [ ] Unit tests for metric collection

### Metrics

```python
# Gauge: alma_drift_score{agent, drift_type}
# Gauge: alma_baseline_age_seconds{agent}
# Counter: alma_drift_events_total{agent, drift_type, severity}
# Counter: alma_drift_corrections_total{agent, strategy, success}
# Histogram: alma_drift_detection_duration_seconds{agent}
```

---

## Sprint 2 Summary

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| DRIFT-001 | 5 | P0 | TODO |
| DRIFT-002 | 5 | P0 | TODO |
| DRIFT-003 | 3 | P1 | TODO |
| DRIFT-004 | 2 | P1 | TODO |
| DRIFT-005 | 5 | P1 | TODO |
| DRIFT-006 | 3 | P2 | TODO |
| **Total** | **23** | - | - |

---

*Created by: Aria (System Architect)*
*Date: 2026-01-29*
