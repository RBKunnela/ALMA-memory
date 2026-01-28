# ALMA Advanced Agentic Patterns Architecture

**Version:** 0.6.0-draft
**Author:** Aria (System Architect)
**Date:** 2026-01-29
**Status:** Design Phase

---

## Executive Summary

This document outlines the architectural design for three advanced agentic patterns that will elevate ALMA-memory to full maturity as an AI agent memory system. These patterns address the critical gap between ALMA's current "episodic memory with learning" capabilities and the robust orchestration, quality assurance, and behavioral stability mechanisms required for production-grade multi-agent systems.

### Current State (v0.5.1)

ALMA already implements:
- **Episodic Worker Design** - Outcomes → Heuristics learning cycle
- **External Workflow State** - SessionManager, ProgressTracker
- **Context Engineering** - MemorySlice with token budgeting
- **Worker Isolation** - MemoryScope for agent boundaries
- **MCP Integration** - 8 tools for Claude Code
- **Confidence Scoring** - Forward-looking strategy assessment

### Missing Patterns (Target v0.6.0)

| Pattern | Purpose | Business Value |
|---------|---------|----------------|
| **Two-Tier Orchestration** | Supervisor/worker hierarchy | Scalable multi-agent coordination |
| **Convergence Evaluators** | Quality gates and loop detection | Prevent infinite loops and poor outputs |
| **Drift Detection** | Behavioral stability monitoring | Long-term reliability and consistency |

---

## Pattern 1: Two-Tier Orchestration Layer

### 1.1 Concept Overview

Two-tier orchestration introduces a **Supervisor** layer that manages multiple **Worker** agents. The Supervisor:
1. Decomposes complex tasks into subtasks
2. Assigns subtasks to specialized workers
3. Aggregates and validates worker outputs
4. Handles failures and re-routing

```
                    ┌─────────────────────────────────────────┐
                    │           ORCHESTRATION TIER            │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │          Supervisor              │   │
                    │  │  ┌─────────────────────────┐    │   │
                    │  │  │ TaskDecomposer          │    │   │
                    │  │  │ WorkerRouter            │    │   │
                    │  │  │ ResultAggregator        │    │   │
                    │  │  │ FailureHandler          │    │   │
                    │  │  └─────────────────────────┘    │   │
                    │  └─────────────────────────────────┘   │
                    │              ▲         │               │
                    │              │         ▼               │
                    │  ┌───────────┴─────────┴──────────┐   │
                    │  │        Worker Pool             │   │
                    │  │  ┌────────┐ ┌────────┐        │   │
                    │  │  │Worker A│ │Worker B│ ...    │   │
                    │  │  │(Helena)│ │(Victor)│        │   │
                    │  │  └────────┘ └────────┘        │   │
                    │  └────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │              ALMA MEMORY                │
                    │  ┌──────────────────────────────────┐  │
                    │  │ Shared Memory (with scopes)      │  │
                    │  │ - Orchestration outcomes          │  │
                    │  │ - Routing decisions               │  │
                    │  │ - Worker performance metrics      │  │
                    │  └──────────────────────────────────┘  │
                    └─────────────────────────────────────────┘
```

### 1.2 Core Components

#### 1.2.1 Supervisor (`alma/orchestration/supervisor.py`)

```python
@dataclass
class Supervisor:
    """
    Orchestrates multi-agent task execution.

    The Supervisor maintains its own MemoryScope focused on:
    - Task decomposition patterns
    - Worker routing decisions
    - Aggregation strategies
    """
    name: str
    alma: ALMA
    workers: Dict[str, Worker]
    decomposer: TaskDecomposer
    router: WorkerRouter
    aggregator: ResultAggregator

    async def execute(
        self,
        task: ComplexTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Execute a complex task through worker delegation.

        1. Retrieve orchestration memories
        2. Decompose into subtasks
        3. Route subtasks to workers
        4. Monitor execution
        5. Aggregate results
        6. Learn from orchestration outcome
        """
        pass
```

#### 1.2.2 Worker Interface (`alma/orchestration/worker.py`)

```python
@dataclass
class Worker:
    """
    A specialized agent that executes subtasks.

    Workers have their own MemoryScope and can learn
    from task outcomes within their domain.
    """
    name: str
    capabilities: List[str]  # What this worker can do
    alma: ALMA
    scope: MemoryScope

    async def execute(
        self,
        subtask: Subtask,
        context: WorkerContext,
    ) -> WorkerResult:
        """Execute a subtask within this worker's domain."""
        pass

    def can_handle(self, subtask: Subtask) -> float:
        """Return confidence score for handling this subtask."""
        pass
```

#### 1.2.3 Task Decomposition (`alma/orchestration/decomposer.py`)

```python
class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks.

    Uses ALMA memory to learn optimal decomposition strategies:
    - Which task types decompose well
    - Optimal granularity for different contexts
    - Dependencies between subtasks
    """

    def decompose(
        self,
        task: ComplexTask,
        available_workers: List[Worker],
    ) -> List[Subtask]:
        """
        Decompose task using memory-informed strategies.

        Returns subtasks with:
        - Dependencies (DAG structure)
        - Estimated complexity
        - Suggested worker capabilities
        """
        pass
```

#### 1.2.4 Worker Router (`alma/orchestration/router.py`)

```python
class WorkerRouter:
    """
    Routes subtasks to optimal workers.

    Uses ALMA memory to learn:
    - Which workers excel at which task types
    - Load balancing strategies
    - Fallback routes when primary worker fails
    """

    def route(
        self,
        subtask: Subtask,
        workers: List[Worker],
        context: RoutingContext,
    ) -> RoutingDecision:
        """
        Select optimal worker for subtask.

        Considers:
        - Worker capabilities and current load
        - Historical performance from memory
        - Subtask requirements
        """
        pass
```

### 1.3 Memory Integration

The orchestration layer introduces new memory types:

```python
@dataclass
class OrchestrationOutcome:
    """Memory of orchestration decisions and their results."""
    id: str
    supervisor: str
    task_type: str
    decomposition_strategy: str
    routing_decisions: List[Dict]
    worker_results: List[Dict]
    aggregation_strategy: str
    overall_success: bool
    total_duration_ms: int
    timestamp: datetime

@dataclass
class RoutingHeuristic:
    """Learned rules for worker routing."""
    id: str
    condition: str  # "When task involves API testing"
    routing_strategy: str  # "Prefer Victor over Helena"
    confidence: float
    success_rate: float
```

### 1.4 New MemoryScope Extensions

```python
@dataclass
class MemoryScope:
    # Existing fields...

    # New orchestration fields
    can_orchestrate: bool = False  # Can this agent supervise others?
    subordinate_to: Optional[str] = None  # Who supervises this agent?
    can_delegate_to: List[str] = field(default_factory=list)
```

---

## Pattern 2: Convergence Evaluators (Ralph Wiggum Pattern)

### 2.1 Concept Overview

The "Ralph Wiggum Pattern" (named humorously after The Simpsons character known for non-sequiturs) detects when agents are:
1. **Stuck in loops** - Repeating the same actions without progress
2. **Producing poor quality** - Outputs that don't meet criteria
3. **Failing to converge** - Not making progress toward goals
4. **Generating nonsense** - Outputs that are syntactically valid but semantically wrong

```
                    ┌─────────────────────────────────────────┐
                    │        CONVERGENCE EVALUATION           │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │      ConvergenceMonitor          │   │
                    │  │  ┌─────────┐ ┌─────────────┐    │   │
                    │  │  │ Loop    │ │ Quality     │    │   │
                    │  │  │Detector │ │ Evaluator   │    │   │
                    │  │  └────┬────┘ └──────┬──────┘    │   │
                    │  │       │              │          │   │
                    │  │  ┌────┴──────────────┴────┐    │   │
                    │  │  │   Progress Tracker     │    │   │
                    │  │  └────────────┬───────────┘    │   │
                    │  │               │                 │   │
                    │  │  ┌────────────┴───────────┐    │   │
                    │  │  │   Intervention Engine   │    │   │
                    │  │  └────────────────────────┘    │   │
                    │  └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                    ┌──────────┐            ┌──────────┐
                    │  ALERT   │            │  MEMORY  │
                    │ Webhook  │            │  Update  │
                    └──────────┘            └──────────┘
```

### 2.2 Core Components

#### 2.2.1 Convergence Monitor (`alma/convergence/monitor.py`)

```python
@dataclass
class ConvergenceMonitor:
    """
    Monitors agent execution for convergence issues.

    Runs as a passive observer that can:
    - Raise alerts
    - Trigger interventions
    - Store convergence anti-patterns
    """
    loop_detector: LoopDetector
    quality_evaluator: QualityEvaluator
    progress_tracker: ProgressAnalyzer
    intervention_engine: InterventionEngine

    def observe(
        self,
        agent: str,
        action: AgentAction,
        context: ExecutionContext,
    ) -> ConvergenceSignal:
        """
        Observe an agent action and evaluate convergence.

        Returns a signal with:
        - is_converging: bool
        - confidence: float
        - issues: List[ConvergenceIssue]
        - recommended_intervention: Optional[Intervention]
        """
        pass
```

#### 2.2.2 Loop Detector (`alma/convergence/loop_detector.py`)

```python
class LoopDetector:
    """
    Detects when agents are stuck in repetitive loops.

    Uses multiple detection strategies:
    - Exact action repetition
    - Semantic similarity of outputs
    - State cycling (returning to previous states)
    - Oscillation patterns (A→B→A→B)
    """

    def detect_loop(
        self,
        action_history: List[AgentAction],
        window_size: int = 10,
    ) -> Optional[LoopDetection]:
        """
        Analyze action history for loop patterns.

        Returns:
            LoopDetection with:
            - loop_type: exact | semantic | state_cycle | oscillation
            - loop_start_index: int
            - loop_length: int
            - confidence: float
        """
        pass

    def semantic_similarity(
        self,
        action1: AgentAction,
        action2: AgentAction,
    ) -> float:
        """Use embeddings to detect semantically similar actions."""
        pass
```

#### 2.2.3 Quality Evaluator (`alma/convergence/quality.py`)

```python
class QualityEvaluator:
    """
    Evaluates output quality against expected standards.

    Uses configurable quality gates:
    - Schema validation
    - Semantic coherence
    - Completeness checks
    - Domain-specific rules
    """

    def evaluate(
        self,
        output: Any,
        expected_schema: Optional[Schema] = None,
        quality_gates: Optional[List[QualityGate]] = None,
    ) -> QualityScore:
        """
        Evaluate output quality.

        Returns:
            QualityScore with:
            - overall_score: float (0-1)
            - gate_results: Dict[str, GateResult]
            - issues: List[QualityIssue]
        """
        pass

@dataclass
class QualityGate:
    """A configurable quality check."""
    name: str
    evaluator: Callable[[Any], GateResult]
    weight: float = 1.0
    blocking: bool = True  # Fail if this gate fails
```

#### 2.2.4 Progress Analyzer (`alma/convergence/progress.py`)

```python
class ProgressAnalyzer:
    """
    Analyzes whether agent is making meaningful progress.

    Integrates with ALMA's ProgressTracker but adds:
    - Progress velocity tracking
    - Goal distance estimation
    - Stall detection
    """

    def analyze_progress(
        self,
        work_items: List[WorkItem],
        time_window: timedelta,
    ) -> ProgressAnalysis:
        """
        Analyze progress over a time window.

        Returns:
            ProgressAnalysis with:
            - velocity: float (items/hour)
            - acceleration: float (velocity change)
            - estimated_completion: Optional[datetime]
            - is_stalled: bool
            - stall_duration: Optional[timedelta]
        """
        pass
```

#### 2.2.5 Intervention Engine (`alma/convergence/intervention.py`)

```python
class InterventionEngine:
    """
    Handles interventions when convergence issues are detected.

    Intervention types:
    - BREAK_LOOP: Force agent to try different approach
    - ESCALATE: Send to supervisor
    - RESET_CONTEXT: Clear agent's working memory
    - HUMAN_REVIEW: Request human intervention
    """

    def intervene(
        self,
        agent: str,
        issue: ConvergenceIssue,
        context: ExecutionContext,
    ) -> InterventionResult:
        """
        Execute intervention for convergence issue.

        The intervention is recorded in ALMA memory
        for future learning.
        """
        pass

@dataclass
class Intervention:
    type: InterventionType
    reason: str
    suggested_action: str
    priority: int  # 1=highest
    auto_execute: bool = False
```

### 2.3 Memory Integration

```python
@dataclass
class ConvergenceAntiPattern:
    """Memory of convergence failures for future avoidance."""
    id: str
    agent: str
    issue_type: str  # loop | quality | stall | nonsense
    trigger_conditions: Dict[str, Any]
    action_sequence: List[str]
    intervention_used: str
    intervention_success: bool
    timestamp: datetime

@dataclass
class QualityBaseline:
    """Learned quality expectations per task type."""
    id: str
    agent: str
    task_type: str
    expected_scores: Dict[str, float]  # gate -> expected_score
    historical_distribution: Dict[str, List[float]]
```

### 2.4 Event Integration

```python
class ConvergenceEventType(Enum):
    LOOP_DETECTED = "convergence.loop_detected"
    QUALITY_BELOW_THRESHOLD = "convergence.quality_below_threshold"
    PROGRESS_STALLED = "convergence.progress_stalled"
    INTERVENTION_TRIGGERED = "convergence.intervention_triggered"
    INTERVENTION_SUCCEEDED = "convergence.intervention_succeeded"
    INTERVENTION_FAILED = "convergence.intervention_failed"
```

---

## Pattern 3: Drift Detection

### 3.1 Concept Overview

Drift detection monitors agent behavior over time to detect:
1. **Behavioral Drift** - Agent's strategy choices changing unexpectedly
2. **Performance Drift** - Gradual degradation in success rates
3. **Domain Drift** - Agent wandering outside its intended scope
4. **Preference Drift** - User preferences being ignored over time

```
                    ┌─────────────────────────────────────────┐
                    │           DRIFT DETECTION               │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │       DriftDetector              │   │
                    │  │  ┌────────────┬────────────┐    │   │
                    │  │  │ Baseline   │ Current    │    │   │
                    │  │  │ Profile    │ Behavior   │    │   │
                    │  │  └─────┬──────┴─────┬──────┘    │   │
                    │  │        │            │           │   │
                    │  │  ┌─────┴────────────┴─────┐    │   │
                    │  │  │   Statistical Analyzer  │    │   │
                    │  │  └────────────┬───────────┘    │   │
                    │  │               │                 │   │
                    │  │  ┌────────────┴───────────┐    │   │
                    │  │  │    Drift Classifier    │    │   │
                    │  │  └────────────────────────┘    │   │
                    │  └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                    ┌──────────┐            ┌──────────┐
                    │  ALERT   │            │  AUTO    │
                    │  System  │            │ CORRECT  │
                    └──────────┘            └──────────┘
```

### 3.2 Core Components

#### 3.2.1 Drift Detector (`alma/drift/detector.py`)

```python
@dataclass
class DriftDetector:
    """
    Monitors agent behavior for drift from expected patterns.

    Maintains behavioral baselines and compares current
    behavior against historical norms.
    """
    baseline_manager: BaselineManager
    behavior_sampler: BehaviorSampler
    statistical_analyzer: StatisticalAnalyzer
    drift_classifier: DriftClassifier

    def detect(
        self,
        agent: str,
        time_window: timedelta = timedelta(days=7),
    ) -> DriftReport:
        """
        Analyze agent behavior for drift.

        Returns:
            DriftReport with:
            - behavioral_drift: Optional[DriftSignal]
            - performance_drift: Optional[DriftSignal]
            - domain_drift: Optional[DriftSignal]
            - preference_drift: Optional[DriftSignal]
            - overall_health_score: float
            - recommendations: List[str]
        """
        pass
```

#### 3.2.2 Baseline Manager (`alma/drift/baseline.py`)

```python
class BaselineManager:
    """
    Manages behavioral baselines for agents.

    Baselines capture:
    - Strategy selection distributions
    - Success rates by task type
    - Domain engagement patterns
    - Preference adherence rates
    """

    def create_baseline(
        self,
        agent: str,
        lookback_period: timedelta,
    ) -> AgentBaseline:
        """Create baseline from historical data."""
        pass

    def update_baseline(
        self,
        agent: str,
        new_observations: List[Observation],
        decay_factor: float = 0.95,
    ) -> AgentBaseline:
        """Update baseline with exponential decay."""
        pass

@dataclass
class AgentBaseline:
    """Behavioral baseline for an agent."""
    agent: str
    created_at: datetime
    updated_at: datetime

    # Strategy distributions
    strategy_frequencies: Dict[str, float]
    strategy_success_rates: Dict[str, float]

    # Task type distributions
    task_type_frequencies: Dict[str, float]
    task_type_success_rates: Dict[str, float]

    # Domain engagement
    domain_frequencies: Dict[str, float]

    # Performance metrics
    avg_confidence: float
    avg_duration_ms: float
    success_rate: float
```

#### 3.2.3 Statistical Analyzer (`alma/drift/statistics.py`)

```python
class StatisticalAnalyzer:
    """
    Performs statistical tests to detect significant drift.

    Uses multiple detection methods:
    - Kullback-Leibler divergence for distributions
    - Mann-Whitney U test for performance shifts
    - CUSUM for detecting change points
    - Chi-squared for categorical changes
    """

    def analyze_distribution_drift(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
    ) -> DriftMetric:
        """Compare distributions using KL divergence."""
        pass

    def analyze_performance_drift(
        self,
        baseline_samples: List[float],
        current_samples: List[float],
    ) -> DriftMetric:
        """Detect performance shifts using statistical tests."""
        pass

    def detect_change_point(
        self,
        time_series: List[Tuple[datetime, float]],
    ) -> Optional[datetime]:
        """Find when drift likely started using CUSUM."""
        pass
```

#### 3.2.4 Drift Classifier (`alma/drift/classifier.py`)

```python
class DriftClassifier:
    """
    Classifies detected drift by type and severity.
    """

    def classify(
        self,
        drift_metrics: Dict[str, DriftMetric],
        context: DriftContext,
    ) -> DriftClassification:
        """
        Classify drift type and severity.

        Returns:
            DriftClassification with:
            - drift_type: behavioral | performance | domain | preference
            - severity: low | medium | high | critical
            - is_expected: bool (e.g., seasonal patterns)
            - root_cause_hypothesis: str
            - recommended_action: str
        """
        pass

class DriftSeverity(Enum):
    LOW = "low"  # Monitor only
    MEDIUM = "medium"  # Alert + investigate
    HIGH = "high"  # Alert + auto-correct if possible
    CRITICAL = "critical"  # Immediate human intervention
```

### 3.3 Memory Integration

```python
@dataclass
class DriftEvent:
    """Record of detected drift for historical analysis."""
    id: str
    agent: str
    drift_type: str
    severity: str
    metrics: Dict[str, float]
    detection_time: datetime
    drift_start_estimate: Optional[datetime]
    root_cause: Optional[str]
    resolution: Optional[str]
    resolved_at: Optional[datetime]

@dataclass
class CorrectionHeuristic:
    """Learned rules for drift correction."""
    id: str
    drift_pattern: str  # "strategy_drift when X"
    correction_strategy: str
    success_rate: float
    auto_apply: bool
```

### 3.4 Integration with Existing Systems

```python
# Drift detection integrates with observability
class DriftMetricsCollector:
    """Collects metrics for drift detection."""

    def record_agent_action(
        self,
        agent: str,
        action_type: str,
        strategy: str,
        domain: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record action for drift analysis."""
        pass

# Extends the existing event system
class DriftEventType(Enum):
    DRIFT_DETECTED = "drift.detected"
    DRIFT_SEVERITY_ESCALATED = "drift.severity_escalated"
    DRIFT_AUTO_CORRECTED = "drift.auto_corrected"
    DRIFT_HUMAN_REVIEW_REQUIRED = "drift.human_review_required"
    BASELINE_UPDATED = "drift.baseline_updated"
```

---

## Implementation Roadmap

### Phase 1: Convergence Evaluators (Sprint 1)
**Priority: HIGHEST** - Prevents runaway costs and poor outputs

| Story ID | Title | Points | Dependencies |
|----------|-------|--------|--------------|
| CONV-001 | Loop Detector Core | 5 | None |
| CONV-002 | Quality Evaluator Framework | 5 | None |
| CONV-003 | Progress Analyzer Integration | 3 | Existing ProgressTracker |
| CONV-004 | Intervention Engine | 5 | CONV-001, CONV-002 |
| CONV-005 | Convergence Event Types | 2 | Existing Event System |
| CONV-006 | Convergence Anti-Pattern Memory | 3 | CONV-004 |

### Phase 2: Drift Detection (Sprint 2)
**Priority: HIGH** - Critical for long-term reliability

| Story ID | Title | Points | Dependencies |
|----------|-------|--------|--------------|
| DRIFT-001 | Baseline Manager | 5 | Existing Observability |
| DRIFT-002 | Statistical Analyzer | 5 | DRIFT-001 |
| DRIFT-003 | Drift Classifier | 3 | DRIFT-002 |
| DRIFT-004 | Drift Event Integration | 2 | DRIFT-003 |
| DRIFT-005 | Auto-Correction Engine | 5 | DRIFT-004, CONV-004 |
| DRIFT-006 | Drift Dashboard Metrics | 3 | DRIFT-001 |

### Phase 3: Two-Tier Orchestration (Sprint 3-4)
**Priority: MEDIUM** - Enables multi-agent scaling

| Story ID | Title | Points | Dependencies |
|----------|-------|--------|--------------|
| ORCH-001 | Worker Interface | 3 | None |
| ORCH-002 | Supervisor Core | 8 | ORCH-001 |
| ORCH-003 | Task Decomposer | 5 | ORCH-002 |
| ORCH-004 | Worker Router | 5 | ORCH-001, ORCH-002 |
| ORCH-005 | Result Aggregator | 5 | ORCH-002 |
| ORCH-006 | Orchestration Memory Types | 3 | ORCH-002 |
| ORCH-007 | Failure Handler & Recovery | 5 | ORCH-002, CONV-004 |
| ORCH-008 | MCP Orchestration Tools | 5 | ORCH-002 |

---

## Technical Considerations

### Database Schema Extensions

```sql
-- Convergence tracking
CREATE TABLE convergence_events (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    issue_type TEXT NOT NULL,  -- loop, quality, stall
    severity TEXT NOT NULL,
    action_history JSONB,
    intervention_type TEXT,
    intervention_success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE
);

-- Drift tracking
CREATE TABLE agent_baselines (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL UNIQUE,
    strategy_frequencies JSONB,
    task_type_frequencies JSONB,
    domain_frequencies JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE drift_events (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    drift_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    metrics JSONB,
    detected_at TIMESTAMP WITH TIME ZONE,
    drift_start_estimate TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Orchestration
CREATE TABLE orchestration_outcomes (
    id TEXT PRIMARY KEY,
    supervisor TEXT NOT NULL,
    task_type TEXT NOT NULL,
    decomposition_strategy TEXT,
    routing_decisions JSONB,
    worker_results JSONB,
    overall_success BOOLEAN,
    total_duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE
);
```

### API Extensions

```python
# New MCP tools for advanced patterns

# Convergence
"alma_check_convergence"  # Check if agent is converging
"alma_report_loop"        # Report suspected loop
"alma_force_intervention" # Force intervention

# Drift
"alma_check_drift"        # Get drift report
"alma_update_baseline"    # Force baseline update
"alma_get_health_score"   # Overall agent health

# Orchestration
"alma_delegate_task"      # Supervisor delegates to worker
"alma_report_subtask"     # Worker reports subtask result
"alma_get_orchestration_status"  # Check orchestration status
```

### Performance Considerations

1. **Loop Detection**: Use rolling window with O(n) sliding window algorithm
2. **Baseline Updates**: Use exponential moving average (constant memory)
3. **Drift Statistics**: Pre-compute distributions, update incrementally
4. **Orchestration**: Async execution with configurable parallelism

### Security Considerations

1. **Orchestration**: Validate supervisor has permission to delegate
2. **Intervention**: Rate-limit auto-interventions to prevent amplification
3. **Drift Correction**: Log all auto-corrections for audit

---

## Success Metrics

### Convergence Evaluators
- Loop detection accuracy > 95%
- False positive rate < 5%
- Mean time to detect loop < 10 iterations
- Mean intervention success rate > 80%

### Drift Detection
- Drift detection within 24 hours of onset
- False positive rate < 10%
- Auto-correction success rate > 70%
- Performance degradation detection sensitivity > 5%

### Orchestration
- Task decomposition accuracy > 90%
- Worker routing efficiency > 85% optimal assignment
- End-to-end latency < 2x single-worker latency
- Failure recovery success rate > 95%

---

## Appendix: Story Templates

### Story Template: CONV-001 Loop Detector Core

**Title:** Implement Loop Detection Core System

**As a** system operator,
**I want** ALMA to detect when agents are stuck in repetitive loops,
**So that** we can prevent infinite execution and wasted resources.

**Acceptance Criteria:**
1. Detect exact action repetition within sliding window
2. Detect semantic similarity loops using embeddings
3. Detect state cycling patterns (A→B→A→B)
4. Provide confidence score for loop detection
5. Support configurable detection thresholds
6. Integrate with existing observability metrics

**Technical Notes:**
- Use embedding similarity threshold of 0.85 for semantic loops
- Default sliding window size: 10 actions
- Store loop patterns as AntiPatterns in ALMA memory

---

*Document Version: 0.6.0-draft*
*Next Review: After Sprint 1 completion*
