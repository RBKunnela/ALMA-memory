# Convergence Evaluators - User Stories

**Epic:** Convergence Evaluators (Ralph Wiggum Pattern)
**Target Version:** 0.6.0
**Sprint:** 1

---

## CONV-001: Loop Detector Core

**Story Points:** 5
**Priority:** P0 - Critical
**Dependencies:** None

### User Story

**As a** system operator,
**I want** ALMA to detect when agents are stuck in repetitive loops,
**So that** we can prevent infinite execution and wasted resources.

### Description

Implement the core loop detection system that monitors agent actions and identifies repetitive patterns. The detector should handle:
- Exact action repetition
- Semantically similar action loops
- State cycling (oscillation patterns)
- Configurable detection parameters

### Acceptance Criteria

- [ ] Detect exact action repetition within a sliding window
- [ ] Detect semantic similarity loops using embedding comparison
- [ ] Detect state cycling patterns (A→B→A→B→...)
- [ ] Provide confidence score (0-1) for loop detection
- [ ] Support configurable thresholds:
  - `window_size`: Number of actions to analyze (default: 10)
  - `similarity_threshold`: For semantic loops (default: 0.85)
  - `min_repetitions`: Minimum repetitions to flag (default: 3)
- [ ] Return `LoopDetection` with type, start index, length, and confidence
- [ ] Handle edge cases (insufficient history, mixed patterns)
- [ ] Unit tests with >90% coverage

### Technical Design

```python
# alma/convergence/loop_detector.py

@dataclass
class LoopDetection:
    loop_type: Literal["exact", "semantic", "state_cycle", "oscillation"]
    loop_start_index: int
    loop_length: int
    confidence: float
    repeated_actions: List[str]

class LoopDetector:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        window_size: int = 10,
        similarity_threshold: float = 0.85,
        min_repetitions: int = 3,
    ):
        pass

    def detect_loop(
        self,
        action_history: List[AgentAction],
    ) -> Optional[LoopDetection]:
        pass
```

### Test Cases

1. Exact repetition: `[A, B, A, B, A, B]` → detect oscillation
2. Semantic similarity: `["test login", "verify login", "check login"]`
3. No loop: Random diverse actions
4. Partial loop: Loop starts mid-history
5. Empty history: Return None gracefully

---

## CONV-002: Quality Evaluator Framework

**Story Points:** 5
**Priority:** P0 - Critical
**Dependencies:** None

### User Story

**As a** system operator,
**I want** to define quality gates for agent outputs,
**So that** poor quality results are flagged before propagating.

### Description

Implement a pluggable quality evaluation framework that allows defining custom quality gates. The framework should support schema validation, semantic coherence checks, and domain-specific rules.

### Acceptance Criteria

- [ ] Define `QualityGate` interface for pluggable checks
- [ ] Implement built-in gates:
  - Schema validation (JSON Schema)
  - Non-empty check
  - Length bounds check
  - Semantic coherence (optional, uses LLM)
- [ ] Aggregate gate results with configurable weights
- [ ] Return `QualityScore` with overall score and per-gate details
- [ ] Support blocking gates (fail-fast) and non-blocking gates
- [ ] Allow custom gate registration
- [ ] Unit tests with >90% coverage

### Technical Design

```python
# alma/convergence/quality.py

@dataclass
class QualityGate:
    name: str
    evaluator: Callable[[Any], GateResult]
    weight: float = 1.0
    blocking: bool = True

@dataclass
class GateResult:
    passed: bool
    score: float  # 0-1
    message: Optional[str] = None

@dataclass
class QualityScore:
    overall_score: float  # 0-1
    passed: bool
    gate_results: Dict[str, GateResult]
    issues: List[QualityIssue]

class QualityEvaluator:
    def __init__(self, gates: List[QualityGate]):
        pass

    def evaluate(self, output: Any) -> QualityScore:
        pass

    def add_gate(self, gate: QualityGate) -> None:
        pass
```

### Built-in Gates

1. `non_empty_gate`: Output is not None/empty
2. `json_schema_gate(schema)`: Validates against JSON Schema
3. `length_bounds_gate(min, max)`: String/list length in bounds
4. `coherence_gate()`: LLM-based semantic coherence (expensive)

---

## CONV-003: Progress Analyzer Integration

**Story Points:** 3
**Priority:** P1 - High
**Dependencies:** Existing ProgressTracker

### User Story

**As a** system operator,
**I want** ALMA to detect when agents are stalled,
**So that** I can intervene before resources are wasted.

### Description

Extend the existing ProgressTracker with stall detection and velocity analysis. This should detect when an agent stops making progress even if not in an explicit loop.

### Acceptance Criteria

- [ ] Calculate progress velocity (items/hour)
- [ ] Calculate velocity acceleration (trend)
- [ ] Detect stalls (no progress for configurable duration)
- [ ] Estimate completion time based on velocity
- [ ] Integrate with existing `WorkItem` and `ProgressSummary`
- [ ] Emit events on stall detection
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/convergence/progress.py

@dataclass
class ProgressAnalysis:
    velocity: float  # items/hour
    acceleration: float  # velocity change
    estimated_completion: Optional[datetime]
    is_stalled: bool
    stall_duration: Optional[timedelta]
    stall_threshold_breached: bool

class ProgressAnalyzer:
    def __init__(
        self,
        stall_threshold: timedelta = timedelta(minutes=30),
        min_velocity: float = 0.1,  # items/hour
    ):
        pass

    def analyze(
        self,
        work_items: List[WorkItem],
        time_window: timedelta = timedelta(hours=1),
    ) -> ProgressAnalysis:
        pass
```

---

## CONV-004: Intervention Engine

**Story Points:** 5
**Priority:** P0 - Critical
**Dependencies:** CONV-001, CONV-002

### User Story

**As a** system operator,
**I want** ALMA to automatically intervene when convergence issues are detected,
**So that** problems are resolved without manual intervention when possible.

### Description

Implement an intervention system that can respond to convergence issues with appropriate actions. Interventions should be configurable, auditable, and integrate with the memory system.

### Acceptance Criteria

- [ ] Define intervention types:
  - `BREAK_LOOP`: Suggest alternative approach
  - `ESCALATE`: Send to supervisor
  - `RESET_CONTEXT`: Clear working memory
  - `HUMAN_REVIEW`: Request human intervention
  - `RETRY_WITH_BACKOFF`: Retry with exponential backoff
- [ ] Execute interventions based on issue type and severity
- [ ] Record all interventions in ALMA memory
- [ ] Support auto-execute for low-risk interventions
- [ ] Emit events for all interventions
- [ ] Provide intervention recommendations without execution
- [ ] Rate-limit interventions to prevent cascading issues
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/convergence/intervention.py

class InterventionType(Enum):
    BREAK_LOOP = "break_loop"
    ESCALATE = "escalate"
    RESET_CONTEXT = "reset_context"
    HUMAN_REVIEW = "human_review"
    RETRY_WITH_BACKOFF = "retry_with_backoff"

@dataclass
class Intervention:
    type: InterventionType
    reason: str
    suggested_action: str
    priority: int  # 1=highest
    auto_execute: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InterventionResult:
    intervention: Intervention
    executed: bool
    success: Optional[bool]
    outcome_message: str
    recorded_in_memory: bool

class InterventionEngine:
    def __init__(
        self,
        alma: ALMA,
        rate_limit: int = 5,  # interventions per hour per agent
    ):
        pass

    def recommend(
        self,
        issue: ConvergenceIssue,
        context: ExecutionContext,
    ) -> Intervention:
        pass

    def execute(
        self,
        intervention: Intervention,
        context: ExecutionContext,
    ) -> InterventionResult:
        pass
```

---

## CONV-005: Convergence Event Types

**Story Points:** 2
**Priority:** P1 - High
**Dependencies:** Existing Event System

### User Story

**As a** system integrator,
**I want** convergence events emitted via webhooks,
**So that** I can build dashboards and alerting systems.

### Description

Extend the existing event system with convergence-specific event types.

### Acceptance Criteria

- [ ] Add `ConvergenceEventType` enum
- [ ] Emit events for:
  - Loop detected
  - Quality below threshold
  - Progress stalled
  - Intervention triggered/succeeded/failed
- [ ] Include full context in event payload
- [ ] Integrate with existing `WebhookManager`
- [ ] Unit tests for event emission

### Event Types

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

## CONV-006: Convergence Anti-Pattern Memory

**Story Points:** 3
**Priority:** P1 - High
**Dependencies:** CONV-004

### User Story

**As a** system,
**I want** to remember convergence failures,
**So that** I can avoid similar issues in the future.

### Description

Store convergence issues as anti-patterns in ALMA memory for future avoidance.

### Acceptance Criteria

- [ ] Define `ConvergenceAntiPattern` memory type
- [ ] Auto-create anti-pattern on intervention
- [ ] Include action sequence that led to issue
- [ ] Include successful intervention strategy
- [ ] Query anti-patterns during task planning
- [ ] Integrate with existing storage backends
- [ ] Unit tests with >85% coverage

### Technical Design

```python
@dataclass
class ConvergenceAntiPattern:
    id: str
    agent: str
    issue_type: str  # loop | quality | stall
    trigger_conditions: Dict[str, Any]
    action_sequence: List[str]  # What led to the issue
    intervention_used: str
    intervention_success: bool
    created_at: datetime
    project_id: str
```

---

## Sprint 1 Summary

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| CONV-001 | 5 | P0 | TODO |
| CONV-002 | 5 | P0 | TODO |
| CONV-003 | 3 | P1 | TODO |
| CONV-004 | 5 | P0 | TODO |
| CONV-005 | 2 | P1 | TODO |
| CONV-006 | 3 | P1 | TODO |
| **Total** | **23** | - | - |

---

*Created by: Aria (System Architect)*
*Date: 2026-01-29*
