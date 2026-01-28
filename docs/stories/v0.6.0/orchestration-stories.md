# Two-Tier Orchestration - User Stories

**Epic:** Two-Tier Orchestration Layer
**Target Version:** 0.6.0
**Sprint:** 3-4

---

## ORCH-001: Worker Interface

**Story Points:** 3
**Priority:** P0 - Critical
**Dependencies:** None

### User Story

**As a** developer,
**I want** a standard Worker interface,
**So that** I can create specialized agents that integrate with orchestration.

### Description

Define the base Worker interface that all orchestratable agents must implement. Workers are specialized agents that can receive subtasks from a Supervisor and report results.

### Acceptance Criteria

- [ ] Define `Worker` abstract base class
- [ ] Define `WorkerCapability` for describing what worker can do
- [ ] Define `Subtask` structure for work assignment
- [ ] Define `WorkerResult` for reporting outcomes
- [ ] Define `WorkerContext` for execution context
- [ ] Implement `can_handle()` method for capability matching
- [ ] Workers integrate with existing ALMA memory via their scope
- [ ] Support async execution
- [ ] Unit tests with >90% coverage

### Technical Design

```python
# alma/orchestration/worker.py

@dataclass
class WorkerCapability:
    name: str
    description: str
    task_types: List[str]  # Types this worker handles
    proficiency: float = 1.0  # 0-1, how good at this

@dataclass
class Subtask:
    id: str
    parent_task_id: str
    description: str
    task_type: str
    priority: int
    dependencies: List[str]  # IDs of subtasks that must complete first
    context: Dict[str, Any]
    deadline: Optional[datetime] = None

@dataclass
class WorkerContext:
    subtask: Subtask
    parent_context: Dict[str, Any]
    memory_slice: Optional[MemorySlice] = None
    sibling_results: List["WorkerResult"] = field(default_factory=list)

@dataclass
class WorkerResult:
    subtask_id: str
    worker_name: str
    success: bool
    output: Any
    error_message: Optional[str] = None
    duration_ms: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class Worker(ABC):
    """Base class for all orchestratable workers."""

    def __init__(
        self,
        name: str,
        alma: ALMA,
        capabilities: List[WorkerCapability],
    ):
        self.name = name
        self.alma = alma
        self.capabilities = capabilities
        self.scope = alma.scopes.get(name)

    @property
    def capability_names(self) -> List[str]:
        return [c.name for c in self.capabilities]

    def can_handle(self, subtask: Subtask) -> float:
        """
        Return confidence score (0-1) for handling this subtask.

        Considers:
        - Task type match with capabilities
        - Historical performance on similar tasks
        - Current load (if applicable)
        """
        pass

    @abstractmethod
    async def execute(
        self,
        context: WorkerContext,
    ) -> WorkerResult:
        """Execute the subtask and return result."""
        pass

    async def prepare(self, context: WorkerContext) -> None:
        """Optional: Prepare for execution (retrieve memories, etc.)."""
        pass

    async def cleanup(self, result: WorkerResult) -> None:
        """Optional: Cleanup after execution (learn from result, etc.)."""
        pass
```

---

## ORCH-002: Supervisor Core

**Story Points:** 8
**Priority:** P0 - Critical
**Dependencies:** ORCH-001

### User Story

**As a** system operator,
**I want** a Supervisor that coordinates multiple workers,
**So that** complex tasks can be executed by specialized agents.

### Description

Implement the core Supervisor class that manages task decomposition, worker routing, execution monitoring, and result aggregation.

### Acceptance Criteria

- [ ] Supervisor manages a pool of Workers
- [ ] Supervisor has its own MemoryScope for orchestration learning
- [ ] Execute complex tasks through worker delegation
- [ ] Track execution state and progress
- [ ] Handle worker failures with retry/fallback
- [ ] Learn from orchestration outcomes
- [ ] Support configurable execution strategies
- [ ] Emit events for orchestration lifecycle
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/supervisor.py

@dataclass
class ComplexTask:
    id: str
    description: str
    task_type: str
    context: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    deadline: Optional[datetime] = None

@dataclass
class OrchestrationResult:
    task_id: str
    success: bool
    subtask_results: List[WorkerResult]
    aggregated_output: Any
    total_duration_ms: int
    worker_utilization: Dict[str, float]  # worker -> % time used
    errors: List[str] = field(default_factory=list)

class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"  # One subtask at a time
    PARALLEL = "parallel"  # All independent subtasks in parallel
    ADAPTIVE = "adaptive"  # Start sequential, parallelize when confident

class Supervisor:
    """
    Orchestrates multi-agent task execution.

    The Supervisor:
    1. Decomposes complex tasks into subtasks
    2. Routes subtasks to appropriate workers
    3. Monitors execution and handles failures
    4. Aggregates results
    5. Learns from orchestration outcomes
    """

    def __init__(
        self,
        name: str,
        alma: ALMA,
        workers: List[Worker],
        decomposer: Optional["TaskDecomposer"] = None,
        router: Optional["WorkerRouter"] = None,
        aggregator: Optional["ResultAggregator"] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        max_retries: int = 2,
    ):
        pass

    async def execute(
        self,
        task: ComplexTask,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Execute a complex task through worker delegation.

        1. Retrieve orchestration memories
        2. Decompose into subtasks
        3. Build execution plan (respecting dependencies)
        4. Route subtasks to workers
        5. Monitor execution
        6. Handle failures with retry/fallback
        7. Aggregate results
        8. Learn from orchestration outcome
        """
        pass

    def register_worker(self, worker: Worker) -> None:
        """Add a worker to the pool."""
        pass

    def unregister_worker(self, worker_name: str) -> None:
        """Remove a worker from the pool."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        pass

    async def cancel(self, task_id: str) -> bool:
        """Cancel an in-progress orchestration."""
        pass
```

---

## ORCH-003: Task Decomposer

**Story Points:** 5
**Priority:** P1 - High
**Dependencies:** ORCH-002

### User Story

**As a** supervisor,
**I want** to decompose complex tasks into subtasks,
**So that** work can be distributed to specialized workers.

### Description

Implement a task decomposition system that breaks complex tasks into manageable subtasks with dependencies.

### Acceptance Criteria

- [ ] Decompose tasks based on available worker capabilities
- [ ] Create dependency graph (DAG) for subtasks
- [ ] Estimate subtask complexity/duration
- [ ] Learn optimal decomposition strategies from memory
- [ ] Support different decomposition strategies:
  - Capability-based (match to workers)
  - Phase-based (logical phases)
  - Domain-based (split by domain)
- [ ] Validate decomposition (complete coverage, no cycles)
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/decomposer.py

class DecompositionStrategy(Enum):
    CAPABILITY_BASED = "capability_based"
    PHASE_BASED = "phase_based"
    DOMAIN_BASED = "domain_based"
    HYBRID = "hybrid"

@dataclass
class DecompositionPlan:
    subtasks: List[Subtask]
    dependency_graph: Dict[str, List[str]]  # subtask_id -> depends_on
    estimated_total_duration_ms: int
    parallelism_factor: float  # 1.0 = fully sequential, N = N parallel tracks
    confidence: float

class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks.

    Uses ALMA memory to learn optimal decomposition strategies:
    - Which task types decompose well
    - Optimal granularity for different contexts
    - Dependencies between subtask types
    """

    def __init__(
        self,
        alma: ALMA,
        strategy: DecompositionStrategy = DecompositionStrategy.HYBRID,
        max_subtasks: int = 10,
    ):
        pass

    def decompose(
        self,
        task: ComplexTask,
        available_workers: List[Worker],
    ) -> DecompositionPlan:
        """
        Decompose task into subtasks.

        Steps:
        1. Retrieve decomposition heuristics from memory
        2. Analyze task requirements
        3. Match to worker capabilities
        4. Create subtasks with dependencies
        5. Validate the plan
        """
        pass

    def validate_plan(self, plan: DecompositionPlan) -> List[str]:
        """
        Validate decomposition plan.

        Returns list of issues (empty if valid).
        Checks:
        - No circular dependencies
        - All dependencies exist
        - Every subtask can be handled by at least one worker
        """
        pass
```

---

## ORCH-004: Worker Router

**Story Points:** 5
**Priority:** P1 - High
**Dependencies:** ORCH-001, ORCH-002

### User Story

**As a** supervisor,
**I want** optimal worker assignment for subtasks,
**So that** work is completed efficiently and successfully.

### Description

Implement a routing system that assigns subtasks to the optimal workers based on capabilities, historical performance, and current load.

### Acceptance Criteria

- [ ] Route subtasks based on capability match
- [ ] Consider historical performance from memory
- [ ] Support load balancing across workers
- [ ] Handle worker unavailability
- [ ] Provide fallback routes
- [ ] Learn routing preferences from outcomes
- [ ] Support routing constraints (e.g., specific worker required)
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/router.py

@dataclass
class RoutingContext:
    available_workers: List[Worker]
    worker_loads: Dict[str, float]  # worker -> current load (0-1)
    time_budget_ms: Optional[int] = None
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class RoutingDecision:
    subtask_id: str
    primary_worker: str
    fallback_workers: List[str]
    confidence: float
    reasoning: str

class RoutingStrategy(Enum):
    BEST_FIT = "best_fit"  # Best capability match
    LEAST_LOADED = "least_loaded"  # Lowest current load
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    MEMORY_INFORMED = "memory_informed"  # Based on historical success

class WorkerRouter:
    """
    Routes subtasks to optimal workers.

    Uses ALMA memory to learn:
    - Which workers excel at which task types
    - Load balancing strategies
    - Fallback routes when primary fails
    """

    def __init__(
        self,
        alma: ALMA,
        strategy: RoutingStrategy = RoutingStrategy.MEMORY_INFORMED,
    ):
        pass

    def route(
        self,
        subtask: Subtask,
        context: RoutingContext,
    ) -> RoutingDecision:
        """
        Select optimal worker for subtask.

        Considers:
        - Worker capabilities and proficiency
        - Historical performance on similar tasks
        - Current worker load
        - Routing constraints
        """
        pass

    def route_batch(
        self,
        subtasks: List[Subtask],
        context: RoutingContext,
    ) -> List[RoutingDecision]:
        """
        Route multiple subtasks optimizing overall assignment.

        Uses optimization to balance load while maximizing
        capability match.
        """
        pass
```

---

## ORCH-005: Result Aggregator

**Story Points:** 5
**Priority:** P1 - High
**Dependencies:** ORCH-002

### User Story

**As a** supervisor,
**I want** to aggregate subtask results into a coherent output,
**So that** the complex task has a unified result.

### Description

Implement result aggregation that combines outputs from multiple workers into a single coherent result.

### Acceptance Criteria

- [ ] Aggregate results from multiple subtasks
- [ ] Handle partial failures (some subtasks failed)
- [ ] Resolve conflicts between worker outputs
- [ ] Support different aggregation strategies:
  - Concatenation (collect all)
  - Merge (combine into one)
  - Vote (majority wins)
  - Custom (user-defined)
- [ ] Validate aggregated result quality
- [ ] Learn aggregation preferences from feedback
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/aggregator.py

class AggregationStrategy(Enum):
    CONCATENATE = "concatenate"  # Collect all outputs
    MERGE = "merge"  # Combine into single structure
    VOTE = "vote"  # Majority/weighted vote
    REDUCE = "reduce"  # Sequential reduction
    CUSTOM = "custom"  # User-defined function

@dataclass
class AggregationResult:
    output: Any
    source_results: List[WorkerResult]
    failed_subtasks: List[str]
    conflicts_resolved: List[str]
    confidence: float

class ResultAggregator:
    """
    Aggregates results from multiple workers.

    Handles:
    - Combining outputs
    - Resolving conflicts
    - Handling partial failures
    """

    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.MERGE,
        conflict_resolver: Optional[Callable] = None,
        allow_partial: bool = True,  # Allow result even if some subtasks failed
        min_success_ratio: float = 0.8,  # Min ratio of successful subtasks
    ):
        pass

    def aggregate(
        self,
        results: List[WorkerResult],
        task: ComplexTask,
    ) -> AggregationResult:
        """Aggregate worker results into final output."""
        pass

    def resolve_conflict(
        self,
        conflicting_results: List[WorkerResult],
    ) -> Any:
        """Resolve conflicts between results."""
        pass
```

---

## ORCH-006: Orchestration Memory Types

**Story Points:** 3
**Priority:** P1 - High
**Dependencies:** ORCH-002

### User Story

**As a** system,
**I want** to remember orchestration decisions and outcomes,
**So that** future orchestrations can be more effective.

### Description

Define memory types for storing orchestration outcomes and learned routing/decomposition heuristics.

### Acceptance Criteria

- [ ] Define `OrchestrationOutcome` memory type
- [ ] Define `RoutingHeuristic` memory type
- [ ] Define `DecompositionHeuristic` memory type
- [ ] Integrate with existing storage backends
- [ ] Support retrieval for orchestration planning
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/types.py

@dataclass
class OrchestrationOutcome:
    id: str
    supervisor: str
    project_id: str
    task_type: str
    task_description: str
    decomposition_strategy: str
    subtask_count: int
    routing_decisions: List[Dict]  # [{"subtask": X, "worker": Y, "success": Z}]
    aggregation_strategy: str
    overall_success: bool
    total_duration_ms: int
    worker_utilization: Dict[str, float]
    lessons_learned: Optional[str]
    timestamp: datetime

@dataclass
class RoutingHeuristic:
    id: str
    condition: str  # "When task involves API testing"
    routing_strategy: str  # "Prefer Victor over Helena"
    worker_preference: List[str]  # Ordered preference
    confidence: float
    success_rate: float
    occurrence_count: int
    created_at: datetime

@dataclass
class DecompositionHeuristic:
    id: str
    task_pattern: str  # "Multi-file refactoring"
    decomposition_strategy: str
    typical_subtask_count: int
    parallelism_factor: float
    confidence: float
    success_rate: float
    created_at: datetime
```

---

## ORCH-007: Failure Handler & Recovery

**Story Points:** 5
**Priority:** P1 - High
**Dependencies:** ORCH-002, CONV-004

### User Story

**As a** supervisor,
**I want** robust failure handling during orchestration,
**So that** complex tasks can complete despite worker failures.

### Description

Implement failure handling and recovery mechanisms for orchestration.

### Acceptance Criteria

- [ ] Detect worker failures (timeout, error, poor quality)
- [ ] Retry failed subtasks with same or different worker
- [ ] Support fallback to alternative workers
- [ ] Graceful degradation (partial results)
- [ ] Integration with Convergence Intervention Engine
- [ ] Configurable retry policies
- [ ] Learn from failure patterns
- [ ] Unit tests with >85% coverage

### Technical Design

```python
# alma/orchestration/failure.py

class FailureType(Enum):
    TIMEOUT = "timeout"
    ERROR = "error"
    QUALITY = "quality"
    UNAVAILABLE = "unavailable"

@dataclass
class RetryPolicy:
    max_retries: int = 2
    retry_delay_ms: int = 1000
    exponential_backoff: bool = True
    use_fallback_worker: bool = True

@dataclass
class FailureEvent:
    subtask_id: str
    worker: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime

class FailureHandler:
    def __init__(
        self,
        intervention_engine: Optional[InterventionEngine] = None,
        default_policy: RetryPolicy = RetryPolicy(),
    ):
        pass

    def handle_failure(
        self,
        failure: FailureEvent,
        subtask: Subtask,
        context: RoutingContext,
    ) -> Tuple[bool, Optional[str]]:
        """
        Handle a worker failure.

        Returns:
            (should_retry, alternative_worker)
        """
        pass

    def should_abort(
        self,
        failures: List[FailureEvent],
        task: ComplexTask,
    ) -> bool:
        """Determine if orchestration should abort."""
        pass
```

---

## ORCH-008: MCP Orchestration Tools

**Story Points:** 5
**Priority:** P2 - Medium
**Dependencies:** ORCH-002

### User Story

**As a** Claude Code user,
**I want** MCP tools for orchestration,
**So that** I can coordinate multi-agent tasks from Claude.

### Description

Add MCP tools for orchestration operations.

### Acceptance Criteria

- [ ] `alma_delegate_task`: Supervisor delegates to workers
- [ ] `alma_report_subtask`: Worker reports subtask completion
- [ ] `alma_orchestration_status`: Check orchestration progress
- [ ] `alma_cancel_orchestration`: Cancel in-progress orchestration
- [ ] Integrate with existing MCP server
- [ ] Unit tests for all tools

### Tools

```python
# New MCP tools

"alma_delegate_task"
# Input: task description, task_type, context
# Output: orchestration_id, decomposition plan

"alma_report_subtask"
# Input: orchestration_id, subtask_id, result
# Output: success, next_subtask (if any)

"alma_orchestration_status"
# Input: orchestration_id
# Output: status, progress, worker_status

"alma_cancel_orchestration"
# Input: orchestration_id
# Output: success, cleanup_summary
```

---

## Sprint 3-4 Summary

### Sprint 3 (Foundation)

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| ORCH-001 | 3 | P0 | TODO |
| ORCH-002 | 8 | P0 | TODO |
| ORCH-003 | 5 | P1 | TODO |
| **Total** | **16** | - | - |

### Sprint 4 (Completion)

| Story | Points | Priority | Status |
|-------|--------|----------|--------|
| ORCH-004 | 5 | P1 | TODO |
| ORCH-005 | 5 | P1 | TODO |
| ORCH-006 | 3 | P1 | TODO |
| ORCH-007 | 5 | P1 | TODO |
| ORCH-008 | 5 | P2 | TODO |
| **Total** | **23** | - | - |

---

*Created by: Aria (System Architect)*
*Date: 2026-01-29*
