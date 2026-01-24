# ALMA Expansion PRD: Domain Memory Factory & Intelligent Learning

**Version**: 1.0
**Date**: 2026-01-24
**Author**: Claude Opus 4.5 + User
**Status**: Draft - Awaiting Approval

---

## Executive Summary

Expand ALMA from a coding-focused memory system to a **general-purpose agent memory framework** that enables any AI agent to learn, remember, and improve over time. This expansion introduces domain-agnostic memory schemas, session continuity, progress tracking, and forward-looking confidence signals.

---

## Problem Statement

### Current Limitations

1. **Coding-Only Focus**: ALMA is hardcoded for Helena (frontend) and Victor (backend) agents
2. **No Session Continuity**: Agents can't "pick up where they left off"
3. **No Progress Tracking**: No concept of features, tasks, or completion status
4. **Backward-Looking Only**: Heuristics only reflect past success rates, not predictive confidence
5. **No Initialization Pattern**: Agents dive in without orientation phase

### Opportunity

The text insight: *"General harness pattern with domain-specific memory schema"* - ALMA can become the memory layer for ANY agentic workflow:
- Research assistants
- Sales agents
- Customer support
- Content creation
- Data analysis
- Any tool-using agent

---

## Goals & Success Criteria

### Primary Goals

| Goal | Success Metric |
|------|----------------|
| Domain-agnostic memory | 3+ domain schemas (coding, research, general) |
| Session continuity | Agents resume within 5 seconds of context reload |
| Progress tracking | Feature/task completion visible and queryable |
| Forward-looking confidence | Predictive signals influence retrieval ranking |
| Initialization pattern | All sessions start with orientation phase |

### Non-Goals (Out of Scope)

- Training model weights (this is memory-layer only)
- Real-time collaboration between agents
- External integrations (Jira, GitHub, etc.) - future phase
- UI/Dashboard - API only for now

---

## Feature Specifications

### Feature 1: Domain Memory Factory

**Priority**: P1
**Effort**: Medium

#### Description

A factory pattern for creating domain-specific memory schemas. Instead of hardcoding Helena/Victor categories, allow any domain to define its entity types, relationships, and learning categories.

#### Data Model

```python
@dataclass
class DomainSchema:
    """Defines memory structure for a specific domain."""
    id: str
    name: str  # "coding", "research", "sales"
    description: str

    # What entities exist in this domain
    entity_types: List[EntityType]

    # What relationships between entities
    relationship_types: List[RelationshipType]

    # Learning categories (replaces hardcoded HELENA_CATEGORIES)
    learning_categories: List[str]

    # What can agents in this domain NOT learn
    excluded_categories: List[str]

    # Domain-specific settings
    min_occurrences_for_heuristic: int = 3
    confidence_decay_days: float = 30.0

    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityType:
    """A type of entity in the domain."""
    name: str  # "feature", "test", "paper", "lead"
    description: str
    attributes: List[str]  # ["status", "priority", "owner"]


@dataclass
class RelationshipType:
    """A relationship between entities."""
    name: str  # "implements", "blocks", "supports"
    source_type: str
    target_type: str
```

#### API

```python
class DomainMemoryFactory:
    """Factory for creating domain-specific ALMA instances."""

    @classmethod
    def create_schema(cls, name: str, config: Dict) -> DomainSchema:
        """Create a new domain schema."""
        pass

    @classmethod
    def get_coding_schema(cls) -> DomainSchema:
        """Pre-built schema for coding workflows."""
        pass

    @classmethod
    def get_research_schema(cls) -> DomainSchema:
        """Pre-built schema for research workflows."""
        pass

    @classmethod
    def get_general_schema(cls) -> DomainSchema:
        """Minimal schema for general-purpose agents."""
        pass

    def create_alma(
        self,
        schema: DomainSchema,
        project_id: str,
        storage: StorageBackend,
    ) -> ALMA:
        """Create ALMA instance configured for this domain."""
        pass
```

#### Pre-built Schemas

**Coding Domain** (current ALMA, formalized):
```yaml
name: coding
entity_types:
  - feature: {attributes: [status, tests, files]}
  - bug: {attributes: [severity, reproduction, fix]}
  - test: {attributes: [type, status, coverage]}
learning_categories:
  - testing_strategies
  - selector_patterns
  - api_design_patterns
  - error_handling
  - performance_optimization
```

**Research Domain**:
```yaml
name: research
entity_types:
  - paper: {attributes: [title, authors, year, citations]}
  - hypothesis: {attributes: [statement, confidence, evidence]}
  - experiment: {attributes: [method, results, conclusions]}
learning_categories:
  - literature_review_patterns
  - methodology_selection
  - data_analysis_strategies
  - citation_patterns
```

**Sales Domain**:
```yaml
name: sales
entity_types:
  - lead: {attributes: [stage, value, next_action]}
  - objection: {attributes: [type, response, outcome]}
  - conversation: {attributes: [channel, sentiment, result]}
learning_categories:
  - objection_handling
  - closing_techniques
  - qualification_patterns
  - follow_up_timing
```

---

### Feature 2: Progress Tracking

**Priority**: P1
**Effort**: Low

#### Description

Track work progress at feature/task level. Enables "read progress, pick something to work on" pattern.

#### Data Model

```python
@dataclass
class WorkItem:
    """A trackable unit of work."""
    id: str
    project_id: str
    agent: Optional[str]

    # Work item details
    title: str
    description: str
    item_type: str  # "feature", "bug", "task", "research_question"
    status: Literal["pending", "in_progress", "blocked", "review", "done", "failed"]
    priority: int  # 0-100

    # Progress tracking
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    time_spent_ms: int = 0
    attempt_count: int = 0

    # Relationships
    parent_id: Optional[str]
    blocks: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)

    # Validation
    tests: List[str] = field(default_factory=list)
    tests_passing: bool = False

    # Metadata
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressLog:
    """Session-level progress snapshot."""
    id: str
    project_id: str
    agent: str
    session_id: str

    # Progress snapshot
    items_total: int
    items_done: int
    items_in_progress: int
    items_blocked: int

    # Current focus
    current_item_id: Optional[str]
    current_action: str

    # Session metrics
    session_start: datetime
    actions_taken: int
    outcomes_recorded: int

    created_at: datetime
```

#### API

```python
class ProgressTracker:
    """Track work item progress."""

    def create_work_item(self, item: WorkItem) -> str:
        """Create a new work item."""
        pass

    def update_status(
        self,
        item_id: str,
        status: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Update work item status."""
        pass

    def get_next_item(
        self,
        project_id: str,
        agent: Optional[str] = None,
        strategy: Literal["priority", "blocked_unblock", "quick_win"] = "priority",
    ) -> Optional[WorkItem]:
        """Get next work item to focus on."""
        pass

    def get_progress_summary(
        self,
        project_id: str,
        agent: Optional[str] = None,
    ) -> ProgressSummary:
        """Get progress summary for project/agent."""
        pass

    def log_session_progress(
        self,
        project_id: str,
        agent: str,
        session_id: str,
    ) -> ProgressLog:
        """Create progress snapshot for session."""
        pass
```

---

### Feature 3: Session Handoff & Quick Reload

**Priority**: P2
**Effort**: Low

#### Description

Enable fast session context reload. Agents can "reboot quickly" without full context reconstruction.

#### Data Model

```python
@dataclass
class SessionHandoff:
    """Compressed context for session continuity."""
    id: str
    project_id: str
    agent: str
    session_id: str

    # Where we left off
    last_action: str
    last_outcome: Literal["success", "failure", "interrupted"]
    current_goal: str

    # Quick context (not full history)
    key_decisions: List[str]  # Max 10 most important
    active_files: List[str]  # Files being worked on
    blockers: List[str]  # Current blockers
    next_steps: List[str]  # Planned next actions

    # Test/validation state
    test_status: Dict[str, bool]  # test_name -> passing

    # Emotional/confidence signals
    confidence_level: float  # 0-1, how well is this going
    risk_flags: List[str]  # Concerns noted

    # Timing
    session_start: datetime
    session_end: datetime
    duration_ms: int

    # For retrieval
    embedding: Optional[List[float]] = None

    created_at: datetime


@dataclass
class SessionContext:
    """Full context for starting/resuming a session."""
    # Handoff from previous session
    previous_handoff: Optional[SessionHandoff]

    # Current state
    progress: ProgressSummary
    recent_outcomes: List[Outcome]
    relevant_heuristics: List[Heuristic]

    # Orientation data
    codebase_state: Optional[Dict]  # git status, recent commits
    environment_state: Optional[Dict]  # running services, etc.

    # What to do
    suggested_focus: Optional[WorkItem]
    rules_of_engagement: List[str]
```

#### API

```python
class SessionManager:
    """Manage session continuity."""

    def start_session(
        self,
        project_id: str,
        agent: str,
    ) -> SessionContext:
        """Start a new session, loading previous context."""
        pass

    def create_handoff(
        self,
        project_id: str,
        agent: str,
        session_id: str,
        last_action: str,
        last_outcome: str,
        **context,
    ) -> SessionHandoff:
        """Create handoff for session end."""
        pass

    def get_quick_reload(
        self,
        project_id: str,
        agent: str,
    ) -> str:
        """Get compressed context string for quick reload."""
        # Returns formatted string agent can quickly parse
        pass

    def get_previous_sessions(
        self,
        project_id: str,
        agent: str,
        limit: int = 5,
    ) -> List[SessionHandoff]:
        """Get recent session handoffs."""
        pass
```

---

### Feature 4: Initializer Agent Pattern

**Priority**: P1
**Effort**: Medium

#### Description

A bootstrap phase that orients the agent before work begins. "Stage manager sets the stage, actor performs."

#### Data Model

```python
@dataclass
class InitializationResult:
    """Result of session initialization."""
    session_id: str
    project_id: str
    agent: str

    # Expanded from user prompt
    goal: str
    work_items: List[WorkItem]

    # Orientation results
    codebase_summary: Optional[str]
    recent_activity: List[str]
    relevant_memories: MemorySlice

    # Rules of engagement
    scope_rules: List[str]  # What agent CAN do
    constraints: List[str]  # What agent CANNOT do
    quality_gates: List[str]  # Must pass before "done"

    # Suggested first action
    recommended_start: Optional[WorkItem]

    initialized_at: datetime
```

#### API

```python
class SessionInitializer:
    """Bootstrap domain memory from user prompt."""

    async def initialize(
        self,
        project_id: str,
        agent: str,
        user_prompt: str,
        auto_expand: bool = True,
    ) -> InitializationResult:
        """
        Full initialization:
        1. Expand prompt to work items (if auto_expand)
        2. Orient to current state (git, progress, etc.)
        3. Retrieve relevant memories
        4. Set rules of engagement from scope
        5. Suggest starting point
        """
        pass

    async def expand_prompt(
        self,
        user_prompt: str,
        domain: DomainSchema,
    ) -> List[WorkItem]:
        """Expand user prompt into structured work items."""
        pass

    async def orient_to_codebase(
        self,
        project_id: str,
    ) -> Dict[str, Any]:
        """Read git status, recent commits, file structure."""
        pass

    def get_rules_of_engagement(
        self,
        agent: str,
        scope: MemoryScope,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Get scope_rules, constraints, quality_gates."""
        pass
```

---

### Feature 5: Forward-Looking Confidence (Value Function)

**Priority**: P2
**Effort**: High

#### Description

Add predictive confidence signals to heuristics. Not just "this worked before" but "this will likely work now."

Inspired by Ilya Sutskever's insight: emotions are forward-looking value functions, while RL is backward-looking.

#### Data Model

```python
@dataclass
class ConfidenceSignal:
    """Forward-looking confidence assessment."""
    # What we're assessing
    heuristic_id: Optional[str]
    strategy: str
    context: str

    # Backward-looking (existing)
    historical_success_rate: float
    occurrence_count: int

    # Forward-looking (NEW)
    predicted_success: float  # Expected success in THIS context
    uncertainty: float  # How uncertain is the prediction
    context_similarity: float  # How similar is current context to past successes

    # Risk/opportunity signals ("gut feelings")
    risk_signals: List[RiskSignal]
    opportunity_signals: List[OpportunitySignal]

    # Combined score
    confidence_score: float  # Weighted combination
    recommendation: Literal["strong_yes", "yes", "neutral", "caution", "avoid"]


@dataclass
class RiskSignal:
    """A risk indicator."""
    signal_type: str  # "similar_to_failure", "untested_context", "high_complexity"
    description: str
    severity: float  # 0-1
    source: str  # What triggered this signal


@dataclass
class OpportunitySignal:
    """An opportunity indicator."""
    signal_type: str  # "proven_pattern", "high_similarity", "recent_success"
    description: str
    strength: float  # 0-1
    source: str
```

#### API

```python
class ConfidenceEngine:
    """Compute forward-looking confidence for strategies."""

    def assess_strategy(
        self,
        strategy: str,
        context: str,
        agent: str,
        heuristic: Optional[Heuristic] = None,
    ) -> ConfidenceSignal:
        """Assess confidence for a strategy in current context."""
        pass

    def rank_strategies(
        self,
        strategies: List[str],
        context: str,
        agent: str,
    ) -> List[Tuple[str, ConfidenceSignal]]:
        """Rank multiple strategies by confidence."""
        pass

    def detect_risks(
        self,
        strategy: str,
        context: str,
        agent: str,
    ) -> List[RiskSignal]:
        """Detect risk signals for a strategy."""
        pass

    def detect_opportunities(
        self,
        strategy: str,
        context: str,
        agent: str,
    ) -> List[OpportunitySignal]:
        """Detect opportunity signals for a strategy."""
        pass
```

---

## Implementation Phases

### Phase 1: Foundation (P1 Features)

**Duration**: 1-2 weeks
**Deliverables**:
- [ ] `DomainSchema` and `DomainMemoryFactory` classes
- [ ] Pre-built schemas: coding, research, general
- [ ] `WorkItem` and `ProgressTracker` classes
- [ ] `SessionInitializer` basic implementation
- [ ] Storage backend updates for new types
- [ ] Unit tests for all new components
- [ ] Update MCP tools to use new features

### Phase 2: Session Continuity (P2 Features)

**Duration**: 1 week
**Deliverables**:
- [ ] `SessionHandoff` and `SessionManager` classes
- [ ] Quick reload functionality
- [ ] Session history tracking
- [ ] MCP tools: `alma_start_session`, `alma_end_session`

### Phase 3: Intelligent Confidence (P2 Features)

**Duration**: 2 weeks
**Deliverables**:
- [ ] `ConfidenceSignal` and `ConfidenceEngine` classes
- [ ] Risk/opportunity detection algorithms
- [ ] Integration with retrieval ranking
- [ ] Benchmarks comparing backward vs forward-looking

### Phase 4: Documentation & Polish

**Duration**: 1 week
**Deliverables**:
- [ ] Updated README with new features
- [ ] Domain schema creation guide
- [ ] Migration guide from v0.2 to v0.3
- [ ] Example notebooks for each domain

---

## File Structure

```
alma/
├── __init__.py                    # Update exports
├── core.py                        # Update ALMA class
├── types.py                       # Add new types
├── domains/                       # NEW
│   ├── __init__.py
│   ├── schema.py                  # DomainSchema, EntityType, etc.
│   ├── factory.py                 # DomainMemoryFactory
│   └── presets/                   # Pre-built schemas
│       ├── coding.py
│       ├── research.py
│       └── general.py
├── progress/                      # NEW
│   ├── __init__.py
│   ├── types.py                   # WorkItem, ProgressLog
│   └── tracker.py                 # ProgressTracker
├── session/                       # NEW
│   ├── __init__.py
│   ├── types.py                   # SessionHandoff, SessionContext
│   ├── manager.py                 # SessionManager
│   └── initializer.py             # SessionInitializer
├── confidence/                    # NEW
│   ├── __init__.py
│   ├── types.py                   # ConfidenceSignal, RiskSignal
│   └── engine.py                  # ConfidenceEngine
├── mcp/
│   ├── tools.py                   # Add new MCP tools
│   └── ...
└── storage/
    ├── base.py                    # Add new abstract methods
    └── ...                        # Update implementations
```

---

## New MCP Tools

| Tool | Description |
|------|-------------|
| `alma_create_domain` | Create custom domain schema |
| `alma_create_work_item` | Create trackable work item |
| `alma_update_progress` | Update work item status |
| `alma_get_next_item` | Get next item to work on |
| `alma_start_session` | Initialize session with orientation |
| `alma_end_session` | Create session handoff |
| `alma_quick_reload` | Get compressed context |
| `alma_assess_confidence` | Get confidence for strategy |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Domain generalization | 3+ domains | Count of working domain schemas |
| Session reload time | < 5 seconds | Time from start to oriented |
| Progress visibility | 100% items tracked | Items with status vs total |
| Confidence accuracy | > 70% | Predicted success vs actual |
| Adoption | 2+ non-coding uses | Domains actively used |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-engineering domains | Medium | Start with 3 presets, validate before custom |
| Storage bloat from progress | Low | Add retention policies, archive old sessions |
| Confidence engine complexity | High | Ship basic version first, iterate |
| Breaking changes | Medium | Maintain backward compatibility layer |

---

## Dependencies

- No new external dependencies for Phase 1
- Phase 3 (confidence) may need sklearn for similarity scoring

---

## Approval

- [ ] User approval to proceed
- [ ] Priority confirmation (Phase 1 first?)
- [ ] Any scope adjustments needed?

---

## References

- Original expansion analysis from conversation
- Ilya Sutskever on emotions as value functions
- John Duruk's focus system model
