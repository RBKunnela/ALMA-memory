# Python API Reference

## ALMA Class

The main entry point for the ALMA memory system.

### Initialization

```python
from alma import ALMA

# From configuration file
alma = ALMA.from_config(".alma/config.yaml")

# Or programmatically
alma = ALMA(
    storage=storage_backend,
    retrieval_engine=retrieval_engine,
    learning_protocol=learning_protocol,
    project_id="my-project",
    scopes=agent_scopes
)
```

### Core Methods

#### `retrieve(task, agent, top_k=5, user_id=None, include_shared=True)`

Retrieve relevant memories for a task.

```python
memories = alma.retrieve(
    task="Test the login form",
    agent="helena",
    top_k=5,
    user_id="user-123",      # Optional: for user preferences
    include_shared=True       # Include memories from shared agents
)

# Returns MemorySlice with:
# - memories.heuristics: List[Heuristic]
# - memories.outcomes: List[Outcome]
# - memories.preferences: List[UserPreference]
# - memories.domain_knowledge: List[DomainKnowledge]
# - memories.anti_patterns: List[AntiPattern]
# - memories.to_prompt(): str - Formatted for LLM
```

#### `learn(agent, task, outcome, strategy_used, **kwargs)`

Record a task outcome for learning.

```python
alma.learn(
    agent="helena",
    task="Test login form",
    outcome="success",  # or "failure"
    strategy_used="Tested empty fields, then invalid formats",
    task_type="form_testing",    # Optional: for grouping
    duration_ms=30000,           # Optional
    error_message=None,          # Optional: for failures
    feedback="Works perfectly"   # Optional: user feedback
)
```

#### `add_preference(user_id, category, preference, source="explicit_instruction")`

Add a user preference.

```python
alma.add_preference(
    user_id="user-123",
    category="code_style",
    preference="Always use TypeScript strict mode",
    source="explicit_instruction"
)
```

#### `add_knowledge(agent, domain, fact, source="user_stated")`

Add domain knowledge.

```python
alma.add_knowledge(
    agent="victor",
    domain="authentication",
    fact="API uses JWT with 24h expiry",
    source="documentation"
)
```

#### `add_anti_pattern(agent, pattern, why_bad, better_alternative)`

Record an anti-pattern.

```python
alma.add_anti_pattern(
    agent="helena",
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests",
    better_alternative="Use explicit waits"
)
```

#### `forget(agent=None, older_than_days=90, below_confidence=0.3)`

Prune old or low-confidence memories.

```python
result = alma.forget(
    agent="helena",        # Optional: specific agent
    older_than_days=90,    # Remove outcomes older than this
    below_confidence=0.3   # Remove heuristics below this confidence
)
# Returns: {"pruned_count": 42}
```

#### `stats(agent=None)`

Get memory statistics.

```python
stats = alma.stats()  # All agents
stats = alma.stats("helena")  # Specific agent

# Returns MemoryStats with counts per type
```

### Workflow Methods (v0.6.0)

#### `checkpoint(workflow_id, state, metadata=None)`

Save workflow state.

```python
alma.checkpoint(
    workflow_id="deploy-v2",
    state={"step": "testing", "passed": 42},
    metadata={"agent": "victor"}
)
```

#### `resume(workflow_id)`

Resume from latest checkpoint.

```python
checkpoint = alma.resume(workflow_id="deploy-v2")
# Returns: Checkpoint with state, metadata, timestamp
```

#### `merge_states(workflow_id, states, reducer="latest_wins")`

Merge states from parallel agents.

```python
merged = alma.merge_states(
    workflow_id="deploy-v2",
    states=[state1, state2, state3],
    reducer="latest_wins"  # or "merge_lists", "priority_agent", or callable
)
```

#### `link_artifact(workflow_id, artifact_type, ref, metadata=None)`

Link an output artifact to a workflow.

```python
alma.link_artifact(
    workflow_id="deploy-v2",
    artifact_type="test",  # code, test, document, config, deployment
    ref="tests/integration/test_auth.py",
    metadata={"coverage": "95%"}
)
```

#### `get_artifacts(workflow_id, artifact_type=None)`

Get artifacts for a workflow.

```python
artifacts = alma.get_artifacts(
    workflow_id="deploy-v2",
    artifact_type="test"  # Optional filter
)
```

#### `retrieve_scoped(query, scope, agent=None)`

Retrieve with scope filter.

```python
memories = alma.retrieve_scoped(
    query="authentication patterns",
    scope="workflow_only",  # workflow_only, agent_only, project_wide
    agent="victor"
)
```

## Types

See the `alma.types` module for all type definitions:

- `Heuristic`
- `Outcome`
- `UserPreference`
- `DomainKnowledge`
- `AntiPattern`
- `MemorySlice`
- `MemoryScope`
- `MemoryStats`
