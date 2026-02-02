"""
ALMA Workflow Module.

Provides workflow context, checkpointing, state management, and artifact
linking for integration with workflow orchestration systems like AGtestari.

Sprint 1 Task 1.7
"""

# Context and scoping
# Artifact linking
from alma.workflow.artifacts import (
    ArtifactRef,
    ArtifactType,
    link_artifact,
)

# Checkpoints for crash recovery
from alma.workflow.checkpoint import (
    DEFAULT_MAX_STATE_SIZE,
    Checkpoint,
    CheckpointManager,
)
from alma.workflow.context import (
    RetrievalScope,
    WorkflowContext,
)

# Workflow outcomes for learning
from alma.workflow.outcomes import (
    WorkflowOutcome,
    WorkflowResult,
)

# State reducers for parallel merge
from alma.workflow.reducers import (
    BUILTIN_REDUCERS,
    AppendReducer,
    FirstValueReducer,
    LastValueReducer,
    MaxReducer,
    MergeDictReducer,
    MinReducer,
    ReducerConfig,
    StateMerger,
    StateReducer,
    SumReducer,
    UnionReducer,
    get_reducer,
    merge_states,
)

__all__ = [
    # Context
    "RetrievalScope",
    "WorkflowContext",
    # Checkpoints
    "Checkpoint",
    "CheckpointManager",
    "DEFAULT_MAX_STATE_SIZE",
    # Outcomes
    "WorkflowOutcome",
    "WorkflowResult",
    # Artifacts
    "ArtifactRef",
    "ArtifactType",
    "link_artifact",
    # Reducers
    "StateReducer",
    "AppendReducer",
    "MergeDictReducer",
    "LastValueReducer",
    "FirstValueReducer",
    "SumReducer",
    "MaxReducer",
    "MinReducer",
    "UnionReducer",
    "ReducerConfig",
    "StateMerger",
    "get_reducer",
    "merge_states",
    "BUILTIN_REDUCERS",
]
