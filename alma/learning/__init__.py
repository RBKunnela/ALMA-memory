"""
ALMA Learning Protocols.

Provides learning, validation, forgetting, and heuristic extraction.
"""

from alma.learning.forgetting import (
    CleanupJob,
    CleanupResult,
    # Cleanup Scheduling
    CleanupScheduler,
    # Confidence Decay
    ConfidenceDecayer,
    # Decay Functions
    DecayFunction,
    DecayResult,
    ExponentialDecay,
    # Forgetting Engine
    ForgettingEngine,
    HealthAlert,
    HealthThresholds,
    LinearDecay,
    MemoryHealthMetrics,
    # Memory Health Monitoring
    MemoryHealthMonitor,
    NoDecay,
    PrunePolicy,
    PruneReason,
    PruneResult,
    PruneSummary,
    StepDecay,
)
from alma.learning.heuristic_extractor import (
    ExtractionResult,
    HeuristicExtractor,
    PatternCandidate,
    extract_heuristics_from_outcome,
)
from alma.learning.protocols import LearningProtocol
from alma.learning.validation import (
    ScopeValidator,
    TaskTypeValidator,
    ValidationReport,
    ValidationResult,
    validate_learning_request,
)

__all__ = [
    # Core Protocol
    "LearningProtocol",
    # Validation
    "ScopeValidator",
    "ValidationResult",
    "ValidationReport",
    "TaskTypeValidator",
    "validate_learning_request",
    # Forgetting Engine
    "ForgettingEngine",
    "PrunePolicy",
    "PruneResult",
    "PruneSummary",
    "PruneReason",
    # Decay Functions
    "DecayFunction",
    "ExponentialDecay",
    "LinearDecay",
    "StepDecay",
    "NoDecay",
    # Confidence Decay
    "ConfidenceDecayer",
    "DecayResult",
    # Memory Health Monitoring
    "MemoryHealthMonitor",
    "MemoryHealthMetrics",
    "HealthAlert",
    "HealthThresholds",
    # Cleanup Scheduling
    "CleanupScheduler",
    "CleanupJob",
    "CleanupResult",
    # Heuristic Extraction
    "HeuristicExtractor",
    "PatternCandidate",
    "ExtractionResult",
    "extract_heuristics_from_outcome",
]
