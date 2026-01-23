"""
ALMA Learning Protocols.

Provides learning, validation, forgetting, and heuristic extraction.
"""

from alma.learning.protocols import LearningProtocol
from alma.learning.validation import (
    ScopeValidator,
    ValidationResult,
    ValidationReport,
    TaskTypeValidator,
    validate_learning_request,
)
from alma.learning.forgetting import (
    ForgettingEngine,
    PrunePolicy,
    PruneResult,
    PruneSummary,
    PruneReason,
)
from alma.learning.heuristic_extractor import (
    HeuristicExtractor,
    PatternCandidate,
    ExtractionResult,
    extract_heuristics_from_outcome,
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
    # Forgetting
    "ForgettingEngine",
    "PrunePolicy",
    "PruneResult",
    "PruneSummary",
    "PruneReason",
    # Heuristic Extraction
    "HeuristicExtractor",
    "PatternCandidate",
    "ExtractionResult",
    "extract_heuristics_from_outcome",
]
