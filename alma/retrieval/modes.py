"""
ALMA Retrieval Modes.

Provides mode-aware retrieval strategies that adapt to different cognitive tasks.
Based on Memory Wall principles: "Planning needs breadth. Execution needs precision."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class RetrievalMode(Enum):
    """
    Cognitive modes for retrieval strategy selection.

    Different tasks require fundamentally different retrieval approaches:
    - BROAD: For planning, brainstorming - needs diverse, exploratory results
    - PRECISE: For execution, implementation - needs high-confidence matches
    - DIAGNOSTIC: For debugging, troubleshooting - needs anti-patterns and failures
    - LEARNING: For pattern finding, consolidation - needs similar memories to merge
    - RECALL: For exact memory lookup - prioritizes exact matches
    """

    BROAD = "broad"
    PRECISE = "precise"
    DIAGNOSTIC = "diagnostic"
    LEARNING = "learning"
    RECALL = "recall"


@dataclass
class ModeConfig:
    """
    Configuration for a retrieval mode.

    Attributes:
        top_k: Default number of results to return
        min_confidence: Minimum confidence threshold for results
        weights: Scoring weights (similarity, recency, success, confidence)
        include_anti_patterns: Whether to include anti-patterns in results
        diversity_factor: 0.0 = pure relevance, 1.0 = maximum diversity (MMR)
        prioritize_failures: Boost failed outcomes (for debugging)
        cluster_similar: Group similar results together (for learning)
        exact_match_boost: Multiplier for exact/high-similarity matches
    """

    top_k: int
    min_confidence: float
    weights: Dict[str, float] = field(default_factory=dict)
    include_anti_patterns: bool = True
    diversity_factor: float = 0.0
    prioritize_failures: bool = False
    cluster_similar: bool = False
    exact_match_boost: float = 1.0

    def __post_init__(self):
        """Validate and normalize weights."""
        if self.weights:
            total = sum(self.weights.values())
            if total > 0 and not (0.99 <= total <= 1.01):
                # Normalize weights to sum to 1.0
                self.weights = {k: v / total for k, v in self.weights.items()}


# Default configurations for each mode
MODE_CONFIGS: Dict[RetrievalMode, ModeConfig] = {
    RetrievalMode.BROAD: ModeConfig(
        top_k=15,
        min_confidence=0.3,
        weights={
            "similarity": 0.70,
            "recency": 0.10,
            "success_rate": 0.10,
            "confidence": 0.10,
        },
        include_anti_patterns=False,  # Exploring, don't want negatives
        diversity_factor=0.8,  # High diversity for exploration
        exact_match_boost=1.0,
    ),
    RetrievalMode.PRECISE: ModeConfig(
        top_k=5,
        min_confidence=0.7,
        weights={
            "similarity": 0.30,
            "recency": 0.10,
            "success_rate": 0.40,  # Proven strategies matter
            "confidence": 0.20,
        },
        include_anti_patterns=True,  # Know what to avoid
        diversity_factor=0.2,  # Focused results
        exact_match_boost=2.0,  # Boost exact matches
    ),
    RetrievalMode.DIAGNOSTIC: ModeConfig(
        top_k=10,
        min_confidence=0.4,
        weights={
            "similarity": 0.40,
            "recency": 0.30,  # Recent issues more relevant
            "success_rate": 0.00,  # Don't penalize failures
            "confidence": 0.30,
        },
        include_anti_patterns=True,  # Critical for debugging
        diversity_factor=0.5,
        prioritize_failures=True,  # Failures are valuable here
        exact_match_boost=1.5,
    ),
    RetrievalMode.LEARNING: ModeConfig(
        top_k=20,
        min_confidence=0.2,  # Low threshold to find patterns
        weights={
            "similarity": 0.90,  # Similarity is key for consolidation
            "recency": 0.00,
            "success_rate": 0.05,
            "confidence": 0.05,
        },
        include_anti_patterns=True,
        diversity_factor=0.3,  # Some diversity but group similar
        cluster_similar=True,
        exact_match_boost=1.0,
    ),
    RetrievalMode.RECALL: ModeConfig(
        top_k=3,
        min_confidence=0.5,
        weights={
            "similarity": 0.95,  # Almost pure similarity
            "recency": 0.00,
            "success_rate": 0.00,
            "confidence": 0.05,
        },
        include_anti_patterns=False,
        diversity_factor=0.0,  # No diversity, exact match
        exact_match_boost=3.0,  # Strong boost for exact matches
    ),
}


# Keywords for mode inference
_DIAGNOSTIC_TERMS = frozenset([
    "error", "bug", "fail", "failed", "failing", "broken", "issue",
    "problem", "debug", "fix", "wrong", "crash", "exception", "traceback",
    "not working", "doesn't work", "won't work", "can't"
])

_BROAD_TERMS = frozenset([
    "how should", "what approach", "options for", "ways to", "plan",
    "design", "architect", "strategy", "alternative", "consider",
    "brainstorm", "explore", "ideas for", "possibilities"
])

_RECALL_TERMS = frozenset([
    "what was", "when did", "remember when", "last time", "previously",
    "before", "earlier", "what did we", "history of", "past"
])

_LEARNING_TERMS = frozenset([
    "pattern", "similar", "consolidate", "common", "recurring",
    "repeated", "frequent", "trend", "consistent", "like before"
])


def infer_mode_from_query(query: str) -> RetrievalMode:
    """
    Heuristically infer the best retrieval mode from query text.

    Uses keyword matching to detect the cognitive task type.
    Falls back to PRECISE mode for general queries.

    Args:
        query: The search query or task description

    Returns:
        The inferred RetrievalMode
    """
    query_lower = query.lower()

    # Check for diagnostic terms (errors, bugs, fixes)
    if any(term in query_lower for term in _DIAGNOSTIC_TERMS):
        return RetrievalMode.DIAGNOSTIC

    # Check for broad/planning terms
    if any(term in query_lower for term in _BROAD_TERMS):
        return RetrievalMode.BROAD

    # Check for recall terms (historical lookup)
    if any(term in query_lower for term in _RECALL_TERMS):
        return RetrievalMode.RECALL

    # Check for learning/pattern terms
    if any(term in query_lower for term in _LEARNING_TERMS):
        return RetrievalMode.LEARNING

    # Default to PRECISE for execution/implementation
    return RetrievalMode.PRECISE


def get_mode_config(mode: RetrievalMode) -> ModeConfig:
    """
    Get the configuration for a specific mode.

    Args:
        mode: The retrieval mode

    Returns:
        ModeConfig for the specified mode
    """
    return MODE_CONFIGS[mode]


def get_mode_reason(query: str, mode: RetrievalMode) -> str:
    """
    Explain why a particular mode was inferred.

    Args:
        query: The original query
        mode: The inferred mode

    Returns:
        Human-readable explanation
    """
    query_lower = query.lower()

    if mode == RetrievalMode.DIAGNOSTIC:
        matched = [t for t in _DIAGNOSTIC_TERMS if t in query_lower]
        if matched:
            return f"Query contains diagnostic terms: {', '.join(matched[:3])}"
        return "Query appears to be about debugging or troubleshooting"

    if mode == RetrievalMode.BROAD:
        matched = [t for t in _BROAD_TERMS if t in query_lower]
        if matched:
            return f"Query contains planning/exploration terms: {', '.join(matched[:3])}"
        return "Query appears to be exploratory or planning-related"

    if mode == RetrievalMode.RECALL:
        matched = [t for t in _RECALL_TERMS if t in query_lower]
        if matched:
            return f"Query contains recall terms: {', '.join(matched[:3])}"
        return "Query appears to be looking for past events or decisions"

    if mode == RetrievalMode.LEARNING:
        matched = [t for t in _LEARNING_TERMS if t in query_lower]
        if matched:
            return f"Query contains pattern terms: {', '.join(matched[:3])}"
        return "Query appears to be looking for patterns or similarities"

    return "Default mode for implementation/execution queries"


def create_custom_mode(
    base_mode: RetrievalMode,
    **overrides
) -> ModeConfig:
    """
    Create a custom mode configuration based on an existing mode.

    Args:
        base_mode: The mode to use as a base
        **overrides: Fields to override (top_k, min_confidence, etc.)

    Returns:
        New ModeConfig with overrides applied
    """
    base = get_mode_config(base_mode)

    return ModeConfig(
        top_k=overrides.get("top_k", base.top_k),
        min_confidence=overrides.get("min_confidence", base.min_confidence),
        weights=overrides.get("weights", base.weights.copy()),
        include_anti_patterns=overrides.get(
            "include_anti_patterns", base.include_anti_patterns
        ),
        diversity_factor=overrides.get("diversity_factor", base.diversity_factor),
        prioritize_failures=overrides.get(
            "prioritize_failures", base.prioritize_failures
        ),
        cluster_similar=overrides.get("cluster_similar", base.cluster_similar),
        exact_match_boost=overrides.get("exact_match_boost", base.exact_match_boost),
    )


def validate_mode_config(config: ModeConfig) -> List[str]:
    """
    Validate a mode configuration.

    Args:
        config: The configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if config.top_k < 1:
        errors.append("top_k must be at least 1")

    if not (0.0 <= config.min_confidence <= 1.0):
        errors.append("min_confidence must be between 0.0 and 1.0")

    if not (0.0 <= config.diversity_factor <= 1.0):
        errors.append("diversity_factor must be between 0.0 and 1.0")

    if config.exact_match_boost < 0:
        errors.append("exact_match_boost must be non-negative")

    if config.weights:
        total = sum(config.weights.values())
        if not (0.99 <= total <= 1.01):
            errors.append(f"weights must sum to 1.0 (got {total:.2f})")

        required_keys = {"similarity", "recency", "success_rate", "confidence"}
        missing = required_keys - set(config.weights.keys())
        if missing:
            errors.append(f"weights missing keys: {missing}")

    return errors
