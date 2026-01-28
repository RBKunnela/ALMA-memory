"""
ALMA Confidence Module.

Forward-looking confidence signals for strategies.
Not just "this worked before" but "this will likely work now."

Inspired by Ilya Sutskever's insight: emotions are forward-looking value functions,
while reinforcement learning is backward-looking.

Usage:
    from alma.confidence import ConfidenceEngine, ConfidenceSignal

    engine = ConfidenceEngine(alma)

    # Assess a strategy
    signal = engine.assess_strategy(
        strategy="Use incremental validation for forms",
        context="Testing a registration form with 5 fields",
        agent="Helena",
    )

    print(f"Confidence: {signal.confidence_score:.0%}")
    print(f"Recommendation: {signal.recommendation}")

    # Rank multiple strategies
    rankings = engine.rank_strategies(
        strategies=["Strategy A", "Strategy B"],
        context="Current context",
        agent="Helena",
    )
"""

from alma.confidence.engine import ConfidenceEngine
from alma.confidence.types import (
    ConfidenceSignal,
    OpportunitySignal,
    Recommendation,
    RiskSignal,
)

__all__ = [
    "ConfidenceSignal",
    "ConfidenceEngine",
    "OpportunitySignal",
    "Recommendation",
    "RiskSignal",
]
