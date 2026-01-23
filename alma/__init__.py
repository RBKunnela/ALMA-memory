"""
ALMA - Agent Learning Memory Architecture

Persistent memory system for AI agents that learn and improve over time
through structured memory layers - without model weight updates.
"""

__version__ = "0.1.0"

from alma.core import ALMA
from alma.types import (
    Heuristic,
    Outcome,
    UserPreference,
    DomainKnowledge,
    AntiPattern,
    MemorySlice,
    MemoryScope,
)

__all__ = [
    "ALMA",
    "Heuristic",
    "Outcome",
    "UserPreference",
    "DomainKnowledge",
    "AntiPattern",
    "MemorySlice",
    "MemoryScope",
]
