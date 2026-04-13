"""
ALMA Context Package — 4-Layer MemoryStack.

Inspired by MemPalace layers.py (MIT License).

Provides token-efficient context loading via a layered memory hierarchy:

    Layer 0: Identity       (~100 tokens)   - Always loaded. "Who am I?"
    Layer 1: Essential Story (~500-800)      - Always loaded. Top memories.
    Layer 2: On-Demand      (~200-500 each)  - Topic/domain filtered.
    Layer 3: Deep Search    (unlimited)      - Full semantic search.

Usage:
    from alma.context import MemoryStack

    alma = ALMA.from_config("config.yaml")
    stack = MemoryStack(alma)
    print(stack.wake_up())       # L0 + L1 (~600-900 tokens)
    print(stack.recall("auth"))  # L2 on-demand
"""

from alma.context.identity import IdentityManager
from alma.context.memory_stack import (
    LAYER_DEEP_SEARCH,
    LAYER_ESSENTIAL,
    LAYER_IDENTITY,
    LAYER_ON_DEMAND,
    ContextLayer,
    MemoryStack,
)

__all__ = [
    "ContextLayer",
    "IdentityManager",
    "LAYER_DEEP_SEARCH",
    "LAYER_ESSENTIAL",
    "LAYER_IDENTITY",
    "LAYER_ON_DEMAND",
    "MemoryStack",
]
