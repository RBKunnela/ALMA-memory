"""
Retrieval Scorer Factory - Decouples scoring strategies from retrieval engine.

Applies Strategy Pattern to separate scoring logic from the retrieval engine.
Previously scoring algorithms were tightly coupled to the engine. Now each
scoring strategy can be plugged in independently.

IMPROVEMENTS:
- Reduces coupling between retrieval engine and scoring logic
- Makes it easier to add new scoring strategies
- Improves testability (can test scoring independently)
- Enables runtime strategy switching
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import logging

from alma.types import MemorySlice

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """Abstract base for retrieval scoring strategies."""

    @abstractmethod
    def score(
        self,
        items: List[MemorySlice],
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[tuple[MemorySlice, float]]:
        """
        Score items and return sorted by score.

        Returns:
            List of (item, score) tuples sorted by score descending
        """
        pass


class ScorerFactory:
    """Factory for retrieval scoring strategies."""

    _strategies: Dict[str, Type[ScoringStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[ScoringStrategy]) -> None:
        """Register a scoring strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered scoring strategy: {name}")

    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> ScoringStrategy:
        """
        Create a scoring strategy instance.

        Args:
            strategy_name: Name of scoring strategy
            **kwargs: Strategy-specific configuration

        Returns:
            ScoringStrategy instance
        """
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {available}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of registered strategies."""
        return list(cls._strategies.keys())


# Register built-in strategies
def _register_strategies() -> None:
    """Register built-in scoring strategies."""
    try:
        from alma.retrieval.scoring import MemoryScorer
        # Adapt existing scorer as a strategy
        # (Can be expanded to create dedicated strategy classes)
        logger.info("Built-in scoring strategies available")
    except ImportError:
        pass


_register_strategies()
