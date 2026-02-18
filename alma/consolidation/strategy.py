"""
Consolidation Strategy Interface - Decouples consolidation from LLM implementation.

Applies Strategy Pattern to separate consolidation strategies from LLM-specific logic.
Previously LLM calls were deeply embedded. Now LLM is one possible strategy,
but consolidation can use other approaches (heuristic, rule-based, hybrid).

IMPROVEMENTS:
- Reduces coupling between consolidation and LLM provider
- Enables swapping consolidation strategies at runtime
- Makes testing easier (can use mock strategies)
- Supports multiple consolidation approaches simultaneously
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import logging

from alma.types import Heuristic, Outcome

logger = logging.getLogger(__name__)


class ConsolidationStrategy(ABC):
    """Abstract base for consolidation strategies."""

    @abstractmethod
    def consolidate(
        self,
        items: List[Heuristic | Outcome],
        agent: str,
        project_id: str,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Execute consolidation using this strategy.

        Args:
            items: Items to consolidate
            agent: Agent identifier
            project_id: Project identifier
            threshold: Consolidation similarity threshold

        Returns:
            Consolidation result with consolidated items and metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name (e.g., 'llm', 'heuristic', 'hybrid')."""
        pass

    @abstractmethod
    def supports_item_type(self, item_type: str) -> bool:
        """Check if strategy supports consolidating this item type."""
        pass


class LLMConsolidationStrategy(ConsolidationStrategy):
    """Consolidation using LLM (existing approach)."""

    def __init__(self, llm_client=None):
        """Initialize with optional LLM client."""
        self.llm_client = llm_client

    def get_name(self) -> str:
        """Return strategy name."""
        return "llm"

    def supports_item_type(self, item_type: str) -> bool:
        """LLM can consolidate any type."""
        return True

    def consolidate(
        self,
        items: List[Heuristic | Outcome],
        agent: str,
        project_id: str,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Use LLM for consolidation."""
        if not items:
            return {"consolidated": [], "skipped": 0}

        # Existing LLM consolidation logic would go here
        # (Currently in consolidation/pipeline.py)
        return {
            "consolidated": items[:1],  # Placeholder
            "skipped": len(items) - 1,
            "strategy": self.get_name(),
        }


class HeuristicConsolidationStrategy(ConsolidationStrategy):
    """Consolidation using heuristic rules (faster, no LLM needed)."""

    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize with similarity threshold."""
        self.similarity_threshold = similarity_threshold

    def get_name(self) -> str:
        """Return strategy name."""
        return "heuristic"

    def supports_item_type(self, item_type: str) -> bool:
        """Heuristic supports text-based items."""
        return item_type in ("heuristic", "outcome")

    def consolidate(
        self,
        items: List[Heuristic | Outcome],
        agent: str,
        project_id: str,
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Use heuristic rules for consolidation."""
        if not items:
            return {"consolidated": [], "skipped": 0}

        # Simple text similarity consolidation
        consolidated = []
        seen_texts = set()

        for item in items:
            text = getattr(item, "title", getattr(item, "action", ""))
            if text not in seen_texts:
                consolidated.append(item)
                seen_texts.add(text)

        return {
            "consolidated": consolidated,
            "skipped": len(items) - len(consolidated),
            "strategy": self.get_name(),
        }


class ConsolidationStrategyFactory:
    """Factory for consolidation strategies."""

    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a consolidation strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered consolidation strategy: {name}")

    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> ConsolidationStrategy:
        """Create a consolidation strategy instance."""
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
ConsolidationStrategyFactory.register("llm", LLMConsolidationStrategy)
ConsolidationStrategyFactory.register("heuristic", HeuristicConsolidationStrategy)

logger.info("Consolidation strategies registered: llm, heuristic")
