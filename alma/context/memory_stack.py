"""
4-Layer MemoryStack for ALMA.

Inspired by MemPalace layers.py (MIT License).

Provides token-efficient context loading via a 4-layer memory hierarchy:

    Layer 0: Identity       (~100 tokens)   - Always loaded. "Who am I?"
    Layer 1: Essential Story (~500-800)      - Always loaded. Top memories by confidence.
    Layer 2: On-Demand      (~200-500 each)  - Loaded when a topic/domain comes up.
    Layer 3: Deep Search    (unlimited)      - Full semantic search via ALMA retrieve().

Wake-up cost: ~600-900 tokens (L0+L1). Leaves 95%+ of context free.

Usage:
    from alma.context import MemoryStack

    alma = ALMA.from_config("config.yaml")
    stack = MemoryStack(alma)

    # Session start: inject L0+L1
    system_context = stack.wake_up()

    # Mid-conversation: topic-specific recall
    auth_context = stack.recall("authentication flow", layer=2)

    # Deep search
    results = stack.recall("JWT token expiry edge cases", layer=3)

    # Format everything for prompt injection
    prompt = stack.to_prompt(max_tokens=2000)
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from alma.context.identity import IdentityManager
from alma.observability.logging import get_logger
from alma.types import MemorySlice
from alma.utils.tokenizer import TokenEstimator, estimate_tokens_simple

if TYPE_CHECKING:
    from alma.core import ALMA

logger = get_logger(__name__)

# Layer constants
LAYER_IDENTITY = 0
LAYER_ESSENTIAL = 1
LAYER_ON_DEMAND = 2
LAYER_DEEP_SEARCH = 3

# Default token budgets per layer
_DEFAULT_L1_MAX_TOKENS = 800
_DEFAULT_L2_MAX_TOKENS = 500


class ContextLayer:
    """
    Represents a single layer in the MemoryStack.

    Tracks the layer's content, token usage, and source metadata.

    Attributes:
        level: Layer number (0-3).
        name: Human-readable layer name.
        content: The rendered text content for this layer.
        token_count: Estimated token count for the content.
    """

    def __init__(self, level: int, name: str):
        self.level = level
        self.name = name
        self.content: str = ""
        self.token_count: int = 0
        self._memory_slice: Optional[MemorySlice] = None

    def set_content(self, text: str) -> None:
        """
        Set the layer content and update token estimate.

        Args:
            text: The rendered text content.
        """
        self.content = text
        self.token_count = estimate_tokens_simple(text)

    def set_from_slice(self, memory_slice: MemorySlice) -> None:
        """
        Set layer content from a MemorySlice.

        Args:
            memory_slice: The retrieval result to render.
        """
        self._memory_slice = memory_slice
        self.content = memory_slice.to_prompt(max_tokens=_DEFAULT_L2_MAX_TOKENS)
        self.token_count = estimate_tokens_simple(self.content)

    @property
    def is_loaded(self) -> bool:
        """Whether this layer has content loaded."""
        return bool(self.content)


class MemoryStack:
    """
    4-layer memory hierarchy wrapping ALMA's retrieval engine.

    Provides token-efficient context loading by organizing memories
    into layers with different loading strategies:

    - L0 (Identity): Always loaded from a text file (~100 tokens).
    - L1 (Essential Story): Always loaded. Top memories by confidence.
    - L2 (On-Demand): Loaded on topic/domain queries with scope filtering.
    - L3 (Deep Search): Full semantic search via ALMA's retrieve().

    Args:
        alma_instance: A configured ALMA instance for retrieval.
        identity_path: Path to the identity file. Defaults to
            ~/.alma/identity.txt.
        agent: Agent name for retrieval. Defaults to "default".
        l1_max_tokens: Token budget for Layer 1. Defaults to 800.
    """

    def __init__(
        self,
        alma_instance: "ALMA",
        identity_path: Optional[str] = None,
        agent: str = "default",
        l1_max_tokens: int = _DEFAULT_L1_MAX_TOKENS,
    ):
        self._alma = alma_instance
        self._agent = agent
        self._l1_max_tokens = l1_max_tokens

        # Initialize layers
        self._identity_mgr = IdentityManager(identity_path=identity_path)
        self._layers: Dict[int, ContextLayer] = {
            LAYER_IDENTITY: ContextLayer(LAYER_IDENTITY, "Identity"),
            LAYER_ESSENTIAL: ContextLayer(LAYER_ESSENTIAL, "Essential Story"),
            LAYER_ON_DEMAND: ContextLayer(LAYER_ON_DEMAND, "On-Demand"),
            LAYER_DEEP_SEARCH: ContextLayer(LAYER_DEEP_SEARCH, "Deep Search"),
        }

        # Track active recall results for to_prompt()
        self._active_recalls: List[ContextLayer] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wake_up(
        self,
        domain: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Return L0 + L1 context (~600-900 tokens). Called at session start.

        Loads the identity file (L0) and retrieves the top memories
        scored by confidence (L1) to provide essential context.

        Args:
            domain: Optional domain filter for L1 retrieval.
            user_id: Optional user ID for preference retrieval.

        Returns:
            Combined L0 + L1 text, ready for prompt injection.
        """
        # L0: Identity
        l0 = self._layers[LAYER_IDENTITY]
        identity_text = self._identity_mgr.load()
        l0.set_content(f"## Identity\n{identity_text}")

        # L1: Essential Story — top memories by confidence
        l1 = self._layers[LAYER_ESSENTIAL]
        l1_query = domain or "essential context overview"

        try:
            memory_slice = self._alma.retrieve(
                task=l1_query,
                agent=self._agent,
                user_id=user_id,
                top_k=10,
            )
            l1_text = self._format_essential_story(memory_slice)
            l1.set_content(l1_text)
        except Exception as exc:
            logger.warning(
                "Failed to load L1 essential story",
                error=str(exc),
            )
            l1.set_content("## Essential Story\nNo memories available yet.")

        parts = [l0.content, "", l1.content]
        result = "\n".join(parts)

        logger.info(
            "Wake-up complete",
            l0_tokens=l0.token_count,
            l1_tokens=l1.token_count,
            total_tokens=l0.token_count + l1.token_count,
        )

        return result

    def recall(
        self,
        query: str,
        layer: Optional[int] = None,
        top_k: int = 5,
        domain: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Retrieve memories from specified layer (or auto-select).

        Auto-selection logic:
        - Short queries (< 30 chars) or domain-specific -> L2 (On-Demand)
        - Complex queries or no domain hint -> L3 (Deep Search)

        Args:
            query: The search query.
            layer: Explicit layer (2 or 3). Auto-selects if None.
            top_k: Maximum number of results.
            domain: Optional domain/topic filter for L2.
            user_id: Optional user ID for preference retrieval.

        Returns:
            Formatted memory text from the selected layer.
        """
        if layer is not None:
            effective_layer = layer
        else:
            # Auto-select: short/domain queries -> L2, complex -> L3
            if domain or len(query) < 30:
                effective_layer = LAYER_ON_DEMAND
            else:
                effective_layer = LAYER_DEEP_SEARCH

        if effective_layer == LAYER_ON_DEMAND:
            return self._recall_on_demand(
                query, top_k=top_k, domain=domain, user_id=user_id
            )
        elif effective_layer == LAYER_DEEP_SEARCH:
            return self._recall_deep_search(query, top_k=top_k, user_id=user_id)
        elif effective_layer == LAYER_IDENTITY:
            return self.identity
        elif effective_layer == LAYER_ESSENTIAL:
            if not self._layers[LAYER_ESSENTIAL].is_loaded:
                self.wake_up()
            return self._layers[LAYER_ESSENTIAL].content
        else:
            return self._recall_deep_search(query, top_k=top_k, user_id=user_id)

    def to_prompt(
        self,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> str:
        """
        Format all loaded context as a prompt string.

        Respects the max_tokens budget by including layers in priority
        order: L0 (always), L1 (always), then active recalls (L2/L3)
        until budget is exhausted.

        Args:
            max_tokens: Maximum token budget for the output.
            model: Optional model name for accurate tokenization.

        Returns:
            Formatted prompt string within token budget.
        """
        estimator = TokenEstimator(model=model) if model else TokenEstimator()

        sections: List[str] = []
        tokens_used = 0

        # L0: Identity (always included)
        l0 = self._layers[LAYER_IDENTITY]
        if l0.is_loaded:
            tokens_used += l0.token_count
            sections.append(l0.content)

        # L1: Essential Story (always included if loaded)
        l1 = self._layers[LAYER_ESSENTIAL]
        if l1.is_loaded and tokens_used + l1.token_count <= max_tokens:
            tokens_used += l1.token_count
            sections.append(l1.content)

        # Active recalls (L2/L3) in order of addition
        for recall_layer in self._active_recalls:
            if tokens_used + recall_layer.token_count <= max_tokens:
                tokens_used += recall_layer.token_count
                sections.append(recall_layer.content)
            else:
                # Budget exhausted — truncate remaining
                remaining = max_tokens - tokens_used
                if remaining > 50:
                    truncated = estimator.truncate_to_token_limit(
                        recall_layer.content,
                        max_tokens=remaining,
                        suffix="\n[truncated — token budget reached]",
                    )
                    sections.append(truncated)
                break

        return "\n\n".join(sections)

    @property
    def identity(self) -> str:
        """Return Layer 0 identity text."""
        return self._identity_mgr.text

    @property
    def token_usage(self) -> dict:
        """
        Return per-layer token counts.

        Returns:
            Dict with layer names as keys and token counts as values,
            plus a 'total' key.
        """
        usage: Dict[str, Any] = {}
        total = 0

        for level, layer in self._layers.items():
            usage[f"L{level}_{layer.name.lower().replace(' ', '_')}"] = (
                layer.token_count
            )
            total += layer.token_count

        # Include active recall layers
        recall_tokens = sum(r.token_count for r in self._active_recalls)
        usage["active_recalls"] = recall_tokens
        total += recall_tokens

        usage["total"] = total
        return usage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recall_on_demand(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """L2: Filtered retrieval for a specific topic/domain."""
        try:
            memory_slice = self._alma.retrieve(
                task=query,
                agent=self._agent,
                user_id=user_id,
                top_k=top_k,
            )

            layer = ContextLayer(LAYER_ON_DEMAND, f"On-Demand: {query[:40]}")
            text = self._format_on_demand(memory_slice, query)
            layer.set_content(text)

            # Track for to_prompt()
            self._active_recalls.append(layer)
            self._layers[LAYER_ON_DEMAND] = layer

            logger.info(
                "L2 on-demand recall",
                query=query[:50],
                items=memory_slice.total_items,
                tokens=layer.token_count,
            )

            return text

        except Exception as exc:
            logger.warning(
                "L2 on-demand recall failed",
                query=query[:50],
                error=str(exc),
            )
            return f"No results found for: {query}"

    def _recall_deep_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> str:
        """L3: Full semantic search via ALMA's retrieve()."""
        try:
            memory_slice = self._alma.retrieve(
                task=query,
                agent=self._agent,
                user_id=user_id,
                top_k=top_k,
            )

            layer = ContextLayer(LAYER_DEEP_SEARCH, f"Deep Search: {query[:40]}")
            text = memory_slice.to_prompt(max_tokens=_DEFAULT_L2_MAX_TOKENS)
            if not text.strip():
                text = f'No deep search results for: "{query}"'
            layer.set_content(text)

            # Track for to_prompt()
            self._active_recalls.append(layer)
            self._layers[LAYER_DEEP_SEARCH] = layer

            logger.info(
                "L3 deep search",
                query=query[:50],
                items=memory_slice.total_items,
                tokens=layer.token_count,
            )

            return text

        except Exception as exc:
            logger.warning(
                "L3 deep search failed",
                query=query[:50],
                error=str(exc),
            )
            return f"Search failed for: {query}"

    def _format_essential_story(
        self,
        memory_slice: MemorySlice,
    ) -> str:
        """
        Format L1 essential story from a MemorySlice.

        Groups by type and truncates to the L1 token budget.
        """
        lines = ["## Essential Story"]

        if memory_slice.heuristics:
            lines.append("\n[Strategies]")
            for h in sorted(memory_slice.heuristics, key=lambda x: -x.confidence)[:5]:
                snippet = h.strategy.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                lines.append(f"  - {snippet} (confidence: {h.confidence:.0%})")

        if memory_slice.domain_knowledge:
            lines.append("\n[Domain Knowledge]")
            for dk in memory_slice.domain_knowledge[:5]:
                snippet = dk.fact.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                lines.append(f"  - {snippet}")

        if memory_slice.anti_patterns:
            lines.append("\n[Anti-Patterns]")
            for ap in memory_slice.anti_patterns[:3]:
                lines.append(f"  - Avoid: {ap.pattern}")

        if memory_slice.preferences:
            lines.append("\n[User Preferences]")
            for p in memory_slice.preferences[:3]:
                lines.append(f"  - {p.preference}")

        text = "\n".join(lines)

        # Enforce L1 token budget
        estimated = estimate_tokens_simple(text)
        if estimated > self._l1_max_tokens:
            # Rough truncation: 4 chars per token
            max_chars = self._l1_max_tokens * 4
            text = text[:max_chars] + "\n  ... (more in L3 search)"

        return text

    def _format_on_demand(
        self,
        memory_slice: MemorySlice,
        query: str,
    ) -> str:
        """Format L2 on-demand retrieval results."""
        total = memory_slice.total_items
        if total == 0:
            return f"No on-demand results for: {query}"

        header = f"## On-Demand ({total} items) — {query[:40]}"
        body = memory_slice.to_prompt(max_tokens=_DEFAULT_L2_MAX_TOKENS)

        return f"{header}\n{body}" if body.strip() else header
