"""
ALMA Token Estimation Module.

Provides accurate token counting using tiktoken for OpenAI models
and configurable token budgets per model type.

This module addresses Issue #11 (LOW-001): Token Estimation is Rough.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


class ModelFamily(Enum):
    """Model families with different tokenization schemes."""

    GPT4 = "gpt4"  # GPT-4, GPT-4 Turbo, GPT-4o
    GPT35 = "gpt35"  # GPT-3.5 Turbo
    CLAUDE = "claude"  # Claude 3.x models
    GEMINI = "gemini"  # Google Gemini models
    LLAMA = "llama"  # Meta Llama models
    MISTRAL = "mistral"  # Mistral models
    LOCAL = "local"  # Local/open-source models
    UNKNOWN = "unknown"  # Fallback


@dataclass
class ModelTokenBudget:
    """
    Token budget configuration for a model.

    Attributes:
        context_window: Maximum context window size for the model
        memory_budget: Recommended tokens to allocate for ALMA memories
        response_reserve: Tokens to reserve for model response
        safety_margin: Additional safety margin (percentage, 0.0-1.0)
    """

    context_window: int
    memory_budget: int
    response_reserve: int = 4096
    safety_margin: float = 0.1

    @property
    def effective_memory_budget(self) -> int:
        """Calculate effective memory budget after safety margin."""
        return int(self.memory_budget * (1 - self.safety_margin))


# Default token budgets per model
DEFAULT_TOKEN_BUDGETS: Dict[str, ModelTokenBudget] = {
    # OpenAI GPT-4 family
    "gpt-4": ModelTokenBudget(
        context_window=8192,
        memory_budget=2000,
        response_reserve=2048,
    ),
    "gpt-4-32k": ModelTokenBudget(
        context_window=32768,
        memory_budget=4000,
        response_reserve=4096,
    ),
    "gpt-4-turbo": ModelTokenBudget(
        context_window=128000,
        memory_budget=8000,
        response_reserve=4096,
    ),
    "gpt-4o": ModelTokenBudget(
        context_window=128000,
        memory_budget=8000,
        response_reserve=4096,
    ),
    "gpt-4o-mini": ModelTokenBudget(
        context_window=128000,
        memory_budget=8000,
        response_reserve=4096,
    ),
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": ModelTokenBudget(
        context_window=16385,
        memory_budget=2000,
        response_reserve=2048,
    ),
    "gpt-3.5-turbo-16k": ModelTokenBudget(
        context_window=16385,
        memory_budget=4000,
        response_reserve=4096,
    ),
    # Anthropic Claude family
    "claude-3-opus": ModelTokenBudget(
        context_window=200000,
        memory_budget=10000,
        response_reserve=4096,
    ),
    "claude-3-sonnet": ModelTokenBudget(
        context_window=200000,
        memory_budget=8000,
        response_reserve=4096,
    ),
    "claude-3-haiku": ModelTokenBudget(
        context_window=200000,
        memory_budget=6000,
        response_reserve=4096,
    ),
    "claude-3.5-sonnet": ModelTokenBudget(
        context_window=200000,
        memory_budget=8000,
        response_reserve=4096,
    ),
    "claude-3.5-haiku": ModelTokenBudget(
        context_window=200000,
        memory_budget=6000,
        response_reserve=4096,
    ),
    # Google Gemini family
    "gemini-pro": ModelTokenBudget(
        context_window=32768,
        memory_budget=4000,
        response_reserve=4096,
    ),
    "gemini-1.5-pro": ModelTokenBudget(
        context_window=1000000,
        memory_budget=10000,
        response_reserve=8192,
    ),
    "gemini-1.5-flash": ModelTokenBudget(
        context_window=1000000,
        memory_budget=8000,
        response_reserve=8192,
    ),
    # Local/open-source models (conservative defaults)
    "llama-2-7b": ModelTokenBudget(
        context_window=4096,
        memory_budget=1000,
        response_reserve=1024,
    ),
    "llama-2-70b": ModelTokenBudget(
        context_window=4096,
        memory_budget=1000,
        response_reserve=1024,
    ),
    "llama-3-8b": ModelTokenBudget(
        context_window=8192,
        memory_budget=2000,
        response_reserve=2048,
    ),
    "llama-3-70b": ModelTokenBudget(
        context_window=8192,
        memory_budget=2000,
        response_reserve=2048,
    ),
    "mistral-7b": ModelTokenBudget(
        context_window=8192,
        memory_budget=2000,
        response_reserve=2048,
    ),
    "mixtral-8x7b": ModelTokenBudget(
        context_window=32768,
        memory_budget=4000,
        response_reserve=4096,
    ),
    # Default fallback
    "default": ModelTokenBudget(
        context_window=8192,
        memory_budget=2000,
        response_reserve=2048,
    ),
}


class TokenEstimator:
    """
    Accurate token estimation using tiktoken for OpenAI-compatible tokenization.

    For non-OpenAI models, uses model-specific approximations based on
    documented token-to-character ratios.

    Usage:
        estimator = TokenEstimator(model="gpt-4")
        token_count = estimator.count_tokens("Hello, world!")
        budget = estimator.get_token_budget()
    """

    # Tiktoken encoding cache
    _encoding_cache: Dict[str, "tiktoken.Encoding"] = {}  # type: ignore

    # Approximate tokens-per-character ratios for fallback estimation
    # These are based on documented model characteristics
    TOKENS_PER_CHAR_RATIOS: Dict[ModelFamily, float] = {
        ModelFamily.GPT4: 0.25,  # ~4 chars per token on average
        ModelFamily.GPT35: 0.25,
        ModelFamily.CLAUDE: 0.28,  # Claude tends to be slightly more token-dense
        ModelFamily.GEMINI: 0.25,
        ModelFamily.LLAMA: 0.27,  # Llama tokenizer is similar to GPT
        ModelFamily.MISTRAL: 0.27,
        ModelFamily.LOCAL: 0.25,
        ModelFamily.UNKNOWN: 0.25,
    }

    def __init__(
        self,
        model: str = "gpt-4",
        custom_budget: Optional[ModelTokenBudget] = None,
    ):
        """
        Initialize token estimator.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-sonnet", "llama-3-8b")
            custom_budget: Optional custom token budget to override defaults
        """
        self.model = model.lower()
        self.model_family = self._detect_model_family(self.model)
        self._tiktoken_available = self._check_tiktoken()
        self._encoding = self._get_encoding() if self._tiktoken_available else None
        self._custom_budget = custom_budget

    def _check_tiktoken(self) -> bool:
        """Check if tiktoken is available."""
        try:
            import tiktoken  # noqa: F401

            return True
        except ImportError:
            logger.debug("tiktoken not available, using approximate token estimation")
            return False

    def _detect_model_family(self, model: str) -> ModelFamily:
        """Detect the model family from model name."""
        model_lower = model.lower()

        if any(x in model_lower for x in ["gpt-4", "gpt4"]):
            return ModelFamily.GPT4
        elif any(x in model_lower for x in ["gpt-3.5", "gpt35"]):
            return ModelFamily.GPT35
        elif "claude" in model_lower:
            return ModelFamily.CLAUDE
        elif "gemini" in model_lower:
            return ModelFamily.GEMINI
        elif "llama" in model_lower:
            return ModelFamily.LLAMA
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return ModelFamily.MISTRAL
        else:
            return ModelFamily.UNKNOWN

    def _get_encoding(self) -> Optional["tiktoken.Encoding"]:  # type: ignore
        """Get tiktoken encoding for the model."""
        if not self._tiktoken_available:
            return None

        import tiktoken

        # Map model families to tiktoken encodings
        encoding_map = {
            ModelFamily.GPT4: "cl100k_base",
            ModelFamily.GPT35: "cl100k_base",
            ModelFamily.CLAUDE: "cl100k_base",  # Claude uses similar tokenization
            ModelFamily.GEMINI: "cl100k_base",  # Approximate
            ModelFamily.LLAMA: "cl100k_base",  # Approximate
            ModelFamily.MISTRAL: "cl100k_base",  # Approximate
            ModelFamily.LOCAL: "cl100k_base",
            ModelFamily.UNKNOWN: "cl100k_base",
        }

        encoding_name = encoding_map.get(self.model_family, "cl100k_base")

        # Use cached encoding if available
        if encoding_name not in self._encoding_cache:
            try:
                self._encoding_cache[encoding_name] = tiktoken.get_encoding(
                    encoding_name
                )
            except Exception as e:
                logger.warning(f"Failed to get tiktoken encoding: {e}")
                return None

        return self._encoding_cache[encoding_name]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or fallback estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Use tiktoken if available
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text))
            except Exception as e:
                logger.debug(f"tiktoken encoding failed, using fallback: {e}")

        # Fallback: character-based estimation
        ratio = self.TOKENS_PER_CHAR_RATIOS.get(self.model_family, 0.25)
        return int(len(text) * ratio)

    def count_tokens_for_messages(
        self,
        messages: list[dict[str, str]],
    ) -> int:
        """
        Count tokens for a list of messages (chat format).

        Accounts for message formatting overhead.

        Args:
            messages: List of message dicts with "role" and "content" keys

        Returns:
            Estimated token count including formatting overhead
        """
        total = 0

        # Per-message overhead varies by model
        # GPT-4/3.5: ~4 tokens per message for formatting
        # Claude: ~3 tokens per message
        overhead_per_message = (
            4 if self.model_family in (ModelFamily.GPT4, ModelFamily.GPT35) else 3
        )

        for message in messages:
            content = message.get("content", "")
            total += self.count_tokens(content)
            total += overhead_per_message

        # Add reply priming overhead
        total += 3

        return total

    def get_token_budget(self) -> ModelTokenBudget:
        """
        Get the token budget for the current model.

        Returns custom budget if set, otherwise returns default for model.
        """
        if self._custom_budget:
            return self._custom_budget

        # Try exact model match first
        if self.model in DEFAULT_TOKEN_BUDGETS:
            return DEFAULT_TOKEN_BUDGETS[self.model]

        # Try partial matches - prefer longer key matches
        best_match = None
        best_match_len = 0

        for key, budget in DEFAULT_TOKEN_BUDGETS.items():
            if key == "default":
                continue
            if key in self.model:
                if len(key) > best_match_len:
                    best_match = budget
                    best_match_len = len(key)
            elif self.model in key:
                if len(self.model) > best_match_len:
                    best_match = budget
                    best_match_len = len(self.model)

        if best_match:
            return best_match

        # Return default
        return DEFAULT_TOKEN_BUDGETS["default"]

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "\n[truncated]",
    ) -> str:
        """
        Truncate text to fit within a token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            suffix: Suffix to append if truncated

        Returns:
            Truncated text with suffix if it exceeded the limit
        """
        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Reserve tokens for suffix
        suffix_tokens = self.count_tokens(suffix)
        target_tokens = max_tokens - suffix_tokens

        if target_tokens <= 0:
            return suffix

        # Binary search for the right truncation point
        if self._encoding is not None:
            try:
                tokens = self._encoding.encode(text)
                truncated_tokens = tokens[:target_tokens]
                return self._encoding.decode(truncated_tokens) + suffix
            except Exception:
                pass

        # Fallback: character-based truncation
        ratio = self.TOKENS_PER_CHAR_RATIOS.get(self.model_family, 0.25)
        target_chars = int(target_tokens / ratio)
        return text[:target_chars] + suffix

    def estimate_remaining_budget(
        self,
        used_tokens: int,
        include_response_reserve: bool = True,
    ) -> int:
        """
        Estimate remaining token budget for memories.

        Args:
            used_tokens: Tokens already used in context
            include_response_reserve: Whether to subtract response reserve

        Returns:
            Remaining tokens available for memories
        """
        budget = self.get_token_budget()
        available = budget.context_window - used_tokens

        if include_response_reserve:
            available -= budget.response_reserve

        # Apply safety margin
        available = int(available * (1 - budget.safety_margin))

        return max(0, min(available, budget.effective_memory_budget))


def get_token_estimator(
    model: str = "gpt-4",
    custom_budget: Optional[ModelTokenBudget] = None,
) -> TokenEstimator:
    """
    Factory function to create a TokenEstimator.

    Args:
        model: Model name
        custom_budget: Optional custom token budget

    Returns:
        Configured TokenEstimator instance
    """
    return TokenEstimator(model=model, custom_budget=custom_budget)


def get_default_token_budget(model: str = "gpt-4") -> ModelTokenBudget:
    """
    Get the default token budget for a model.

    Args:
        model: Model name

    Returns:
        Token budget configuration
    """
    model_lower = model.lower()

    # Try exact match
    if model_lower in DEFAULT_TOKEN_BUDGETS:
        return DEFAULT_TOKEN_BUDGETS[model_lower]

    # Try partial match - prefer longer key matches to avoid e.g. "gpt-4" matching "gpt-4o"
    best_match = None
    best_match_len = 0

    for key, budget in DEFAULT_TOKEN_BUDGETS.items():
        if key == "default":
            continue
        if key in model_lower:
            if len(key) > best_match_len:
                best_match = budget
                best_match_len = len(key)
        elif model_lower in key:
            if len(model_lower) > best_match_len:
                best_match = budget
                best_match_len = len(model_lower)

    if best_match:
        return best_match

    return DEFAULT_TOKEN_BUDGETS["default"]


def estimate_tokens_simple(text: str) -> int:
    """
    Simple token estimation without model context.

    Uses the standard ~4 characters per token approximation.
    For more accurate estimation, use TokenEstimator.

    Args:
        text: Text to estimate tokens for

    Returns:
        Approximate token count
    """
    if not text:
        return 0
    # Standard approximation: 1 token ~ 4 characters
    return max(1, len(text) // 4)
