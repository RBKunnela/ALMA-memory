"""
LLM Adapters (SCAFFOLD ONLY -- v1.1)

Stubs for end-to-end LoCoMo mode. The retrieval-only benchmark does not need
any of these; they exist so v1.1 can plug in without reshaping the runner CLI.

Every adapter subclasses :class:`LLMAdapter` and wires up:

- ``generate(prompt)`` -- produce an answer from retrieved context
- ``judge(question, answer, reference)`` -- LLM-as-judge scoring

Transport: ``httpx`` (sync) -- kept minimal so no provider SDK becomes a
benchmark dependency. Every stub's ``generate()`` / ``judge()`` currently
raises :class:`NotImplementedError`; ``from_env`` reads API keys so the CLI
can fail fast with a clear message instead of half-loading.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Type

PROVIDER_ENV: Dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "ollama_cloud": "OLLAMA_CLOUD_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

NOT_IMPLEMENTED_MSG = (
    "End-to-end LoCoMo mode is not implemented in v1.0. "
    "LLM generation and judge scoring are coming in v1.1. "
    "Use --mode retrieval for the current build."
)


@dataclass
class LLMAdapter(ABC):
    """
    Abstract base for LLM provider adapters used in end-to-end mode.

    Subclasses are expected to implement :meth:`generate` (for answer
    production from retrieved context) and :meth:`judge` (for LLM-as-judge
    scoring against reference answers). Both are stubs in v1.0.
    """

    api_key: str
    model: str
    base_url: str = ""
    timeout_s: float = 60.0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate an answer given a fully assembled prompt.

        Returns:
            Model-generated answer text.

        Raises:
            NotImplementedError: Until v1.1 lands the HTTP implementation.
        """

    @abstractmethod
    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        """
        Score an answer against a reference answer (LLM-as-judge).

        Returns:
            Score in ``[0.0, 1.0]``.

        Raises:
            NotImplementedError: Until v1.1 lands the judge prompt + parser.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, provider: str, model: str = "") -> "LLMAdapter":
        """
        Construct an adapter from environment variables.

        Looks up the provider's expected env var (see :data:`PROVIDER_ENV`)
        and dispatches to the matching concrete subclass. Raises
        :class:`RuntimeError` if the API key env var is missing, and
        :class:`ValueError` for unknown providers.

        Args:
            provider: One of ``openrouter``, ``zai``, ``ollama_cloud``,
                      ``openai``, ``anthropic``.
            model: Model identifier specific to the provider.
        """
        provider = provider.lower().strip()
        if provider not in PROVIDER_ENV:
            raise ValueError(
                f"Unknown LLM provider: {provider!r}. "
                f"Expected one of {sorted(PROVIDER_ENV)}."
            )

        env_var = PROVIDER_ENV[provider]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise RuntimeError(
                f"Missing env var {env_var} for provider {provider!r}. "
                f"Set it before running end-to-end mode."
            )

        adapter_cls: Type[LLMAdapter] = _PROVIDER_CLASSES[provider]
        return adapter_cls(api_key=api_key, model=model or adapter_cls._default_model)


# ---------------------------------------------------------------------------
# Concrete stubs -- all raise NotImplementedError on actual call paths
# ---------------------------------------------------------------------------


@dataclass
class OpenRouterAdapter(LLMAdapter):
    """OpenRouter adapter stub -- endpoints + payload shape TBD in v1.1."""

    base_url: str = "https://openrouter.ai/api/v1"
    _default_model: str = "anthropic/claude-sonnet-4"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


@dataclass
class ZaiAdapter(LLMAdapter):
    """Z.AI adapter stub."""

    base_url: str = "https://api.z.ai/api/paas/v4"
    _default_model: str = "glm-4.6"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


@dataclass
class OllamaCloudAdapter(LLMAdapter):
    """Ollama Cloud adapter stub."""

    base_url: str = "https://ollama.com/api"
    _default_model: str = "llama3.1:70b"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


@dataclass
class OpenAIAdapter(LLMAdapter):
    """OpenAI adapter stub."""

    base_url: str = "https://api.openai.com/v1"
    _default_model: str = "gpt-4o-mini"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


@dataclass
class AnthropicAdapter(LLMAdapter):
    """Anthropic adapter stub."""

    base_url: str = "https://api.anthropic.com/v1"
    _default_model: str = "claude-sonnet-4-5"

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def judge(self, question: str, answer: str, reference: str, **kwargs) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)


_PROVIDER_CLASSES: Dict[str, Type[LLMAdapter]] = {
    "openrouter": OpenRouterAdapter,
    "zai": ZaiAdapter,
    "ollama_cloud": OllamaCloudAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


def resolve_adapter(
    provider: Optional[str], model: Optional[str] = None
) -> Optional[LLMAdapter]:
    """Convenience used by the runner when ``--mode end-to-end`` is requested."""
    if not provider:
        return None
    return LLMAdapter.from_env(provider, model or "")
