"""
ALMA Consolidation Configuration.

Manages environment-based configuration for the consolidation module,
including LLM API keys, rate limits, and cache settings.

This fixes the security issue where API keys could be hardcoded in config files.
"""

import os
from dataclasses import dataclass
from typing import Optional

from alma.consolidation.exceptions import ValidationError


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation engine."""

    llm_api_key: str
    llm_model: str = "gpt-4"
    rate_limit_calls: int = 100
    rate_limit_period: int = 60  # seconds
    cache_maxsize: int = 1000
    similarity_threshold: float = 0.85

    @classmethod
    def from_environment(cls) -> "ConsolidationConfig":
        """
        Load configuration from environment variables.

        Required env vars:
        - LLM_API_KEY: API key for LLM service

        Optional env vars:
        - LLM_MODEL: LLM model name (default: gpt-4)
        - CONSOLIDATION_RATE_LIMIT_CALLS: Rate limit calls (default: 100)
        - CONSOLIDATION_RATE_LIMIT_PERIOD: Rate limit period in seconds (default: 60)
        - CONSOLIDATION_CACHE_MAXSIZE: LRU cache max size (default: 1000)
        - CONSOLIDATION_SIMILARITY_THRESHOLD: Similarity threshold (default: 0.85)

        Returns:
            ConsolidationConfig instance

        Raises:
            ValidationError: If required env vars missing or invalid
        """
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise ValidationError(
                "LLM_API_KEY environment variable not set. "
                "Please set it before using consolidation features."
            )

        if len(api_key.strip()) < 10:
            raise ValidationError(
                "LLM_API_KEY appears to be invalid (too short). "
                "Please check your environment configuration."
            )

        return cls(
            llm_api_key=api_key,
            llm_model=os.environ.get("LLM_MODEL", "gpt-4"),
            rate_limit_calls=int(
                os.environ.get("CONSOLIDATION_RATE_LIMIT_CALLS", "100")
            ),
            rate_limit_period=int(
                os.environ.get("CONSOLIDATION_RATE_LIMIT_PERIOD", "60")
            ),
            cache_maxsize=int(
                os.environ.get("CONSOLIDATION_CACHE_MAXSIZE", "1000")
            ),
            similarity_threshold=float(
                os.environ.get("CONSOLIDATION_SIMILARITY_THRESHOLD", "0.85")
            ),
        )


def get_llm_api_key() -> str:
    """
    Get LLM API key from environment.

    Returns:
        API key string

    Raises:
        ValidationError: If LLM_API_KEY not set or invalid
    """
    config = ConsolidationConfig.from_environment()
    return config.llm_api_key
