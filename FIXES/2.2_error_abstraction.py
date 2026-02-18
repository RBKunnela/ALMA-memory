# Fix 2.2: Error Abstraction Layer (1 hour)
# File: alma/consolidation/exceptions.py
# Impact: Fixes boundary violation (LLM errors no longer leak)

class ConsolidationError(Exception):
    """Base exception for all consolidation errors."""
    pass


class LLMError(ConsolidationError):
    """Raised when LLM API call fails."""
    pass


class InvalidLLMResponse(ConsolidationError):
    """Raised when LLM response has invalid structure."""
    pass


class CacheError(ConsolidationError):
    """Raised when cache operation fails."""
    pass


class ValidationError(ConsolidationError):
    """Raised when validation fails."""
    pass


# ═════════════════════════════════════════════════════════════
# USAGE IN llm_interface.py:
# ═════════════════════════════════════════════════════════════

from alma.consolidation.exceptions import (
    ConsolidationError,
    LLMError,
    InvalidLLMResponse,
)
import openai


def call_llm(prompt: str) -> dict:
    """
    Call LLM with error abstraction.

    Wraps all OpenAI-specific errors in ConsolidationError.
    Callers don't need to know about OpenAI implementation.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    except openai.OpenAIError as e:
        # BEFORE: raise e (caller knows about OpenAI)
        # AFTER: wrap in module exception
        raise LLMError(
            f"LLM API call failed: {e.__class__.__name__}: {e}"
        ) from e
    except Exception as e:
        raise ConsolidationError(
            f"Unexpected error during LLM call: {e}"
        ) from e

    # ... rest of validation and response handling
    return validate_and_parse_response(response)


# Result: Callers only see ConsolidationError family
# No coupling to OpenAI implementation details!
