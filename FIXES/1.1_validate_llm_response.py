# Fix 1.1: Validate LLM Response Structure
# File: alma/consolidation/llm_interface.py
# Impact: Prevents KeyError crashes, potential injection

from typing import Dict, Any
import openai


class ConsolidationError(Exception):
    """Base exception for consolidation errors."""
    pass


class InvalidLLMResponse(ConsolidationError):
    """Raised when LLM response has invalid structure."""
    pass


# Response schema validation
LLM_RESPONSE_SCHEMA = {
    'deduplication_result': (dict, type(None)),
    'confidence': (float, int),
    'consolidated_count': int,
    'consolidation_strategy': str,
}


def validate_llm_response(response: Any) -> Dict[str, Any]:
    """
    Validate LLM response structure before accessing fields.

    Args:
        response: Raw response from LLM API

    Returns:
        Validated response dict

    Raises:
        InvalidLLMResponse: If response structure is invalid
    """
    if not isinstance(response, dict):
        raise InvalidLLMResponse(f"Expected dict, got {type(response)}")

    for key, expected_types in LLM_RESPONSE_SCHEMA.items():
        if key not in response:
            raise InvalidLLMResponse(f"Missing required field: {key}")

        value = response[key]
        if not isinstance(value, expected_types):
            raise InvalidLLMResponse(
                f"Field {key}: expected {expected_types}, got {type(value)}"
            )

    return response


def call_llm(prompt: str) -> dict:
    """
    Call LLM with consolidation prompt and validate response.

    Args:
        prompt: Consolidation prompt for LLM

    Returns:
        Validated consolidation results

    Raises:
        ConsolidationError: If LLM call fails or response invalid
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    except openai.OpenAIError as e:
        raise ConsolidationError(f"LLM API call failed: {e}") from e

    # Extract content from OpenAI response format
    try:
        content = response['choices'][0]['message']['content']
    except (KeyError, IndexError) as e:
        raise ConsolidationError(f"Unexpected OpenAI response format: {e}") from e

    # Parse JSON from content
    import json
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ConsolidationError(f"LLM response is not valid JSON: {e}") from e

    # VALIDATE before using
    validated = validate_llm_response(parsed)

    return validated
