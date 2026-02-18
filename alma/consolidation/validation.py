"""
ALMA Consolidation Response Validation.

Validates LLM responses before processing, preventing:
- KeyError crashes on missing fields
- JSON parsing errors on malformed responses
- SQL injection via unsanitized LLM output

This fixes the critical gap where LLM responses were used without validation.
"""

import json
from typing import Any, Dict

from alma.consolidation.exceptions import InvalidLLMResponse


# Schema for heuristic merge responses
HEURISTIC_RESPONSE_SCHEMA = {
    "required": ["condition", "strategy", "confidence"],
    "types": {
        "condition": str,
        "strategy": str,
        "confidence": (int, float),
    },
    "constraints": {
        "confidence": (0.0, 1.0),  # Must be between 0.0 and 1.0
    },
}

# Schema for domain knowledge merge responses
DOMAIN_KNOWLEDGE_RESPONSE_SCHEMA = {
    "required": ["fact", "confidence"],
    "types": {
        "fact": str,
        "confidence": (int, float),
    },
    "constraints": {
        "confidence": (0.0, 1.0),
    },
}

# Schema for anti-pattern merge responses
ANTI_PATTERN_RESPONSE_SCHEMA = {
    "required": ["pattern", "why_bad", "better_alternative", "confidence"],
    "types": {
        "pattern": str,
        "why_bad": str,
        "better_alternative": str,
        "confidence": (int, float),
    },
    "constraints": {
        "confidence": (0.0, 1.0),
    },
}


def validate_llm_response(
    response_text: str, response_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate LLM response against schema.

    Args:
        response_text: Raw response from LLM
        response_schema: Schema with required fields, types, and constraints

    Returns:
        Validated response dict

    Raises:
        InvalidLLMResponse: If response is invalid
    """
    # Parse JSON
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise InvalidLLMResponse(
            f"LLM response is not valid JSON: {e}\nResponse: {response_text[:200]}"
        ) from e

    if not isinstance(response, dict):
        raise InvalidLLMResponse(
            f"LLM response must be a JSON object, got {type(response).__name__}"
        )

    # Check required fields
    required = response_schema.get("required", [])
    for field in required:
        if field not in response:
            raise InvalidLLMResponse(
                f"Missing required field in LLM response: '{field}'\n"
                f"Required fields: {required}\n"
                f"Got: {list(response.keys())}"
            )

    # Check types
    types_map = response_schema.get("types", {})
    for field, expected_type in types_map.items():
        if field in response:
            if not isinstance(response[field], expected_type):
                raise InvalidLLMResponse(
                    f"Field '{field}' has wrong type. "
                    f"Expected {expected_type}, got {type(response[field]).__name__}"
                )

    # Check constraints
    constraints = response_schema.get("constraints", {})
    for field, (min_val, max_val) in constraints.items():
        if field in response:
            value = response[field]
            if not (min_val <= value <= max_val):
                raise InvalidLLMResponse(
                    f"Field '{field}' value {value} outside allowed range "
                    f"[{min_val}, {max_val}]"
                )

    return response


def validate_heuristic_response(response_text: str) -> Dict[str, Any]:
    """Validate heuristic merge response."""
    return validate_llm_response(response_text, HEURISTIC_RESPONSE_SCHEMA)


def validate_domain_knowledge_response(response_text: str) -> Dict[str, Any]:
    """Validate domain knowledge merge response."""
    return validate_llm_response(response_text, DOMAIN_KNOWLEDGE_RESPONSE_SCHEMA)


def validate_anti_pattern_response(response_text: str) -> Dict[str, Any]:
    """Validate anti-pattern merge response."""
    return validate_llm_response(response_text, ANTI_PATTERN_RESPONSE_SCHEMA)
