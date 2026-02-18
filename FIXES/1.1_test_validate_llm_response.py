# Test cases for Fix 1.1: LLM Response Validation
# File: tests/unit/consolidation/test_llm_interface.py

import pytest
from alma.consolidation.llm_interface import (
    validate_llm_response,
    InvalidLLMResponse,
    call_llm,
)


class TestValidateLLMResponse:
    """Test LLM response validation."""

    def test_valid_response(self):
        """Valid response passes validation."""
        response = {
            'deduplication_result': {'memory_1': ['memory_2']},
            'confidence': 0.95,
            'consolidated_count': 2,
            'consolidation_strategy': 'semantic',
        }
        validated = validate_llm_response(response)
        assert validated == response

    def test_missing_required_field(self):
        """Missing required field raises error."""
        response = {
            'deduplication_result': {},
            # Missing 'confidence'
            'consolidated_count': 2,
            'consolidation_strategy': 'semantic',
        }
        with pytest.raises(InvalidLLMResponse, match="Missing required field: confidence"):
            validate_llm_response(response)

    def test_wrong_type_confidence(self):
        """Wrong type for confidence raises error."""
        response = {
            'deduplication_result': {},
            'confidence': "0.95",  # String, not float!
            'consolidated_count': 2,
            'consolidation_strategy': 'semantic',
        }
        with pytest.raises(InvalidLLMResponse, match="Field confidence"):
            validate_llm_response(response)

    def test_null_deduplication_result(self):
        """Null deduplication result is allowed."""
        response = {
            'deduplication_result': None,
            'confidence': 0.0,
            'consolidated_count': 0,
            'consolidation_strategy': 'semantic',
        }
        validated = validate_llm_response(response)
        assert validated['deduplication_result'] is None

    def test_not_dict_response(self):
        """Non-dict response raises error."""
        with pytest.raises(InvalidLLMResponse, match="Expected dict"):
            validate_llm_response([1, 2, 3])

    def test_empty_dict(self):
        """Empty dict fails validation."""
        with pytest.raises(InvalidLLMResponse, match="Missing required field"):
            validate_llm_response({})


class TestCallLLMWithValidation:
    """Test call_llm with response validation."""

    def test_valid_call(self, mocker):
        """Valid LLM call with validation."""
        # Mock OpenAI response
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': '''{
                            "deduplication_result": {"m1": ["m2"]},
                            "confidence": 0.95,
                            "consolidated_count": 2,
                            "consolidation_strategy": "semantic"
                        }'''
                    }
                }
            ]
        }
        mocker.patch('openai.ChatCompletion.create', return_value=mock_response)

        result = call_llm("test prompt")
        assert result['confidence'] == 0.95
        assert result['consolidated_count'] == 2

    def test_invalid_json_response(self, mocker):
        """Invalid JSON from LLM raises error."""
        mock_response = {
            'choices': [{'message': {'content': 'not valid json'}}]
        }
        mocker.patch('openai.ChatCompletion.create', return_value=mock_response)

        with pytest.raises(ConsolidationError, match="not valid JSON"):
            call_llm("test prompt")

    def test_malformed_openai_response(self, mocker):
        """Malformed OpenAI response format."""
        mock_response = {'choices': []}  # Missing message
        mocker.patch('openai.ChatCompletion.create', return_value=mock_response)

        with pytest.raises(ConsolidationError, match="Unexpected OpenAI response"):
            call_llm("test prompt")

    def test_openai_api_error(self, mocker):
        """OpenAI API error is caught and wrapped."""
        mocker.patch(
            'openai.ChatCompletion.create',
            side_effect=openai.OpenAIError("API error")
        )

        with pytest.raises(ConsolidationError, match="LLM API call failed"):
            call_llm("test prompt")

    def test_invalid_response_structure_from_llm(self, mocker):
        """LLM returns valid JSON but invalid structure."""
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': '{"wrong_field": "value"}'
                    }
                }
            ]
        }
        mocker.patch('openai.ChatCompletion.create', return_value=mock_response)

        with pytest.raises(InvalidLLMResponse, match="Missing required field"):
            call_llm("test prompt")
