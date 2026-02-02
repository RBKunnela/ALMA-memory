"""
Unit tests for ALMA token estimation module.

Tests the TokenEstimator class and related functionality for accurate
token counting and budget management.
"""

from alma.utils.tokenizer import (
    DEFAULT_TOKEN_BUDGETS,
    ModelFamily,
    ModelTokenBudget,
    TokenEstimator,
    estimate_tokens_simple,
    get_default_token_budget,
    get_token_estimator,
)


class TestModelTokenBudget:
    """Tests for ModelTokenBudget dataclass."""

    def test_effective_memory_budget(self):
        """Test effective budget calculation with safety margin."""
        budget = ModelTokenBudget(
            context_window=8192,
            memory_budget=2000,
            response_reserve=2048,
            safety_margin=0.1,
        )
        # 2000 * (1 - 0.1) = 1800
        assert budget.effective_memory_budget == 1800

    def test_zero_safety_margin(self):
        """Test effective budget with no safety margin."""
        budget = ModelTokenBudget(
            context_window=8192,
            memory_budget=2000,
            response_reserve=2048,
            safety_margin=0.0,
        )
        assert budget.effective_memory_budget == 2000

    def test_high_safety_margin(self):
        """Test effective budget with high safety margin."""
        budget = ModelTokenBudget(
            context_window=8192,
            memory_budget=2000,
            response_reserve=2048,
            safety_margin=0.5,
        )
        assert budget.effective_memory_budget == 1000


class TestTokenEstimator:
    """Tests for TokenEstimator class."""

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        estimator = TokenEstimator()
        assert estimator.model == "gpt-4"
        assert estimator.model_family == ModelFamily.GPT4

    def test_init_with_specific_model(self):
        """Test initialization with specific model."""
        estimator = TokenEstimator(model="claude-3-sonnet")
        assert estimator.model == "claude-3-sonnet"
        assert estimator.model_family == ModelFamily.CLAUDE

    def test_detect_model_family_gpt4(self):
        """Test GPT-4 family detection."""
        estimator = TokenEstimator(model="gpt-4-turbo")
        assert estimator.model_family == ModelFamily.GPT4

        estimator = TokenEstimator(model="gpt4o")
        assert estimator.model_family == ModelFamily.GPT4

    def test_detect_model_family_gpt35(self):
        """Test GPT-3.5 family detection."""
        estimator = TokenEstimator(model="gpt-3.5-turbo")
        assert estimator.model_family == ModelFamily.GPT35

    def test_detect_model_family_claude(self):
        """Test Claude family detection."""
        estimator = TokenEstimator(model="claude-3-opus")
        assert estimator.model_family == ModelFamily.CLAUDE

        estimator = TokenEstimator(model="claude-3.5-sonnet")
        assert estimator.model_family == ModelFamily.CLAUDE

    def test_detect_model_family_gemini(self):
        """Test Gemini family detection."""
        estimator = TokenEstimator(model="gemini-pro")
        assert estimator.model_family == ModelFamily.GEMINI

        estimator = TokenEstimator(model="gemini-1.5-flash")
        assert estimator.model_family == ModelFamily.GEMINI

    def test_detect_model_family_llama(self):
        """Test Llama family detection."""
        estimator = TokenEstimator(model="llama-3-70b")
        assert estimator.model_family == ModelFamily.LLAMA

    def test_detect_model_family_mistral(self):
        """Test Mistral family detection."""
        estimator = TokenEstimator(model="mistral-7b")
        assert estimator.model_family == ModelFamily.MISTRAL

        estimator = TokenEstimator(model="mixtral-8x7b")
        assert estimator.model_family == ModelFamily.MISTRAL

    def test_detect_model_family_unknown(self):
        """Test unknown model family."""
        estimator = TokenEstimator(model="some-unknown-model")
        assert estimator.model_family == ModelFamily.UNKNOWN

    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string."""
        estimator = TokenEstimator()
        assert estimator.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Test counting tokens in simple text."""
        estimator = TokenEstimator(model="gpt-4")
        # "Hello, world!" is typically 4 tokens
        count = estimator.count_tokens("Hello, world!")
        # Allow some variance depending on tiktoken availability
        assert 1 <= count <= 10

    def test_count_tokens_longer_text(self):
        """Test counting tokens in longer text."""
        estimator = TokenEstimator(model="gpt-4")
        text = "This is a longer piece of text that should have more tokens. " * 10
        count = estimator.count_tokens(text)
        # Should be significantly more than a short text
        assert count > 50

    def test_count_tokens_consistency(self):
        """Test that token counting is consistent for same text."""
        estimator = TokenEstimator(model="gpt-4")
        text = "The quick brown fox jumps over the lazy dog."
        count1 = estimator.count_tokens(text)
        count2 = estimator.count_tokens(text)
        assert count1 == count2

    def test_count_tokens_for_messages(self):
        """Test counting tokens for chat messages."""
        estimator = TokenEstimator(model="gpt-4")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        count = estimator.count_tokens_for_messages(messages)
        # Should include message overhead
        assert count > 0

    def test_get_token_budget_known_model(self):
        """Test getting budget for known model."""
        estimator = TokenEstimator(model="gpt-4")
        budget = estimator.get_token_budget()
        assert budget.context_window == 8192
        assert budget.memory_budget == 2000

    def test_get_token_budget_custom(self):
        """Test getting custom budget."""
        custom = ModelTokenBudget(
            context_window=16000,
            memory_budget=5000,
            response_reserve=3000,
        )
        estimator = TokenEstimator(model="gpt-4", custom_budget=custom)
        budget = estimator.get_token_budget()
        assert budget.context_window == 16000
        assert budget.memory_budget == 5000

    def test_get_token_budget_unknown_model(self):
        """Test getting budget for unknown model uses default."""
        estimator = TokenEstimator(model="unknown-model-xyz")
        budget = estimator.get_token_budget()
        # Should fall back to default
        assert budget.context_window == DEFAULT_TOKEN_BUDGETS["default"].context_window

    def test_truncate_to_token_limit_no_truncation_needed(self):
        """Test truncation when text is within limit."""
        estimator = TokenEstimator(model="gpt-4")
        text = "Short text"
        result = estimator.truncate_to_token_limit(text, max_tokens=1000)
        assert result == text
        assert "[truncated]" not in result

    def test_truncate_to_token_limit_truncation_needed(self):
        """Test truncation when text exceeds limit."""
        estimator = TokenEstimator(model="gpt-4")
        text = "This is a test sentence. " * 100
        result = estimator.truncate_to_token_limit(text, max_tokens=50)
        assert len(result) < len(text)
        assert result.endswith("[truncated]")

    def test_truncate_to_token_limit_custom_suffix(self):
        """Test truncation with custom suffix."""
        estimator = TokenEstimator(model="gpt-4")
        text = "This is a test sentence. " * 100
        result = estimator.truncate_to_token_limit(
            text, max_tokens=50, suffix="... [more]"
        )
        assert result.endswith("... [more]")

    def test_estimate_remaining_budget(self):
        """Test remaining budget estimation."""
        estimator = TokenEstimator(model="gpt-4")
        budget = estimator.get_token_budget()

        # With no usage
        remaining = estimator.estimate_remaining_budget(used_tokens=0)
        assert remaining > 0
        assert remaining <= budget.effective_memory_budget

        # With some usage
        remaining = estimator.estimate_remaining_budget(used_tokens=1000)
        remaining_no_usage = estimator.estimate_remaining_budget(used_tokens=0)
        assert remaining <= remaining_no_usage

    def test_estimate_remaining_budget_no_response_reserve(self):
        """Test remaining budget without response reserve."""
        estimator = TokenEstimator(model="gpt-4")

        with_reserve = estimator.estimate_remaining_budget(
            used_tokens=0, include_response_reserve=True
        )
        without_reserve = estimator.estimate_remaining_budget(
            used_tokens=0, include_response_reserve=False
        )

        # Without reserve should have more available
        assert without_reserve >= with_reserve


class TestFactoryFunctions:
    """Tests for module-level factory functions."""

    def test_get_token_estimator(self):
        """Test get_token_estimator factory function."""
        estimator = get_token_estimator(model="claude-3-sonnet")
        assert isinstance(estimator, TokenEstimator)
        assert estimator.model == "claude-3-sonnet"

    def test_get_token_estimator_with_custom_budget(self):
        """Test get_token_estimator with custom budget."""
        custom = ModelTokenBudget(
            context_window=50000,
            memory_budget=10000,
        )
        estimator = get_token_estimator(model="gpt-4", custom_budget=custom)
        assert estimator.get_token_budget().memory_budget == 10000

    def test_get_default_token_budget_exact_match(self):
        """Test getting default budget with exact model name."""
        budget = get_default_token_budget("gpt-4")
        assert budget.context_window == 8192

    def test_get_default_token_budget_partial_match(self):
        """Test getting default budget with partial model name."""
        # Using a model that should match gpt-4o (128000 context)
        budget = get_default_token_budget("gpt-4o-2024-05-13")
        # Should match gpt-4o which has 128000 context
        assert budget.context_window == 128000

    def test_get_default_token_budget_unknown(self):
        """Test getting default budget for unknown model."""
        budget = get_default_token_budget("totally-unknown-model")
        assert budget == DEFAULT_TOKEN_BUDGETS["default"]


class TestEstimateTokensSimple:
    """Tests for simple token estimation function."""

    def test_empty_string(self):
        """Test simple estimation with empty string."""
        assert estimate_tokens_simple("") == 0

    def test_short_text(self):
        """Test simple estimation with short text."""
        # 12 characters -> ~3 tokens
        result = estimate_tokens_simple("Hello world!")
        assert result >= 1

    def test_approximation_ratio(self):
        """Test that approximation follows ~4 chars per token."""
        text = "a" * 100
        result = estimate_tokens_simple(text)
        # 100 / 4 = 25
        assert result == 25

    def test_minimum_one_token(self):
        """Test that at least 1 token is returned for non-empty text."""
        result = estimate_tokens_simple("Hi")
        assert result >= 1


class TestDefaultTokenBudgets:
    """Tests for default token budget configurations."""

    def test_all_models_have_required_fields(self):
        """Test that all default budgets have required fields."""
        for model, budget in DEFAULT_TOKEN_BUDGETS.items():
            assert budget.context_window > 0, f"{model} missing context_window"
            assert budget.memory_budget > 0, f"{model} missing memory_budget"
            assert budget.response_reserve > 0, f"{model} missing response_reserve"

    def test_memory_budget_less_than_context(self):
        """Test that memory budget is less than context window."""
        for model, budget in DEFAULT_TOKEN_BUDGETS.items():
            assert budget.memory_budget < budget.context_window, (
                f"{model}: memory_budget should be less than context_window"
            )

    def test_claude_models_have_large_context(self):
        """Test that Claude models have large context windows."""
        claude_models = [k for k in DEFAULT_TOKEN_BUDGETS if "claude" in k]
        for model in claude_models:
            budget = DEFAULT_TOKEN_BUDGETS[model]
            assert budget.context_window >= 200000

    def test_gpt4_turbo_has_large_context(self):
        """Test that GPT-4 Turbo has large context window."""
        budget = DEFAULT_TOKEN_BUDGETS["gpt-4-turbo"]
        assert budget.context_window == 128000
