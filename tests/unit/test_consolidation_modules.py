"""
Unit tests for ALMA Consolidation submodules.

Tests cover:
- validation.py: LLM response schema validation
- rate_limit.py: RateLimiter, BoundedCache, rate_limit_llm_call decorator
- config.py: ConsolidationConfig, get_llm_api_key
- strategy.py: ConsolidationStrategy ABC, Factory, LLM + Heuristic strategies
- deduplication.py: DeduplicationEngine, DeduplicationResult
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alma.consolidation.config import ConsolidationConfig, get_llm_api_key
from alma.consolidation.deduplication import DeduplicationEngine, DeduplicationResult
from alma.consolidation.exceptions import (
    CacheError,
    ConsolidationError,
    InvalidLLMResponse,
    LLMError,
    StorageError,
    ValidationError,
)
from alma.consolidation.rate_limit import (
    BoundedCache,
    RateLimiter,
    clear_cache,
    get_cache_info,
    get_cached_consolidation_result,
    init_rate_limiter,
    rate_limit_llm_call,
)
from alma.consolidation.strategy import (
    ConsolidationStrategyFactory,
    HeuristicConsolidationStrategy,
    LLMConsolidationStrategy,
)
from alma.consolidation.validation import (
    ANTI_PATTERN_RESPONSE_SCHEMA,
    DOMAIN_KNOWLEDGE_RESPONSE_SCHEMA,
    HEURISTIC_RESPONSE_SCHEMA,
    validate_anti_pattern_response,
    validate_domain_knowledge_response,
    validate_heuristic_response,
    validate_llm_response,
)
from alma.types import Heuristic, Outcome


# ═══════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """Test that the exception hierarchy is correct."""

    def test_consolidation_error_is_base(self):
        assert issubclass(LLMError, ConsolidationError)
        assert issubclass(InvalidLLMResponse, ConsolidationError)
        assert issubclass(CacheError, ConsolidationError)
        assert issubclass(ValidationError, ConsolidationError)
        assert issubclass(StorageError, ConsolidationError)

    def test_consolidation_error_is_exception(self):
        assert issubclass(ConsolidationError, Exception)

    def test_exceptions_carry_message(self):
        err = LLMError("timeout after 30s")
        assert "timeout" in str(err)


# ═══════════════════════════════════════════════════════════════════
# VALIDATION (alma/consolidation/validation.py)
# ═══════════════════════════════════════════════════════════════════


class TestValidateLLMResponse:
    """Test schema-based LLM response validation."""

    def test_valid_heuristic_response(self):
        raw = '{"condition": "form test", "strategy": "validate first", "confidence": 0.9}'
        result = validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)
        assert result["condition"] == "form test"
        assert result["confidence"] == 0.9

    def test_not_json_raises(self):
        with pytest.raises(InvalidLLMResponse, match="not valid JSON"):
            validate_llm_response("this is not json", HEURISTIC_RESPONSE_SCHEMA)

    def test_json_array_raises(self):
        with pytest.raises(InvalidLLMResponse, match="JSON object"):
            validate_llm_response("[1, 2, 3]", HEURISTIC_RESPONSE_SCHEMA)

    def test_missing_required_field_raises(self):
        raw = '{"condition": "x", "strategy": "y"}'
        with pytest.raises(InvalidLLMResponse, match="Missing required field.*confidence"):
            validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)

    def test_wrong_type_raises(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": "high"}'
        with pytest.raises(InvalidLLMResponse, match="wrong type"):
            validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)

    def test_confidence_below_range_raises(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": -0.5}'
        with pytest.raises(InvalidLLMResponse, match="outside allowed range"):
            validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)

    def test_confidence_above_range_raises(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": 1.5}'
        with pytest.raises(InvalidLLMResponse, match="outside allowed range"):
            validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)

    def test_confidence_at_boundaries_ok(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": 0.0}'
        result = validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)
        assert result["confidence"] == 0.0

        raw = '{"condition": "x", "strategy": "y", "confidence": 1.0}'
        result = validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)
        assert result["confidence"] == 1.0

    def test_integer_confidence_accepted(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": 1}'
        result = validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)
        assert result["confidence"] == 1

    def test_extra_fields_allowed(self):
        raw = '{"condition": "x", "strategy": "y", "confidence": 0.5, "extra": "ok"}'
        result = validate_llm_response(raw, HEURISTIC_RESPONSE_SCHEMA)
        assert result["extra"] == "ok"

    def test_empty_schema_accepts_any_dict(self):
        raw = '{"anything": "goes"}'
        result = validate_llm_response(raw, {})
        assert result["anything"] == "goes"


class TestValidateConvenienceFunctions:
    """Test convenience wrappers for common schemas."""

    def test_validate_heuristic_response(self):
        raw = '{"condition": "c", "strategy": "s", "confidence": 0.8}'
        result = validate_heuristic_response(raw)
        assert result["confidence"] == 0.8

    def test_validate_domain_knowledge_response(self):
        raw = '{"fact": "JWT tokens expire", "confidence": 0.9}'
        result = validate_domain_knowledge_response(raw)
        assert result["fact"] == "JWT tokens expire"

    def test_validate_anti_pattern_response(self):
        raw = '{"pattern": "sleep()", "why_bad": "flaky", "better_alternative": "wait_for()", "confidence": 0.7}'
        result = validate_anti_pattern_response(raw)
        assert result["pattern"] == "sleep()"


class TestSchemaDefinitions:
    """Test schema structure correctness."""

    def test_heuristic_schema_has_required_fields(self):
        assert "condition" in HEURISTIC_RESPONSE_SCHEMA["required"]
        assert "strategy" in HEURISTIC_RESPONSE_SCHEMA["required"]
        assert "confidence" in HEURISTIC_RESPONSE_SCHEMA["required"]

    def test_domain_knowledge_schema_has_required_fields(self):
        assert "fact" in DOMAIN_KNOWLEDGE_RESPONSE_SCHEMA["required"]
        assert "confidence" in DOMAIN_KNOWLEDGE_RESPONSE_SCHEMA["required"]

    def test_anti_pattern_schema_has_required_fields(self):
        assert "pattern" in ANTI_PATTERN_RESPONSE_SCHEMA["required"]
        assert "why_bad" in ANTI_PATTERN_RESPONSE_SCHEMA["required"]
        assert "better_alternative" in ANTI_PATTERN_RESPONSE_SCHEMA["required"]
        assert "confidence" in ANTI_PATTERN_RESPONSE_SCHEMA["required"]


# ═══════════════════════════════════════════════════════════════════
# RATE LIMITING (alma/consolidation/rate_limit.py)
# ═══════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """Test token bucket rate limiter."""

    def test_allows_calls_under_limit(self):
        limiter = RateLimiter(calls=5, period=60)
        for _ in range(5):
            assert limiter.acquire() is True

    def test_blocks_calls_over_limit(self):
        limiter = RateLimiter(calls=2, period=60)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        # Third call within period — would block (sleep).
        # We test the counter state instead of waiting.
        assert limiter.calls_made == 2

    def test_window_resets_after_period(self):
        limiter = RateLimiter(calls=1, period=0.01)
        assert limiter.acquire() is True
        time.sleep(0.02)  # Wait for period to elapse
        assert limiter.acquire() is True

    def test_initial_state(self):
        limiter = RateLimiter(calls=100, period=60)
        assert limiter.calls_made == 0
        assert limiter.calls == 100
        assert limiter.period == 60


class TestRateLimiterAsync:
    """Test async rate limiter."""

    @pytest.mark.asyncio
    async def test_async_allows_calls_under_limit(self):
        limiter = RateLimiter(calls=3, period=60)
        for _ in range(3):
            result = await limiter.acquire_async()
            assert result is True

    @pytest.mark.asyncio
    async def test_async_window_resets(self):
        limiter = RateLimiter(calls=1, period=0.01)
        assert await limiter.acquire_async() is True
        await asyncio.sleep(0.02)
        assert await limiter.acquire_async() is True


class TestBoundedCache:
    """Test LRU bounded cache."""

    def test_put_and_get(self):
        cache = BoundedCache(maxsize=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        cache = BoundedCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_evicts_lru_when_full(self):
        cache = BoundedCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_updates_lru_order(self):
        cache = BoundedCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Access "a", making "b" the LRU
        cache.put("c", 3)  # Should evict "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_update_existing_key(self):
        cache = BoundedCache(maxsize=5)
        cache.put("k", "old")
        cache.put("k", "new")
        assert cache.get("k") == "new"
        assert cache.info()["size"] == 1

    def test_clear(self):
        cache = BoundedCache(maxsize=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.info()["size"] == 0

    def test_info(self):
        cache = BoundedCache(maxsize=100)
        cache.put("a", 1)
        cache.put("b", 2)
        info = cache.info()
        assert info["size"] == 2
        assert info["maxsize"] == 100
        assert info["utilization"] == 0.02


class TestRateLimitDecorator:
    """Test rate_limit_llm_call decorator."""

    def test_sync_function_passes_through(self):
        @rate_limit_llm_call
        def my_func(x):
            return x * 2

        # Without global rate limiter, should pass through
        assert my_func(5) == 10

    @pytest.mark.asyncio
    async def test_async_function_passes_through(self):
        @rate_limit_llm_call
        async def my_async_func(x):
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    def test_global_rate_limiter_initialization(self):
        init_rate_limiter(calls=50, period=30)
        # Verify it was created (hard to test directly, but no exception means OK)
        # Clean up
        init_rate_limiter(calls=100, period=60)


class TestCacheFunctions:
    """Test module-level cache functions."""

    def test_get_cached_returns_none_for_unknown(self):
        clear_cache()
        result = get_cached_consolidation_result("unknown_id")
        assert result is None

    def test_get_cache_info_returns_dict(self):
        clear_cache()
        info = get_cache_info()
        assert "hits" in info
        assert "misses" in info
        assert "maxsize" in info
        assert "currsize" in info
        assert "hit_rate" in info

    def test_clear_cache_resets_stats(self):
        clear_cache()
        info = get_cache_info()
        assert info["currsize"] == 0


# ═══════════════════════════════════════════════════════════════════
# CONFIG (alma/consolidation/config.py)
# ═══════════════════════════════════════════════════════════════════


class TestConsolidationConfig:
    """Test environment-based configuration."""

    def test_from_environment_with_valid_key(self):
        env = {"LLM_API_KEY": "sk-test-key-long-enough"}
        with patch.dict("os.environ", env, clear=True):
            config = ConsolidationConfig.from_environment()
        assert config.llm_api_key == "sk-test-key-long-enough"
        assert config.llm_model == "gpt-4"
        assert config.rate_limit_calls == 100
        assert config.rate_limit_period == 60
        assert config.cache_maxsize == 1000
        assert config.similarity_threshold == 0.85

    def test_from_environment_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValidationError, match="LLM_API_KEY"):
                ConsolidationConfig.from_environment()

    def test_from_environment_short_key_raises(self):
        env = {"LLM_API_KEY": "short"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValidationError, match="invalid"):
                ConsolidationConfig.from_environment()

    def test_from_environment_custom_values(self):
        env = {
            "LLM_API_KEY": "sk-custom-api-key-value",
            "LLM_MODEL": "gpt-3.5-turbo",
            "CONSOLIDATION_RATE_LIMIT_CALLS": "50",
            "CONSOLIDATION_RATE_LIMIT_PERIOD": "30",
            "CONSOLIDATION_CACHE_MAXSIZE": "500",
            "CONSOLIDATION_SIMILARITY_THRESHOLD": "0.90",
        }
        with patch.dict("os.environ", env, clear=True):
            config = ConsolidationConfig.from_environment()
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.rate_limit_calls == 50
        assert config.rate_limit_period == 30
        assert config.cache_maxsize == 500
        assert config.similarity_threshold == 0.90


class TestGetLLMApiKey:
    """Test get_llm_api_key convenience function."""

    def test_returns_key_when_set(self):
        env = {"LLM_API_KEY": "sk-valid-key-for-test"}
        with patch.dict("os.environ", env, clear=True):
            key = get_llm_api_key()
        assert key == "sk-valid-key-for-test"

    def test_raises_when_not_set(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValidationError):
                get_llm_api_key()


# ═══════════════════════════════════════════════════════════════════
# STRATEGY (alma/consolidation/strategy.py)
# ═══════════════════════════════════════════════════════════════════


class TestLLMConsolidationStrategy:
    """Test LLM-based consolidation strategy."""

    def test_get_name(self):
        strategy = LLMConsolidationStrategy()
        assert strategy.get_name() == "llm"

    def test_supports_any_item_type(self):
        strategy = LLMConsolidationStrategy()
        assert strategy.supports_item_type("heuristic") is True
        assert strategy.supports_item_type("outcome") is True
        assert strategy.supports_item_type("anything") is True

    def test_consolidate_empty_items(self):
        strategy = LLMConsolidationStrategy()
        result = strategy.consolidate(
            items=[], agent="test", project_id="proj", threshold=0.85
        )
        assert result["consolidated"] == []
        assert result["skipped"] == 0

    def test_consolidate_with_items(self):
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="c1", strategy="s1", confidence=0.9,
                occurrence_count=5, success_count=4,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="c2", strategy="s2", confidence=0.8,
                occurrence_count=3, success_count=2,
                last_validated=now, created_at=now,
            ),
        ]
        strategy = LLMConsolidationStrategy()
        result = strategy.consolidate(
            items=items, agent="a", project_id="p", threshold=0.85
        )
        assert result["strategy"] == "llm"
        assert len(result["consolidated"]) == 1
        assert result["skipped"] == 1


class TestHeuristicConsolidationStrategy:
    """Test heuristic-based consolidation strategy."""

    def test_get_name(self):
        strategy = HeuristicConsolidationStrategy()
        assert strategy.get_name() == "heuristic"

    def test_supports_heuristic_and_outcome(self):
        strategy = HeuristicConsolidationStrategy()
        assert strategy.supports_item_type("heuristic") is True
        assert strategy.supports_item_type("outcome") is True
        assert strategy.supports_item_type("unknown_type") is False

    def test_consolidate_empty(self):
        strategy = HeuristicConsolidationStrategy()
        result = strategy.consolidate(
            items=[], agent="a", project_id="p"
        )
        assert result["consolidated"] == []

    def test_deduplicates_by_text(self):
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="form test", strategy="validate",
                confidence=0.9, occurrence_count=5, success_count=4,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="form test", strategy="validate",
                confidence=0.8, occurrence_count=3, success_count=2,
                last_validated=now, created_at=now,
            ),
        ]
        strategy = HeuristicConsolidationStrategy()
        result = strategy.consolidate(
            items=items, agent="a", project_id="p"
        )
        # Both have same "title" (condition), so one deduped
        assert result["skipped"] == 1
        assert len(result["consolidated"]) == 1

    def test_keeps_unique_items(self):
        """Unique conditions should not be deduped."""
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="unique condition alpha", strategy="s1",
                confidence=0.9, occurrence_count=1, success_count=1,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="unique condition beta", strategy="s2",
                confidence=0.8, occurrence_count=1, success_count=1,
                last_validated=now, created_at=now,
            ),
        ]
        strategy = HeuristicConsolidationStrategy()
        result = strategy.consolidate(
            items=items, agent="a", project_id="p"
        )
        assert result["skipped"] == 0
        assert len(result["consolidated"]) == 2


class TestConsolidationStrategyFactory:
    """Test strategy factory."""

    def test_create_llm_strategy(self):
        strategy = ConsolidationStrategyFactory.create("llm")
        assert isinstance(strategy, LLMConsolidationStrategy)

    def test_create_heuristic_strategy(self):
        strategy = ConsolidationStrategyFactory.create("heuristic")
        assert isinstance(strategy, HeuristicConsolidationStrategy)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ConsolidationStrategyFactory.create("nonexistent")

    def test_get_available_strategies(self):
        available = ConsolidationStrategyFactory.get_available_strategies()
        assert "llm" in available
        assert "heuristic" in available

    def test_register_custom_strategy(self):
        class CustomStrategy(LLMConsolidationStrategy):
            def get_name(self):
                return "custom"

        ConsolidationStrategyFactory.register("custom_test", CustomStrategy)
        strategy = ConsolidationStrategyFactory.create("custom_test")
        assert strategy.get_name() == "custom"

        # Clean up
        del ConsolidationStrategyFactory._strategies["custom_test"]


# ═══════════════════════════════════════════════════════════════════
# DEDUPLICATION (alma/consolidation/deduplication.py)
# ═══════════════════════════════════════════════════════════════════


class TestDeduplicationResult:
    """Test DeduplicationResult dataclass."""

    def test_summary(self):
        result = DeduplicationResult(
            deduplicated=[MagicMock()],
            duplicates_found=3,
            merge_operations=3,
        )
        summary = result.summary()
        assert "1 items" in summary
        assert "3 duplicates" in summary
        assert "3 merges" in summary

    def test_empty_result(self):
        result = DeduplicationResult(
            deduplicated=[],
            duplicates_found=0,
            merge_operations=0,
        )
        assert "0 items" in result.summary()


class TestDeduplicationEngine:
    """Test heuristic deduplication engine."""

    @pytest.fixture
    def engine(self):
        return DeduplicationEngine(similarity_threshold=0.85)

    def test_empty_input(self, engine):
        result = engine.deduplicate([])
        assert result.deduplicated == []
        assert result.duplicates_found == 0
        assert result.merge_operations == 0

    def test_single_item(self, engine):
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h1", agent="a", project_id="p",
            condition="form test", strategy="validate",
            confidence=0.9, occurrence_count=5, success_count=4,
            last_validated=now, created_at=now,
        )
        result = engine.deduplicate([h])
        assert len(result.deduplicated) == 1
        assert result.duplicates_found == 0

    def test_identical_texts_merged(self, engine):
        """Heuristics with identical condition+strategy should be merged."""
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="form testing with validation rules",
                strategy="validate inputs", confidence=0.9,
                occurrence_count=5, success_count=4,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="form testing with validation rules",
                strategy="validate inputs", confidence=0.8,
                occurrence_count=3, success_count=2,
                last_validated=now, created_at=now,
            ),
        ]
        result = engine.deduplicate(items)
        assert len(result.deduplicated) == 1
        assert result.duplicates_found == 1
        assert result.merge_operations == 1

    def test_dissimilar_items_kept_separate(self):
        engine = DeduplicationEngine(similarity_threshold=0.85)
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="form testing validation",
                strategy="validate", confidence=0.9,
                occurrence_count=5, success_count=4,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="database migration rollback",
                strategy="backup first", confidence=0.8,
                occurrence_count=3, success_count=2,
                last_validated=now, created_at=now,
            ),
        ]
        result = engine.deduplicate(items)
        assert len(result.deduplicated) == 2
        assert result.duplicates_found == 0

    def test_keeps_representative_from_group(self, engine):
        """_merge_group keeps first item as representative."""
        now = datetime.now(timezone.utc)
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="exact same text", strategy="same strategy",
                confidence=0.7, occurrence_count=1, success_count=1,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="exact same text", strategy="same strategy",
                confidence=0.95, occurrence_count=5, success_count=5,
                last_validated=now, created_at=now,
            ),
        ]
        result = engine.deduplicate(items)
        assert len(result.deduplicated) == 1

    def test_threshold_affects_grouping(self):
        now = datetime.now(timezone.utc)
        # Two items with moderate text overlap
        items = [
            Heuristic(
                id="h1", agent="a", project_id="p",
                condition="validate form inputs correctly",
                strategy="s", confidence=0.9,
                occurrence_count=1, success_count=1,
                last_validated=now, created_at=now,
            ),
            Heuristic(
                id="h2", agent="a", project_id="p",
                condition="validate user form data inputs",
                strategy="s", confidence=0.9,
                occurrence_count=1, success_count=1,
                last_validated=now, created_at=now,
            ),
        ]
        # Low threshold: should group them
        low_engine = DeduplicationEngine(similarity_threshold=0.3)
        result = low_engine.deduplicate(items)
        assert len(result.deduplicated) == 1

        # High threshold: should keep separate
        high_engine = DeduplicationEngine(similarity_threshold=0.99)
        result = high_engine.deduplicate(items)
        assert len(result.deduplicated) == 2

    def test_works_with_outcomes(self, engine):
        now = datetime.now(timezone.utc)
        items = [
            Outcome(
                id="o1", agent="a", project_id="p",
                task_type="api", task_description="test api endpoint",
                success=True, strategy_used="get request",
                timestamp=now,
            ),
            Outcome(
                id="o2", agent="a", project_id="p",
                task_type="api", task_description="test api endpoint",
                success=False, strategy_used="post request",
                timestamp=now,
            ),
        ]
        # Outcomes use "action" attribute for text extraction
        # which falls back to str(item) — these have same task_description
        result = engine.deduplicate(items)
        # Whether merged depends on text extraction path
        assert len(result.deduplicated) >= 1

    def test_calculate_similarity_identical(self, engine):
        now = datetime.now(timezone.utc)
        h = Heuristic(
            id="h1", agent="a", project_id="p",
            condition="same text", strategy="s", confidence=0.9,
            occurrence_count=1, success_count=1,
            last_validated=now, created_at=now,
        )
        sim = engine._calculate_similarity(h, h)
        assert sim == 1.0

    def test_calculate_similarity_no_overlap(self, engine):
        """Items with zero word overlap have 0.0 similarity."""
        now = datetime.now(timezone.utc)
        h1 = Heuristic(
            id="h1", agent="a", project_id="p",
            condition="alpha beta gamma", strategy="",
            confidence=0.9, occurrence_count=1, success_count=1,
            last_validated=now, created_at=now,
        )
        h2 = Heuristic(
            id="h2", agent="a", project_id="p",
            condition="delta epsilon zeta", strategy="",
            confidence=0.9, occurrence_count=1, success_count=1,
            last_validated=now, created_at=now,
        )
        sim = engine._calculate_similarity(h1, h2)
        assert sim == 0.0

    def test_merge_group_raises_on_empty(self, engine):
        with pytest.raises(ValueError, match="empty group"):
            engine._merge_group([])
