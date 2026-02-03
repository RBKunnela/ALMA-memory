"""
Unit tests for ALMA Retrieval Modes.

Tests mode definitions, inference, and validation.
"""

import pytest

from alma.retrieval.modes import (
    MODE_CONFIGS,
    ModeConfig,
    RetrievalMode,
    create_custom_mode,
    get_mode_config,
    get_mode_reason,
    infer_mode_from_query,
    validate_mode_config,
)


class TestRetrievalMode:
    """Tests for RetrievalMode enum."""

    def test_all_modes_exist(self):
        """All expected modes should be defined."""
        expected = {"broad", "precise", "diagnostic", "learning", "recall"}
        actual = {mode.value for mode in RetrievalMode}
        assert actual == expected

    def test_mode_values_are_strings(self):
        """Mode values should be lowercase strings."""
        for mode in RetrievalMode:
            assert isinstance(mode.value, str)
            assert mode.value == mode.value.lower()


class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_default_config_creation(self):
        """Should create a valid config with defaults."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
            weights={"similarity": 0.4, "recency": 0.3, "success_rate": 0.2, "confidence": 0.1},
            include_anti_patterns=True,
            diversity_factor=0.5,
        )
        assert config.top_k == 5
        assert config.min_confidence == 0.5
        assert config.diversity_factor == 0.5

    def test_weight_normalization(self):
        """Weights should be normalized to sum to 1.0."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
            weights={"similarity": 0.8, "recency": 0.8, "success_rate": 0.8, "confidence": 0.8},
            include_anti_patterns=True,
            diversity_factor=0.5,
        )
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_optional_fields_defaults(self):
        """Optional fields should have sensible defaults."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
        )
        assert config.prioritize_failures is False
        assert config.cluster_similar is False
        assert config.exact_match_boost == 1.0


class TestModeConfigs:
    """Tests for MODE_CONFIGS dictionary."""

    def test_all_modes_have_configs(self):
        """Every mode should have a corresponding config."""
        for mode in RetrievalMode:
            assert mode in MODE_CONFIGS

    def test_configs_have_valid_weights(self):
        """All configs should have weights summing to 1.0."""
        for mode, config in MODE_CONFIGS.items():
            if config.weights:
                total = sum(config.weights.values())
                assert abs(total - 1.0) < 0.01, f"{mode.value} weights don't sum to 1.0"

    def test_configs_have_required_weight_keys(self):
        """All configs should have all required weight keys."""
        required = {"similarity", "recency", "success_rate", "confidence"}
        for mode, config in MODE_CONFIGS.items():
            if config.weights:
                assert required <= set(config.weights.keys()), f"{mode.value} missing weight keys"

    def test_broad_mode_config(self):
        """BROAD mode should have high diversity and low confidence threshold."""
        config = MODE_CONFIGS[RetrievalMode.BROAD]
        assert config.top_k >= 10
        assert config.min_confidence < 0.5
        assert config.diversity_factor >= 0.5
        assert config.include_anti_patterns is False

    def test_precise_mode_config(self):
        """PRECISE mode should have high confidence and low diversity."""
        config = MODE_CONFIGS[RetrievalMode.PRECISE]
        assert config.min_confidence >= 0.6
        assert config.diversity_factor < 0.5
        assert config.exact_match_boost > 1.0
        assert config.include_anti_patterns is True

    def test_diagnostic_mode_config(self):
        """DIAGNOSTIC mode should prioritize failures and include anti-patterns."""
        config = MODE_CONFIGS[RetrievalMode.DIAGNOSTIC]
        assert config.prioritize_failures is True
        assert config.include_anti_patterns is True

    def test_learning_mode_config(self):
        """LEARNING mode should have high top_k and low threshold."""
        config = MODE_CONFIGS[RetrievalMode.LEARNING]
        assert config.top_k >= 15
        assert config.min_confidence < 0.3
        assert config.cluster_similar is True

    def test_recall_mode_config(self):
        """RECALL mode should have low top_k and high exact match boost."""
        config = MODE_CONFIGS[RetrievalMode.RECALL]
        assert config.top_k <= 5
        assert config.exact_match_boost >= 2.0
        assert config.diversity_factor == 0.0


class TestModeInference:
    """Tests for infer_mode_from_query function."""

    @pytest.mark.parametrize("query,expected", [
        # DIAGNOSTIC queries
        ("Why is the login failing?", RetrievalMode.DIAGNOSTIC),
        ("Debug the authentication error", RetrievalMode.DIAGNOSTIC),
        ("Fix the broken API endpoint", RetrievalMode.DIAGNOSTIC),
        ("The system crashed, what went wrong?", RetrievalMode.DIAGNOSTIC),
        ("There's a bug in the payment flow", RetrievalMode.DIAGNOSTIC),
        ("This feature is not working", RetrievalMode.DIAGNOSTIC),
        ("Error handling in the checkout", RetrievalMode.DIAGNOSTIC),
        # BROAD queries
        ("What are our options for authentication?", RetrievalMode.BROAD),
        ("How should we design the API?", RetrievalMode.BROAD),
        ("Ways to implement caching", RetrievalMode.BROAD),
        ("Let's brainstorm solutions", RetrievalMode.BROAD),
        ("Explore different architectures", RetrievalMode.BROAD),
        ("Plan the database migration", RetrievalMode.BROAD),
        # RECALL queries
        ("What did we decide last week?", RetrievalMode.RECALL),
        ("Remember when we updated the config?", RetrievalMode.RECALL),
        ("What was the previous approach?", RetrievalMode.RECALL),
        ("When did we last update the API?", RetrievalMode.RECALL),
        ("Previously we tried...", RetrievalMode.RECALL),
        # LEARNING queries
        ("Find similar patterns in our tests", RetrievalMode.LEARNING),
        ("What common themes have we seen?", RetrievalMode.LEARNING),
        ("Consolidate recurring themes", RetrievalMode.LEARNING),
        ("Look for recurring patterns", RetrievalMode.LEARNING),
        # PRECISE queries (default)
        ("Implement the login form", RetrievalMode.PRECISE),
        ("Add validation to the email field", RetrievalMode.PRECISE),
        ("Create a new API endpoint", RetrievalMode.PRECISE),
        ("Write the unit tests", RetrievalMode.PRECISE),
    ])
    def test_mode_inference(self, query, expected):
        """Test mode inference for various queries."""
        result = infer_mode_from_query(query)
        assert result == expected, f"Expected {expected.value} for '{query}', got {result.value}"

    def test_empty_query_returns_precise(self):
        """Empty query should default to PRECISE."""
        assert infer_mode_from_query("") == RetrievalMode.PRECISE

    def test_case_insensitive(self):
        """Inference should be case-insensitive."""
        assert infer_mode_from_query("ERROR in the system") == RetrievalMode.DIAGNOSTIC
        assert infer_mode_from_query("error in the system") == RetrievalMode.DIAGNOSTIC

    def test_multiple_keywords_first_wins(self):
        """When multiple keywords match, diagnostic terms take priority."""
        # Contains both "error" (diagnostic) and "plan" (broad)
        result = infer_mode_from_query("Plan to fix the error")
        # Diagnostic should win since it's checked first
        assert result == RetrievalMode.DIAGNOSTIC


class TestGetModeReason:
    """Tests for get_mode_reason function."""

    def test_diagnostic_reason_includes_terms(self):
        """Diagnostic reason should mention matched terms."""
        reason = get_mode_reason("Fix the bug", RetrievalMode.DIAGNOSTIC)
        assert "bug" in reason.lower() or "diagnostic" in reason.lower()

    def test_broad_reason_includes_terms(self):
        """Broad reason should mention matched terms."""
        reason = get_mode_reason("How should we design this?", RetrievalMode.BROAD)
        assert "plan" in reason.lower() or "explor" in reason.lower() or "how should" in reason.lower()

    def test_precise_default_reason(self):
        """Precise mode should have a default reason."""
        reason = get_mode_reason("Implement the feature", RetrievalMode.PRECISE)
        assert len(reason) > 0
        assert "default" in reason.lower() or "implementation" in reason.lower() or "execution" in reason.lower()


class TestGetModeConfig:
    """Tests for get_mode_config function."""

    def test_returns_correct_config(self):
        """Should return the correct config for each mode."""
        for mode in RetrievalMode:
            config = get_mode_config(mode)
            assert config == MODE_CONFIGS[mode]

    def test_returns_modeconfig_instance(self):
        """Should return a ModeConfig instance."""
        config = get_mode_config(RetrievalMode.BROAD)
        assert isinstance(config, ModeConfig)


class TestCreateCustomMode:
    """Tests for create_custom_mode function."""

    def test_override_top_k(self):
        """Should override top_k."""
        custom = create_custom_mode(RetrievalMode.PRECISE, top_k=20)
        original = get_mode_config(RetrievalMode.PRECISE)
        assert custom.top_k == 20
        assert custom.min_confidence == original.min_confidence

    def test_override_min_confidence(self):
        """Should override min_confidence."""
        custom = create_custom_mode(RetrievalMode.BROAD, min_confidence=0.8)
        assert custom.min_confidence == 0.8

    def test_override_multiple_fields(self):
        """Should override multiple fields."""
        custom = create_custom_mode(
            RetrievalMode.DIAGNOSTIC,
            top_k=15,
            diversity_factor=0.9,
            prioritize_failures=False,
        )
        assert custom.top_k == 15
        assert custom.diversity_factor == 0.9
        assert custom.prioritize_failures is False

    def test_preserves_unoverridden_fields(self):
        """Should preserve fields that aren't overridden."""
        original = get_mode_config(RetrievalMode.PRECISE)
        custom = create_custom_mode(RetrievalMode.PRECISE, top_k=20)
        assert custom.include_anti_patterns == original.include_anti_patterns
        assert custom.exact_match_boost == original.exact_match_boost


class TestValidateModeConfig:
    """Tests for validate_mode_config function."""

    def test_valid_config_returns_empty_list(self):
        """Valid config should return no errors."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
            weights={"similarity": 0.4, "recency": 0.3, "success_rate": 0.2, "confidence": 0.1},
            include_anti_patterns=True,
            diversity_factor=0.5,
        )
        errors = validate_mode_config(config)
        assert errors == []

    def test_invalid_top_k(self):
        """top_k < 1 should be invalid."""
        config = ModeConfig(top_k=0, min_confidence=0.5)
        errors = validate_mode_config(config)
        assert any("top_k" in e for e in errors)

    def test_invalid_min_confidence(self):
        """min_confidence outside [0, 1] should be invalid."""
        config = ModeConfig(top_k=5, min_confidence=1.5)
        errors = validate_mode_config(config)
        assert any("min_confidence" in e for e in errors)

        config = ModeConfig(top_k=5, min_confidence=-0.1)
        errors = validate_mode_config(config)
        assert any("min_confidence" in e for e in errors)

    def test_invalid_diversity_factor(self):
        """diversity_factor outside [0, 1] should be invalid."""
        config = ModeConfig(top_k=5, min_confidence=0.5, diversity_factor=1.5)
        errors = validate_mode_config(config)
        assert any("diversity_factor" in e for e in errors)

    def test_negative_exact_match_boost(self):
        """exact_match_boost < 0 should be invalid."""
        config = ModeConfig(top_k=5, min_confidence=0.5, exact_match_boost=-1.0)
        errors = validate_mode_config(config)
        assert any("exact_match_boost" in e for e in errors)

    def test_weights_not_summing_to_one(self):
        """Weights not summing to 1.0 should be invalid."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
            weights={"similarity": 0.5, "recency": 0.5, "success_rate": 0.5, "confidence": 0.5},
        )
        # Note: __post_init__ normalizes, so we need to bypass
        config.weights = {"similarity": 0.5, "recency": 0.5, "success_rate": 0.5, "confidence": 0.5}
        errors = validate_mode_config(config)
        assert any("weights" in e and "sum" in e for e in errors)

    def test_missing_weight_keys(self):
        """Missing weight keys should be invalid."""
        config = ModeConfig(
            top_k=5,
            min_confidence=0.5,
            weights={"similarity": 1.0},  # Missing other keys
        )
        errors = validate_mode_config(config)
        assert any("missing" in e.lower() for e in errors)

    def test_all_default_configs_are_valid(self):
        """All default mode configs should pass validation."""
        for mode, config in MODE_CONFIGS.items():
            errors = validate_mode_config(config)
            assert errors == [], f"{mode.value} config has errors: {errors}"


class TestModeConfigWeightingBehavior:
    """Tests for how different modes weight factors."""

    def test_broad_prioritizes_similarity(self):
        """BROAD mode should give high weight to similarity for exploration."""
        config = get_mode_config(RetrievalMode.BROAD)
        assert config.weights["similarity"] >= 0.5

    def test_precise_prioritizes_success(self):
        """PRECISE mode should give high weight to success rate."""
        config = get_mode_config(RetrievalMode.PRECISE)
        # Success rate should be significant
        assert config.weights["success_rate"] >= 0.3

    def test_diagnostic_ignores_success(self):
        """DIAGNOSTIC mode should not penalize failures."""
        config = get_mode_config(RetrievalMode.DIAGNOSTIC)
        # Success rate weight should be low/zero for debugging
        assert config.weights["success_rate"] <= 0.1

    def test_learning_heavily_weights_similarity(self):
        """LEARNING mode should heavily weight similarity for pattern finding."""
        config = get_mode_config(RetrievalMode.LEARNING)
        assert config.weights["similarity"] >= 0.8

    def test_recall_almost_pure_similarity(self):
        """RECALL mode should be almost pure similarity."""
        config = get_mode_config(RetrievalMode.RECALL)
        assert config.weights["similarity"] >= 0.9


class TestModeDiversitySettings:
    """Tests for diversity settings across modes."""

    def test_broad_has_high_diversity(self):
        """BROAD mode should have high diversity for exploration."""
        config = get_mode_config(RetrievalMode.BROAD)
        assert config.diversity_factor >= 0.7

    def test_precise_has_low_diversity(self):
        """PRECISE mode should have low diversity for focused results."""
        config = get_mode_config(RetrievalMode.PRECISE)
        assert config.diversity_factor <= 0.3

    def test_recall_has_no_diversity(self):
        """RECALL mode should have no diversity (exact match)."""
        config = get_mode_config(RetrievalMode.RECALL)
        assert config.diversity_factor == 0.0
