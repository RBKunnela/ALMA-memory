"""
Unit tests for ALMA Learning Validation.
"""

import pytest

from alma.learning.validation import (
    ScopeValidator,
    TaskTypeValidator,
    ValidationReport,
    ValidationResult,
    validate_learning_request,
)
from alma.types import MemoryScope


class TestScopeValidator:
    """Tests for ScopeValidator."""

    @pytest.fixture
    def scopes(self):
        """Create sample scopes for testing."""
        return {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing_strategies", "ui_patterns", "form_testing"],
                cannot_learn=["backend_logic", "database"],
                min_occurrences_for_heuristic=3,
            ),
            "victor": MemoryScope(
                agent_name="victor",
                can_learn=["api_testing", "database_validation"],
                cannot_learn=["frontend_logic", "ui_testing"],
                min_occurrences_for_heuristic=3,
            ),
            "open_agent": MemoryScope(
                agent_name="open_agent",
                can_learn=[],  # Empty = all allowed
                cannot_learn=["forbidden_domain"],
                min_occurrences_for_heuristic=3,
            ),
        }

    def test_allowed_domain(self, scopes):
        """Test that allowed domains pass validation."""
        validator = ScopeValidator(scopes)

        result = validator.validate("helena", "testing_strategies")

        assert result.is_allowed
        assert result.result == ValidationResult.ALLOWED

    def test_forbidden_domain_denied(self, scopes):
        """Test that forbidden domains are denied."""
        validator = ScopeValidator(scopes)

        result = validator.validate("helena", "backend_logic")

        assert not result.is_allowed
        assert result.result == ValidationResult.DENIED_FORBIDDEN
        assert "forbidden" in result.reason.lower()

    def test_out_of_scope_denied_strict(self, scopes):
        """Test that out-of-scope domains are denied in strict mode."""
        validator = ScopeValidator(scopes, strict_mode=True)

        # "research" is not in helena's allowed list
        result = validator.validate("helena", "research")

        assert not result.is_allowed
        assert result.result == ValidationResult.DENIED_OUT_OF_SCOPE

    def test_out_of_scope_partial_match(self, scopes):
        """Test that partial matches are allowed."""
        validator = ScopeValidator(scopes, strict_mode=True)

        # "form_testing" should match "form_testing"
        result = validator.validate("helena", "form_testing")

        assert result.is_allowed

    def test_unknown_agent_strict(self, scopes):
        """Test that unknown agents are denied when not allowed."""
        validator = ScopeValidator(scopes, allow_unknown_agents=False)

        result = validator.validate("unknown_agent", "some_domain")

        assert not result.is_allowed
        assert result.result == ValidationResult.DENIED_UNKNOWN_AGENT

    def test_unknown_agent_allowed(self, scopes):
        """Test that unknown agents can be allowed."""
        validator = ScopeValidator(scopes, allow_unknown_agents=True)

        result = validator.validate("unknown_agent", "some_domain")

        assert result.is_allowed
        assert result.result == ValidationResult.WARNING_NO_SCOPE

    def test_open_scope_allows_all_except_forbidden(self, scopes):
        """Test that empty can_learn list allows all except forbidden."""
        validator = ScopeValidator(scopes, strict_mode=True)

        # open_agent has empty can_learn, should allow anything
        result = validator.validate("open_agent", "random_domain")
        assert result.is_allowed

        # But not forbidden domains
        result = validator.validate("open_agent", "forbidden_domain")
        assert not result.is_allowed

    def test_validation_stats(self, scopes):
        """Test that statistics are tracked."""
        validator = ScopeValidator(scopes)

        # Generate some validations
        validator.validate("helena", "testing_strategies")  # allowed
        validator.validate("helena", "backend_logic")  # denied
        validator.validate("helena", "ui_patterns")  # allowed

        stats = validator.get_stats()

        assert stats["total_validations"] == 3
        assert stats["allowed"] == 2
        assert stats["denied"] == 1

    def test_batch_validation(self, scopes):
        """Test batch validation of multiple domains."""
        validator = ScopeValidator(scopes)

        domains = ["testing_strategies", "backend_logic", "ui_patterns"]
        results = validator.validate_batch("helena", domains)

        assert len(results) == 3
        assert results["testing_strategies"].is_allowed
        assert not results["backend_logic"].is_allowed
        assert results["ui_patterns"].is_allowed

    def test_get_allowed_domains(self, scopes):
        """Test getting allowed domains for an agent."""
        validator = ScopeValidator(scopes)

        allowed = validator.get_allowed_domains("helena")

        assert "testing_strategies" in allowed
        assert "ui_patterns" in allowed
        assert "backend_logic" not in allowed

    def test_get_forbidden_domains(self, scopes):
        """Test getting forbidden domains for an agent."""
        validator = ScopeValidator(scopes)

        forbidden = validator.get_forbidden_domains("helena")

        assert "backend_logic" in forbidden
        assert "database" in forbidden

    def test_is_allowed_quick_check(self, scopes):
        """Test quick check method."""
        validator = ScopeValidator(scopes)

        assert validator.is_allowed("helena", "testing_strategies")
        assert not validator.is_allowed("helena", "backend_logic")


class TestTaskTypeValidator:
    """Tests for TaskTypeValidator."""

    def test_infer_type_testing(self):
        """Test inferring testing task types."""
        validator = TaskTypeValidator()

        # Use cases where specific type keywords have higher scores
        # "form input validation" matches "form", "input", "validation" -> form_testing (3)
        assert validator.infer_type("Validate form input field") == "form_testing"
        # "api endpoint request" matches "api", "endpoint", "request" -> api_testing (3)
        assert validator.infer_type("Test API endpoint request") == "api_testing"
        # "database query" matches both keywords -> database_validation (2)
        assert (
            validator.infer_type("Run database query validation")
            == "database_validation"
        )

    def test_infer_type_ui(self):
        """Test inferring UI task types."""
        validator = TaskTypeValidator()

        assert validator.infer_type("Click the button") == "ui_testing"
        assert validator.infer_type("Verify the component") == "testing"

    def test_infer_type_general(self):
        """Test that unknown tasks return general."""
        validator = TaskTypeValidator()

        result = validator.infer_type("Do something random")
        assert result == "general"

    def test_normalize_type(self):
        """Test type normalization."""
        validator = TaskTypeValidator()

        assert validator.normalize_type("form_testing") == "form_testing"
        assert validator.normalize_type("api") == "api_testing"

    def test_validate_type(self):
        """Test type validation."""
        validator = TaskTypeValidator()

        assert validator.validate_type("testing")
        assert validator.validate_type("general")
        # Unknown types still validate (might be custom)

    def test_custom_types(self):
        """Test custom type definitions."""
        custom = {"ml_testing": ["model", "inference", "prediction"]}
        validator = TaskTypeValidator(custom_types=custom)

        result = validator.infer_type("Test the ML model inference")
        assert result == "ml_testing"


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_is_allowed_allowed(self):
        """Test is_allowed for allowed result."""
        report = ValidationReport(
            result=ValidationResult.ALLOWED,
            agent="test",
            domain="test_domain",
            reason="OK",
        )
        assert report.is_allowed

    def test_is_allowed_warning(self):
        """Test is_allowed for warning result (still allowed)."""
        report = ValidationReport(
            result=ValidationResult.WARNING_NO_SCOPE,
            agent="test",
            domain="test_domain",
            reason="Warning",
        )
        assert report.is_allowed

    def test_is_allowed_denied(self):
        """Test is_allowed for denied result."""
        report = ValidationReport(
            result=ValidationResult.DENIED_FORBIDDEN,
            agent="test",
            domain="test_domain",
            reason="Denied",
        )
        assert not report.is_allowed

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ValidationReport(
            result=ValidationResult.ALLOWED,
            agent="helena",
            domain="testing",
            reason="OK",
            allowed_domains=["testing", "ui"],
            forbidden_domains=["backend"],
        )

        d = report.to_dict()

        assert d["result"] == "allowed"
        assert d["agent"] == "helena"
        assert d["is_allowed"] is True


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_validate_learning_request(self):
        """Test the convenience function."""
        scopes = {
            "helena": MemoryScope(
                agent_name="helena",
                can_learn=["testing"],
                cannot_learn=["backend"],
                min_occurrences_for_heuristic=3,
            ),
        }

        result = validate_learning_request("helena", "testing", scopes)
        assert result.is_allowed

        result = validate_learning_request("helena", "backend", scopes)
        assert not result.is_allowed
