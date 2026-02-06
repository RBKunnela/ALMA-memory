"""
ALMA Learning Validation.

Enforces scope constraints and validates learning requests.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from alma.types import MemoryScope

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of a validation check."""

    ALLOWED = "allowed"
    DENIED_OUT_OF_SCOPE = "denied_out_of_scope"
    DENIED_FORBIDDEN = "denied_forbidden"
    DENIED_UNKNOWN_AGENT = "denied_unknown_agent"
    WARNING_NO_SCOPE = "warning_no_scope"


@dataclass
class ValidationReport:
    """Detailed report of a validation check."""

    result: ValidationResult
    agent: str
    domain: str
    reason: str
    allowed_domains: List[str] = field(default_factory=list)
    forbidden_domains: List[str] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        """Check if the validation passed."""
        return self.result in (
            ValidationResult.ALLOWED,
            ValidationResult.WARNING_NO_SCOPE,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result.value,
            "agent": self.agent,
            "domain": self.domain,
            "reason": self.reason,
            "is_allowed": self.is_allowed,
            "allowed_domains": self.allowed_domains,
            "forbidden_domains": self.forbidden_domains,
        }


class ScopeValidator:
    """
    Validates that learning requests are within agent scope.

    Provides strict enforcement with detailed reporting.
    """

    def __init__(
        self,
        scopes: Dict[str, MemoryScope],
        strict_mode: bool = True,
        allow_unknown_agents: bool = False,
    ):
        """
        Initialize validator.

        Args:
            scopes: Dict of agent_name -> MemoryScope
            strict_mode: If True, deny requests for unknown domains
            allow_unknown_agents: If True, allow learning for agents without scopes
        """
        self.scopes = scopes
        self.strict_mode = strict_mode
        self.allow_unknown_agents = allow_unknown_agents

        # Track validation statistics
        self._stats = {
            "total_validations": 0,
            "allowed": 0,
            "denied": 0,
            "warnings": 0,
        }

    def validate(
        self,
        agent: str,
        domain: str,
        task_type: Optional[str] = None,
    ) -> ValidationReport:
        """
        Validate a learning request.

        Args:
            agent: Agent attempting to learn
            domain: Knowledge domain to learn in
            task_type: Optional task type for context

        Returns:
            ValidationReport with detailed results
        """
        self._stats["total_validations"] += 1

        # Check if agent has a scope
        scope = self.scopes.get(agent)

        if scope is None:
            if self.allow_unknown_agents:
                self._stats["warnings"] += 1
                logger.warning(f"Agent '{agent}' has no defined scope, allowing anyway")
                return ValidationReport(
                    result=ValidationResult.WARNING_NO_SCOPE,
                    agent=agent,
                    domain=domain,
                    reason=f"Agent '{agent}' has no defined scope",
                )
            else:
                self._stats["denied"] += 1
                logger.warning(f"Agent '{agent}' denied: no scope defined")
                return ValidationReport(
                    result=ValidationResult.DENIED_UNKNOWN_AGENT,
                    agent=agent,
                    domain=domain,
                    reason=f"Agent '{agent}' has no defined scope and unknown agents are not allowed",
                )

        # Check if domain is explicitly forbidden
        if domain in scope.cannot_learn:
            self._stats["denied"] += 1
            logger.warning(
                f"Agent '{agent}' denied learning in '{domain}': explicitly forbidden"
            )
            return ValidationReport(
                result=ValidationResult.DENIED_FORBIDDEN,
                agent=agent,
                domain=domain,
                reason=f"Domain '{domain}' is explicitly forbidden for agent '{agent}'",
                allowed_domains=scope.can_learn,
                forbidden_domains=scope.cannot_learn,
            )

        # Check if domain is in allowed list
        if scope.can_learn:  # If list is not empty, it's an allowlist
            if domain not in scope.can_learn:
                # Check for partial matches (e.g., "form_testing" contains "testing")
                partial_match = any(
                    allowed in domain or domain in allowed
                    for allowed in scope.can_learn
                )

                if not partial_match and self.strict_mode:
                    self._stats["denied"] += 1
                    logger.warning(
                        f"Agent '{agent}' denied learning in '{domain}': not in allowed list"
                    )
                    return ValidationReport(
                        result=ValidationResult.DENIED_OUT_OF_SCOPE,
                        agent=agent,
                        domain=domain,
                        reason=f"Domain '{domain}' is not in agent '{agent}'s allowed domains",
                        allowed_domains=scope.can_learn,
                        forbidden_domains=scope.cannot_learn,
                    )

        # Allowed
        self._stats["allowed"] += 1
        return ValidationReport(
            result=ValidationResult.ALLOWED,
            agent=agent,
            domain=domain,
            reason="Learning allowed",
            allowed_domains=scope.can_learn,
            forbidden_domains=scope.cannot_learn,
        )

    def validate_batch(
        self,
        agent: str,
        domains: List[str],
    ) -> Dict[str, ValidationReport]:
        """
        Validate multiple domains at once.

        Args:
            agent: Agent attempting to learn
            domains: List of domains to validate

        Returns:
            Dict of domain -> ValidationReport
        """
        return {domain: self.validate(agent, domain) for domain in domains}

    def get_allowed_domains(self, agent: str) -> Set[str]:
        """Get all allowed domains for an agent."""
        scope = self.scopes.get(agent)
        if scope is None:
            return set()
        return set(scope.can_learn)

    def get_forbidden_domains(self, agent: str) -> Set[str]:
        """Get all forbidden domains for an agent."""
        scope = self.scopes.get(agent)
        if scope is None:
            return set()
        return set(scope.cannot_learn)

    def is_allowed(self, agent: str, domain: str) -> bool:
        """Quick check if learning is allowed (no detailed report)."""
        return self.validate(agent, domain).is_allowed

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self._stats["total_validations"]
        return {
            **self._stats,
            "allow_rate": self._stats["allowed"] / total if total > 0 else 0,
            "deny_rate": self._stats["denied"] / total if total > 0 else 0,
        }

    def reset_stats(self):
        """Reset validation statistics."""
        self._stats = {
            "total_validations": 0,
            "allowed": 0,
            "denied": 0,
            "warnings": 0,
        }


class TaskTypeValidator:
    """
    Validates and normalizes task types.

    Ensures consistent categorization across learning events.
    """

    # Standard task type categories
    STANDARD_TYPES = {
        "testing": ["test", "validate", "verify", "check", "qa"],
        "form_testing": ["form", "input", "field", "validation"],
        "api_testing": ["api", "endpoint", "rest", "graphql", "request"],
        "database_validation": ["database", "query", "sql", "schema"],
        "ui_testing": ["ui", "component", "button", "click", "element"],
        "performance_testing": ["performance", "load", "stress", "speed"],
        "security_testing": ["security", "auth", "permission", "xss", "injection"],
        "accessibility_testing": ["accessibility", "a11y", "aria", "screen reader"],
    }

    def __init__(self, custom_types: Optional[Dict[str, List[str]]] = None):
        """
        Initialize validator.

        Args:
            custom_types: Additional custom type mappings
        """
        self.type_mappings = {**self.STANDARD_TYPES}
        if custom_types:
            self.type_mappings.update(custom_types)

        # Build reverse lookup
        self._keyword_to_type: Dict[str, str] = {}
        for task_type, keywords in self.type_mappings.items():
            for keyword in keywords:
                self._keyword_to_type[keyword.lower()] = task_type

    def infer_type(self, task_description: str) -> str:
        """
        Infer task type from description.

        Args:
            task_description: Description of the task

        Returns:
            Inferred task type or "general"
        """
        task_lower = task_description.lower()

        # Check for keyword matches
        scores: Dict[str, int] = {}
        for task_type, keywords in self.type_mappings.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[task_type] = score

        if scores:
            # Return the type with highest score
            return max(scores.keys(), key=lambda k: scores[k])

        return "general"

    def normalize_type(self, task_type: str) -> str:
        """
        Normalize a task type to standard category.

        Args:
            task_type: Raw task type

        Returns:
            Normalized task type
        """
        type_lower = task_type.lower().replace("_", " ").replace("-", " ")

        # Check direct match
        if type_lower in self.type_mappings:
            return type_lower

        # Check keyword lookup
        for word in type_lower.split():
            if word in self._keyword_to_type:
                return self._keyword_to_type[word]

        return task_type  # Return original if no match

    def validate_type(self, task_type: str) -> bool:
        """Check if task type is recognized."""
        normalized = self.normalize_type(task_type)
        return normalized in self.type_mappings or task_type == "general"


def validate_learning_request(
    agent: str,
    domain: str,
    scopes: Dict[str, MemoryScope],
    strict: bool = True,
) -> ValidationReport:
    """
    Convenience function for one-off validation.

    Args:
        agent: Agent attempting to learn
        domain: Knowledge domain
        scopes: Dict of agent scopes
        strict: Use strict mode

    Returns:
        ValidationReport
    """
    validator = ScopeValidator(scopes, strict_mode=strict)
    return validator.validate(agent, domain)
