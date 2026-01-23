"""
ALMA Helena Integration.

Helena-specific integration for frontend QA testing with ALMA memory.

Helena specializes in:
- Playwright automation
- UI/UX testing
- Accessibility validation
- Visual regression testing
- Form testing patterns

This module provides Helena-specific memory categories, prompts, and utilities.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from alma.core import ALMA
from alma.types import MemorySlice
from alma.harness.domains import CodingDomain
from alma.integration.claude_agents import (
    ClaudeAgentHooks,
    TaskContext,
    TaskOutcome,
    AgentType,
)

logger = logging.getLogger(__name__)


# Helena's learning categories
HELENA_CATEGORIES = [
    "testing_strategies",
    "selector_patterns",
    "ui_component_patterns",
    "form_testing",
    "accessibility_testing",
]

# Categories Helena should NOT learn
HELENA_FORBIDDEN = [
    "backend_logic",
    "database_queries",
    "api_design",
    "infrastructure",
]


@dataclass
class UITestContext(TaskContext):
    """
    Helena-specific test context.

    Extends TaskContext with UI testing-specific fields.
    """
    component_type: Optional[str] = None
    page_url: Optional[str] = None
    viewport: Optional[Dict[str, int]] = None
    browser: str = "chromium"
    is_accessibility_test: bool = False
    is_visual_test: bool = False

    def __post_init__(self):
        # Ensure agent name is helena
        self.agent_name = "helena"
        # Set default task type if not specified
        if not self.task_type:
            self.task_type = self._infer_task_type()

    def _infer_task_type(self) -> str:
        """Infer task type from context."""
        if self.is_accessibility_test:
            return "accessibility_testing"
        if self.is_visual_test:
            return "visual_testing"
        if self.component_type:
            if "form" in self.component_type.lower():
                return "form_testing"
            return "ui_component_patterns"
        return "testing_strategies"


@dataclass
class UITestOutcome(TaskOutcome):
    """
    Helena-specific test outcome.

    Extends TaskOutcome with UI testing-specific results.
    """
    selectors_used: List[str] = field(default_factory=list)
    accessibility_issues: List[Dict[str, Any]] = field(default_factory=list)
    visual_diffs: List[str] = field(default_factory=list)
    flaky_elements: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Add selectors to tools_used
        if self.selectors_used and not self.tools_used:
            self.tools_used = self.selectors_used


class HelenaHooks(ClaudeAgentHooks):
    """
    Helena-specific integration hooks.

    Extends ClaudeAgentHooks with UI testing-specific functionality.
    """

    def __init__(self, alma: ALMA, auto_learn: bool = True):
        """
        Initialize Helena hooks.

        Args:
            alma: ALMA instance
            auto_learn: Whether to automatically learn from outcomes
        """
        harness = CodingDomain.create_helena(alma)
        super().__init__(
            alma=alma,
            agent_type=AgentType.HELENA,
            harness=harness,
            auto_learn=auto_learn,
        )

    def get_selector_patterns(
        self,
        component_type: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get proven selector patterns for a component type.

        Args:
            component_type: Type of UI component (button, input, modal, etc.)
            top_k: Maximum patterns to return

        Returns:
            List of selector patterns with success rates
        """
        memories = self.alma.retrieve(
            task=f"selector patterns for {component_type}",
            agent=self.agent_name,
            top_k=top_k,
        )

        patterns = []
        for h in memories.heuristics:
            if "selector" in h.condition.lower():
                patterns.append({
                    "pattern": h.strategy,
                    "confidence": h.confidence,
                    "occurrences": h.occurrence_count,
                })

        return patterns

    def get_form_testing_strategies(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get proven form testing strategies.

        Returns strategies for testing forms, validation, and submissions.
        """
        memories = self.alma.retrieve(
            task="form testing validation submit strategies",
            agent=self.agent_name,
            top_k=top_k,
        )

        strategies = []
        for h in memories.heuristics:
            if any(kw in h.condition.lower() for kw in ["form", "validation", "input"]):
                strategies.append({
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                })

        return strategies

    def get_accessibility_patterns(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get accessibility testing patterns.

        Returns patterns for ARIA, keyboard navigation, screen readers.
        """
        memories = self.alma.retrieve(
            task="accessibility ARIA keyboard screen reader testing",
            agent=self.agent_name,
            top_k=top_k,
        )

        patterns = []
        for h in memories.heuristics:
            if any(kw in h.condition.lower() for kw in ["access", "aria", "keyboard"]):
                patterns.append({
                    "condition": h.condition,
                    "strategy": h.strategy,
                    "confidence": h.confidence,
                })

        for dk in memories.domain_knowledge:
            if dk.domain == "accessibility_testing":
                patterns.append({
                    "fact": dk.fact,
                    "source": dk.source,
                })

        return patterns

    def record_selector_pattern(
        self,
        selector: str,
        component_type: str,
        success: bool,
        stability_score: float = 1.0,
    ) -> bool:
        """
        Record a selector pattern for learning.

        Args:
            selector: The CSS/XPath selector used
            component_type: Type of component targeted
            success: Whether the selector worked
            stability_score: How stable the selector is (0-1)

        Returns:
            True if recorded successfully
        """
        # Add as domain knowledge with stability metadata
        description = f"Selector '{selector}' for {component_type}"
        if stability_score < 0.7:
            description += " (may be flaky)"

        return self.add_knowledge(
            domain="selector_patterns",
            fact=description,
            source=f"test_run:stability={stability_score:.2f}",
        )

    def record_accessibility_issue(
        self,
        issue_type: str,
        element: str,
        fix_suggestion: str,
    ) -> bool:
        """
        Record an accessibility issue found.

        Args:
            issue_type: Type of issue (ARIA, contrast, keyboard, etc.)
            element: Element with the issue
            fix_suggestion: How to fix it

        Returns:
            True if recorded successfully
        """
        fact = f"Issue: {issue_type} on {element}. Fix: {fix_suggestion}"
        return self.add_knowledge(
            domain="accessibility_testing",
            fact=fact,
            source="accessibility_audit",
        )

    def format_ui_test_prompt(
        self,
        memories: MemorySlice,
        test_context: UITestContext,
    ) -> str:
        """
        Format memories for Helena's UI testing prompt.

        Provides Helena-specific formatting with test context.
        """
        sections = []

        # Base memory formatting
        base_format = self.format_memories_for_prompt(memories)
        if base_format:
            sections.append(base_format)

        # Add test context
        sections.append("\n## Current Test Context")
        sections.append(f"- **Task**: {test_context.task_description}")
        sections.append(f"- **Task Type**: {test_context.task_type}")

        if test_context.component_type:
            sections.append(f"- **Component**: {test_context.component_type}")
        if test_context.page_url:
            sections.append(f"- **Page**: {test_context.page_url}")
        if test_context.viewport:
            sections.append(
                f"- **Viewport**: {test_context.viewport.get('width', '?')}x"
                f"{test_context.viewport.get('height', '?')}"
            )

        if test_context.is_accessibility_test:
            sections.append("- **Focus**: Accessibility validation")
        if test_context.is_visual_test:
            sections.append("- **Focus**: Visual regression testing")

        return "\n".join(sections)


def create_helena_hooks(alma: ALMA, auto_learn: bool = True) -> HelenaHooks:
    """
    Convenience function to create Helena hooks.

    Args:
        alma: ALMA instance
        auto_learn: Whether to automatically learn

    Returns:
        Configured HelenaHooks
    """
    return HelenaHooks(alma=alma, auto_learn=auto_learn)


def helena_pre_task(
    alma: ALMA,
    task: str,
    component_type: Optional[str] = None,
    page_url: Optional[str] = None,
    project_id: str = "default",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function for Helena pre-task hook.

    Quick integration without creating full hooks object.

    Args:
        alma: ALMA instance
        task: Task description
        component_type: Optional component type
        page_url: Optional page URL
        project_id: Project ID
        top_k: Max memories per type

    Returns:
        Dict with memories and formatted prompt
    """
    hooks = HelenaHooks(alma=alma, auto_learn=False)

    context = UITestContext(
        task_description=task,
        task_type="",  # Will be inferred
        agent_name="helena",
        project_id=project_id,
        component_type=component_type,
        page_url=page_url,
    )

    memories = hooks.pre_task(context, top_k=top_k)
    prompt = hooks.format_ui_test_prompt(memories, context)

    return {
        "memories": memories,
        "prompt": prompt,
        "context": context,
    }


def helena_post_task(
    alma: ALMA,
    task: str,
    success: bool,
    strategy_used: str,
    selectors_used: Optional[List[str]] = None,
    accessibility_issues: Optional[List[Dict[str, Any]]] = None,
    project_id: str = "default",
    duration_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    Convenience function for Helena post-task hook.

    Quick integration without creating full hooks object.

    Args:
        alma: ALMA instance
        task: Task description
        success: Whether task succeeded
        strategy_used: Strategy used
        selectors_used: Selectors used during test
        accessibility_issues: Any accessibility issues found
        project_id: Project ID
        duration_ms: Task duration
        error_message: Error if failed

    Returns:
        True if learning was recorded
    """
    hooks = HelenaHooks(alma=alma, auto_learn=True)

    context = UITestContext(
        task_description=task,
        task_type="",  # Will be inferred
        agent_name="helena",
        project_id=project_id,
    )

    outcome = UITestOutcome(
        success=success,
        strategy_used=strategy_used,
        selectors_used=selectors_used or [],
        accessibility_issues=accessibility_issues or [],
        duration_ms=duration_ms,
        error_message=error_message,
    )

    return hooks.post_task(context, outcome)
