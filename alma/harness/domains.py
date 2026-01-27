"""
ALMA Domain-Specific Harness Configurations.

Pre-built configurations for common domains:
- Coding: Testing, API development, code review
- Research: Market analysis, competitive intelligence, data synthesis
- Content: Marketing copy, documentation, creative writing
- Operations: Customer support, automation, process management

Each domain includes:
- Default tools and settings
- Memory schema with appropriate scopes
- Agent templates
"""

from typing import Any

from alma.harness.base import (
    Agent,
    Harness,
    MemorySchema,
    Setting,
    Tool,
    ToolType,
)

# =============================================================================
# CODING DOMAIN
# =============================================================================

class CodingDomain:
    """Pre-built configurations for coding/development agents."""

    @staticmethod
    def testing_schema() -> MemorySchema:
        """Memory schema for testing agents (Helena-style)."""
        return MemorySchema(
            domain="testing",
            description="Frontend and backend testing patterns, strategies, and outcomes",
            learnable_categories=[
                "testing_strategies",
                "selector_patterns",
                "ui_component_patterns",
                "form_testing",
                "accessibility_testing",
                "api_testing",
                "database_validation",
                "error_handling_patterns",
            ],
            forbidden_categories=[
                "deployment_procedures",
                "infrastructure_config",
                "security_credentials",
            ],
            heuristic_templates=[
                "When testing {component_type}, {strategy} works {confidence}% of the time",
                "For {error_type} errors, check {diagnostic_steps} first",
                "Selector pattern {pattern} is reliable for {use_case}",
            ],
            outcome_fields=[
                "test_type",
                "component_tested",
                "pass_rate",
                "flakiness_score",
                "execution_time_ms",
            ],
            min_occurrences=3,
        )

    @staticmethod
    def api_dev_schema() -> MemorySchema:
        """Memory schema for API development agents (Victor-style)."""
        return MemorySchema(
            domain="api_development",
            description="API design, implementation patterns, and validation strategies",
            learnable_categories=[
                "api_design_patterns",
                "authentication_patterns",
                "error_handling",
                "performance_optimization",
                "database_query_patterns",
                "caching_strategies",
            ],
            forbidden_categories=[
                "frontend_styling",
                "ui_testing",
                "marketing_content",
            ],
            heuristic_templates=[
                "For {endpoint_type} endpoints, use {pattern} for better {metric}",
                "When handling {error_type}, return {response_pattern}",
                "Database queries for {use_case} perform best with {optimization}",
            ],
            outcome_fields=[
                "endpoint",
                "method",
                "response_time_ms",
                "error_rate",
                "validation_passed",
            ],
            min_occurrences=3,
        )

    @staticmethod
    def testing_setting() -> Setting:
        """Default setting for testing agents."""
        return Setting(
            name="Testing Environment",
            description="Environment for automated testing with browser and API tools",
            tools=[
                Tool(
                    name="playwright",
                    description="Browser automation for UI testing",
                    tool_type=ToolType.EXECUTION,
                    constraints=["Use explicit waits, not sleep()", "Prefer role-based selectors"],
                ),
                Tool(
                    name="api_client",
                    description="HTTP client for API testing",
                    tool_type=ToolType.DATA_ACCESS,
                    constraints=["Log all requests/responses", "Handle timeouts gracefully"],
                ),
                Tool(
                    name="database_query",
                    description="Direct database access for validation",
                    tool_type=ToolType.DATA_ACCESS,
                    constraints=["Read-only queries", "No production writes"],
                ),
            ],
            global_constraints=[
                "Never commit test credentials",
                "Clean up test data after runs",
                "Log all test outcomes for learning",
            ],
        )

    @staticmethod
    def create_helena(alma: Any) -> Harness:
        """Create a Helena-style frontend testing agent."""
        setting = CodingDomain.testing_setting()
        schema = CodingDomain.testing_schema()

        agent = Agent(
            name="helena",
            role="Frontend QA Specialist",
            description=(
                "Expert in frontend testing, Playwright automation, accessibility "
                "validation, and UI consistency. Methodical, thorough, documents everything."
            ),
            memory_schema=schema,
            traits=[
                "Methodical and systematic",
                "Documents edge cases thoroughly",
                "Prioritizes user experience",
                "Catches visual regressions",
            ],
            default_actions=[
                "Test happy path first",
                "Check accessibility for all interactive elements",
                "Validate responsive behavior",
                "Document all findings with screenshots",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)

    @staticmethod
    def create_victor(alma: Any) -> Harness:
        """Create a Victor-style backend testing agent."""
        setting = CodingDomain.testing_setting()
        schema = CodingDomain.api_dev_schema()

        agent = Agent(
            name="victor",
            role="Backend QA Specialist",
            description=(
                "Expert in API testing, database validation, performance testing, "
                "and backend robustness. Focuses on edge cases and error handling."
            ),
            memory_schema=schema,
            traits=[
                "Security-conscious",
                "Performance-focused",
                "Thorough error case coverage",
                "Data integrity guardian",
            ],
            default_actions=[
                "Test authentication first",
                "Verify error responses are informative",
                "Check rate limiting",
                "Validate data persistence",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)


# =============================================================================
# RESEARCH DOMAIN
# =============================================================================

class ResearchDomain:
    """Pre-built configurations for research and analysis agents."""

    @staticmethod
    def market_research_schema() -> MemorySchema:
        """Memory schema for market research agents."""
        return MemorySchema(
            domain="market_research",
            description="Market trends, competitive intelligence, and source reliability",
            learnable_categories=[
                "trend_patterns",
                "source_reliability",
                "query_refinements",
                "data_synthesis_patterns",
                "competitor_tracking",
                "market_indicators",
            ],
            forbidden_categories=[
                "code_implementation",
                "ui_design",
                "personal_data",
            ],
            heuristic_templates=[
                "For {market_segment} research, prioritize {source_type} sources",
                "Query '{refined_query}' yields better results than '{original_query}'",
                "Source {source_name} is {reliability}% accurate for {topic}",
                "Trend pattern: {pattern} typically indicates {outcome}",
            ],
            outcome_fields=[
                "query",
                "sources_used",
                "insights_generated",
                "accuracy_score",
                "time_to_insight_ms",
            ],
            min_occurrences=5,  # Higher threshold for research patterns
        )

    @staticmethod
    def research_setting() -> Setting:
        """Default setting for research agents."""
        return Setting(
            name="Research Environment",
            description="Environment for market research and competitive analysis",
            tools=[
                Tool(
                    name="web_search",
                    description="Search the web for information",
                    tool_type=ToolType.SEARCH,
                    constraints=["Verify sources", "Note publication dates"],
                ),
                Tool(
                    name="data_api",
                    description="Access structured data APIs (financial, market)",
                    tool_type=ToolType.DATA_ACCESS,
                    constraints=["Respect rate limits", "Cache responses"],
                ),
                Tool(
                    name="semantic_search",
                    description="Search internal knowledge bases",
                    tool_type=ToolType.SEARCH,
                    constraints=["Include source citations"],
                ),
                Tool(
                    name="synthesis",
                    description="Combine multiple sources into insights",
                    tool_type=ToolType.ANALYSIS,
                    constraints=["Note conflicting information", "Confidence levels required"],
                ),
            ],
            global_constraints=[
                "Always cite sources",
                "Note data freshness",
                "Flag speculative vs factual statements",
                "Respect copyright and terms of service",
            ],
        )

    @staticmethod
    def create_researcher(alma: Any, focus: str = "general") -> Harness:
        """Create a market research agent."""
        setting = ResearchDomain.research_setting()
        schema = ResearchDomain.market_research_schema()

        agent = Agent(
            name="researcher",
            role="Market Research Analyst",
            description=(
                f"Expert in {focus} market research, competitive intelligence, "
                "and trend analysis. Synthesizes multiple sources into actionable insights."
            ),
            memory_schema=schema,
            traits=[
                "Skeptical of single sources",
                "Quantifies confidence levels",
                "Tracks information provenance",
                "Identifies patterns across data",
            ],
            default_actions=[
                "Search multiple sources for corroboration",
                "Check publication dates and author credibility",
                "Note conflicting information",
                "Summarize with confidence levels",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)


# =============================================================================
# CONTENT DOMAIN
# =============================================================================

class ContentDomain:
    """Pre-built configurations for content creation agents."""

    @staticmethod
    def marketing_schema() -> MemorySchema:
        """Memory schema for marketing content agents."""
        return MemorySchema(
            domain="marketing_content",
            description="Marketing copy patterns, engagement metrics, and audience preferences",
            learnable_categories=[
                "headline_patterns",
                "call_to_action_patterns",
                "audience_preferences",
                "engagement_metrics",
                "tone_effectiveness",
                "visual_pairing",
            ],
            forbidden_categories=[
                "technical_implementation",
                "pricing_decisions",
                "legal_claims",
            ],
            heuristic_templates=[
                "For {audience_segment}, {tone} tone increases engagement by {percent}%",
                "Headline pattern '{pattern}' performs best for {content_type}",
                "Call-to-action '{cta}' converts {rate}% for {context}",
                "Visual style {style} resonates with {audience}",
            ],
            outcome_fields=[
                "content_type",
                "target_audience",
                "engagement_rate",
                "conversion_rate",
                "a_b_test_winner",
            ],
            min_occurrences=5,
        )

    @staticmethod
    def documentation_schema() -> MemorySchema:
        """Memory schema for technical documentation agents."""
        return MemorySchema(
            domain="documentation",
            description="Documentation patterns, clarity metrics, and user comprehension",
            learnable_categories=[
                "explanation_patterns",
                "code_example_patterns",
                "structure_preferences",
                "terminology_choices",
                "diagram_usage",
            ],
            forbidden_categories=[
                "marketing_claims",
                "pricing_information",
                "security_details",
            ],
            heuristic_templates=[
                "For {concept_complexity}, use {explanation_approach}",
                "Code examples should demonstrate {pattern} for {use_case}",
                "Users understand {term_a} better than {term_b}",
            ],
            outcome_fields=[
                "doc_type",
                "reader_level",
                "comprehension_score",
                "time_to_understanding",
                "questions_reduced",
            ],
            min_occurrences=3,
        )

    @staticmethod
    def content_setting() -> Setting:
        """Default setting for content creation agents."""
        return Setting(
            name="Content Creation Environment",
            description="Environment for creating marketing and documentation content",
            tools=[
                Tool(
                    name="trend_search",
                    description="Search for trending topics and phrases",
                    tool_type=ToolType.SEARCH,
                    constraints=["Note trend velocity", "Check regional variations"],
                ),
                Tool(
                    name="image_search",
                    description="Find relevant images and visual assets",
                    tool_type=ToolType.SEARCH,
                    constraints=["Verify licensing", "Check brand alignment"],
                ),
                Tool(
                    name="analytics",
                    description="Access content performance metrics",
                    tool_type=ToolType.ANALYSIS,
                    constraints=["Compare against baselines", "Note sample sizes"],
                ),
                Tool(
                    name="style_check",
                    description="Validate content against style guides",
                    tool_type=ToolType.ANALYSIS,
                    constraints=["Flag deviations", "Suggest alternatives"],
                ),
            ],
            global_constraints=[
                "Adhere to brand voice guidelines",
                "No unsubstantiated claims",
                "Respect copyright",
                "Include accessibility considerations",
            ],
        )

    @staticmethod
    def create_copywriter(alma: Any) -> Harness:
        """Create a marketing copywriter agent."""
        setting = ContentDomain.content_setting()
        schema = ContentDomain.marketing_schema()

        agent = Agent(
            name="copywriter",
            role="Marketing Copywriter",
            description=(
                "Expert in persuasive writing, audience engagement, and brand voice. "
                "Creates compelling content that drives action while staying authentic."
            ),
            memory_schema=schema,
            traits=[
                "Audience-first thinking",
                "Data-informed creativity",
                "Brand voice guardian",
                "A/B testing mindset",
            ],
            default_actions=[
                "Research target audience first",
                "Generate multiple headline options",
                "Include clear call-to-action",
                "Check against brand guidelines",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)

    @staticmethod
    def create_documenter(alma: Any) -> Harness:
        """Create a technical documentation agent."""
        setting = ContentDomain.content_setting()
        schema = ContentDomain.documentation_schema()

        agent = Agent(
            name="documenter",
            role="Technical Writer",
            description=(
                "Expert in explaining complex concepts clearly. Creates documentation "
                "that helps users succeed with minimal friction."
            ),
            memory_schema=schema,
            traits=[
                "Clarity obsessed",
                "User empathy",
                "Example-driven",
                "Progressive disclosure",
            ],
            default_actions=[
                "Start with user's goal",
                "Use concrete examples",
                "Include common pitfalls",
                "Link related concepts",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)


# =============================================================================
# OPERATIONS DOMAIN
# =============================================================================

class OperationsDomain:
    """Pre-built configurations for operations and support agents."""

    @staticmethod
    def support_schema() -> MemorySchema:
        """Memory schema for customer support agents."""
        return MemorySchema(
            domain="customer_support",
            description="Issue resolution patterns, customer preferences, and escalation paths",
            learnable_categories=[
                "issue_patterns",
                "resolution_steps",
                "customer_preferences",
                "escalation_triggers",
                "product_knowledge",
                "communication_styles",
            ],
            forbidden_categories=[
                "billing_modifications",
                "security_overrides",
                "contract_changes",
            ],
            heuristic_templates=[
                "For {issue_type}, try {resolution_steps} first (resolves {rate}%)",
                "Customer {customer_type} prefers {communication_style}",
                "Escalate when {condition} - resolution rate drops after {threshold}",
                "Product issue {issue} is usually caused by {root_cause}",
            ],
            outcome_fields=[
                "issue_type",
                "resolution_time_ms",
                "customer_satisfaction",
                "escalated",
                "first_contact_resolution",
            ],
            min_occurrences=5,
        )

    @staticmethod
    def automation_schema() -> MemorySchema:
        """Memory schema for automation/workflow agents."""
        return MemorySchema(
            domain="automation",
            description="Workflow patterns, optimization opportunities, and failure modes",
            learnable_categories=[
                "workflow_patterns",
                "optimization_opportunities",
                "failure_modes",
                "retry_strategies",
                "resource_usage",
            ],
            forbidden_categories=[
                "manual_overrides",
                "security_bypasses",
                "data_deletion",
            ],
            heuristic_templates=[
                "Workflow {workflow} fails {rate}% when {condition}",
                "Retry strategy {strategy} succeeds for {error_type} after {attempts} attempts",
                "Resource {resource} peaks at {time} - schedule {action} accordingly",
            ],
            outcome_fields=[
                "workflow_name",
                "execution_time_ms",
                "success_rate",
                "resource_usage",
                "error_type",
            ],
            min_occurrences=10,  # Higher threshold for automation patterns
        )

    @staticmethod
    def operations_setting() -> Setting:
        """Default setting for operations agents."""
        return Setting(
            name="Operations Environment",
            description="Environment for customer support and automation tasks",
            tools=[
                Tool(
                    name="ticket_system",
                    description="Access and update support tickets",
                    tool_type=ToolType.DATA_ACCESS,
                    constraints=["Log all actions", "Respect SLAs"],
                ),
                Tool(
                    name="knowledge_base",
                    description="Search internal documentation and solutions",
                    tool_type=ToolType.SEARCH,
                    constraints=["Check article freshness", "Flag outdated content"],
                ),
                Tool(
                    name="customer_history",
                    description="Access customer interaction history",
                    tool_type=ToolType.DATA_ACCESS,
                    constraints=["Respect privacy", "Only relevant history"],
                ),
                Tool(
                    name="escalation",
                    description="Escalate to human agents or specialists",
                    tool_type=ToolType.COMMUNICATION,
                    constraints=["Include context", "Note attempted resolutions"],
                ),
            ],
            global_constraints=[
                "Never share customer data externally",
                "Log all customer interactions",
                "Escalate security concerns immediately",
                "Maintain professional tone",
            ],
        )

    @staticmethod
    def create_support_agent(alma: Any) -> Harness:
        """Create a customer support agent."""
        setting = OperationsDomain.operations_setting()
        schema = OperationsDomain.support_schema()

        agent = Agent(
            name="support",
            role="Customer Support Specialist",
            description=(
                "Expert in resolving customer issues efficiently and empathetically. "
                "Balances speed with thoroughness, knows when to escalate."
            ),
            memory_schema=schema,
            traits=[
                "Empathetic communication",
                "Efficient problem-solving",
                "Knows when to escalate",
                "Proactive issue prevention",
            ],
            default_actions=[
                "Acknowledge the issue",
                "Check known solutions first",
                "Gather necessary context",
                "Follow up to confirm resolution",
            ],
        )

        return Harness(setting=setting, agent=agent, alma=alma)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_harness(
    domain: str,
    agent_type: str,
    alma: Any,
    **kwargs,
) -> Harness:
    """
    Factory function to create domain-specific harnesses.

    Args:
        domain: One of "coding", "research", "content", "operations"
        agent_type: Specific agent within domain
        alma: ALMA instance
        **kwargs: Additional configuration

    Returns:
        Configured Harness instance

    Examples:
        harness = create_harness("coding", "helena", alma)
        harness = create_harness("research", "researcher", alma, focus="biotech")
        harness = create_harness("content", "copywriter", alma)
        harness = create_harness("operations", "support", alma)
    """
    factories = {
        "coding": {
            "helena": CodingDomain.create_helena,
            "victor": CodingDomain.create_victor,
        },
        "research": {
            "researcher": lambda a: ResearchDomain.create_researcher(a, kwargs.get("focus", "general")),
        },
        "content": {
            "copywriter": ContentDomain.create_copywriter,
            "documenter": ContentDomain.create_documenter,
        },
        "operations": {
            "support": OperationsDomain.create_support_agent,
        },
    }

    if domain not in factories:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(factories.keys())}")

    if agent_type not in factories[domain]:
        raise ValueError(
            f"Unknown agent type '{agent_type}' for domain '{domain}'. "
            f"Available: {list(factories[domain].keys())}"
        )

    return factories[domain][agent_type](alma)
