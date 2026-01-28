"""
Pre-built Domain Schemas.

Standard domain schemas for common use cases.
"""

from alma.domains.types import DomainSchema


def get_coding_schema() -> DomainSchema:
    """
    Pre-built schema for coding workflows.

    This is the formalized version of ALMA's original Helena/Victor schema.
    Suitable for: Frontend testing, backend testing, general development.
    """
    schema = DomainSchema.create(
        name="coding",
        description="Memory schema for software development workflows",
        learning_categories=[
            "testing_strategies",
            "selector_patterns",
            "api_design_patterns",
            "error_handling",
            "performance_optimization",
            "debugging_techniques",
            "code_review_patterns",
            "refactoring_strategies",
        ],
    )

    # Entity types
    schema.add_entity_type(
        name="feature",
        description="A software feature or capability",
        attributes=["status", "tests", "files", "priority", "owner"],
    )
    schema.add_entity_type(
        name="bug",
        description="A software defect or issue",
        attributes=["severity", "reproduction_steps", "fix", "status", "root_cause"],
    )
    schema.add_entity_type(
        name="test",
        description="A test case or test suite",
        attributes=["type", "status", "coverage", "flaky", "last_run"],
    )
    schema.add_entity_type(
        name="component",
        description="A code component or module",
        attributes=["path", "type", "dependencies", "tests"],
    )
    schema.add_entity_type(
        name="api_endpoint",
        description="An API endpoint",
        attributes=["method", "path", "request_schema", "response_schema", "auth"],
    )

    # Relationships
    schema.add_relationship_type(
        name="tests",
        description="Test covers a feature or component",
        source_type="test",
        target_type="feature",
    )
    schema.add_relationship_type(
        name="fixes",
        description="Commit or change fixes a bug",
        source_type="feature",
        target_type="bug",
    )
    schema.add_relationship_type(
        name="depends_on",
        description="Component depends on another component",
        source_type="component",
        target_type="component",
    )
    schema.add_relationship_type(
        name="implements",
        description="Component implements an API endpoint",
        source_type="component",
        target_type="api_endpoint",
    )

    return schema


def get_research_schema() -> DomainSchema:
    """
    Pre-built schema for research workflows.

    Suitable for: Literature review, hypothesis testing, academic research.
    """
    schema = DomainSchema.create(
        name="research",
        description="Memory schema for research and academic workflows",
        learning_categories=[
            "literature_review_patterns",
            "methodology_selection",
            "data_analysis_strategies",
            "citation_patterns",
            "hypothesis_formulation",
            "experiment_design",
            "peer_review_patterns",
            "synthesis_techniques",
        ],
    )

    # Entity types
    schema.add_entity_type(
        name="paper",
        description="An academic paper or article",
        attributes=[
            "title",
            "authors",
            "year",
            "citations",
            "abstract",
            "venue",
            "doi",
        ],
    )
    schema.add_entity_type(
        name="hypothesis",
        description="A research hypothesis",
        attributes=[
            "statement",
            "confidence",
            "evidence_for",
            "evidence_against",
            "status",
        ],
    )
    schema.add_entity_type(
        name="experiment",
        description="An experiment or study",
        attributes=["method", "results", "conclusions", "status", "sample_size"],
    )
    schema.add_entity_type(
        name="dataset",
        description="A dataset used in research",
        attributes=["name", "size", "format", "source", "license"],
    )
    schema.add_entity_type(
        name="finding",
        description="A research finding or insight",
        attributes=["summary", "significance", "confidence", "supporting_evidence"],
    )

    # Relationships
    schema.add_relationship_type(
        name="cites",
        description="Paper cites another paper",
        source_type="paper",
        target_type="paper",
    )
    schema.add_relationship_type(
        name="tests",
        description="Experiment tests a hypothesis",
        source_type="experiment",
        target_type="hypothesis",
    )
    schema.add_relationship_type(
        name="uses",
        description="Experiment uses a dataset",
        source_type="experiment",
        target_type="dataset",
    )
    schema.add_relationship_type(
        name="supports",
        description="Finding supports a hypothesis",
        source_type="finding",
        target_type="hypothesis",
    )

    return schema


def get_sales_schema() -> DomainSchema:
    """
    Pre-built schema for sales workflows.

    Suitable for: Lead management, customer conversations, deal tracking.
    """
    schema = DomainSchema.create(
        name="sales",
        description="Memory schema for sales and customer engagement workflows",
        learning_categories=[
            "objection_handling",
            "closing_techniques",
            "qualification_patterns",
            "follow_up_timing",
            "value_proposition",
            "discovery_questions",
            "relationship_building",
            "negotiation_strategies",
        ],
    )

    # Entity types
    schema.add_entity_type(
        name="lead",
        description="A potential customer or prospect",
        attributes=["stage", "value", "next_action", "source", "company", "title"],
    )
    schema.add_entity_type(
        name="objection",
        description="A customer objection or concern",
        attributes=["type", "response", "outcome", "context"],
    )
    schema.add_entity_type(
        name="conversation",
        description="A customer interaction",
        attributes=["channel", "sentiment", "result", "summary", "follow_up"],
    )
    schema.add_entity_type(
        name="deal",
        description="A sales deal or opportunity",
        attributes=["stage", "value", "close_date", "probability", "stakeholders"],
    )
    schema.add_entity_type(
        name="product",
        description="A product or service being sold",
        attributes=["name", "price", "features", "competitors"],
    )

    # Relationships
    schema.add_relationship_type(
        name="converts_to",
        description="Lead converts to a deal",
        source_type="lead",
        target_type="deal",
    )
    schema.add_relationship_type(
        name="raised",
        description="Lead raised an objection",
        source_type="lead",
        target_type="objection",
    )
    schema.add_relationship_type(
        name="had",
        description="Lead had a conversation",
        source_type="lead",
        target_type="conversation",
    )
    schema.add_relationship_type(
        name="interested_in",
        description="Lead is interested in a product",
        source_type="lead",
        target_type="product",
    )

    return schema


def get_general_schema() -> DomainSchema:
    """
    Minimal schema for general-purpose agents.

    This is a flexible schema that can be extended for any domain.
    Suitable for: General assistants, tool-using agents, custom workflows.
    """
    schema = DomainSchema.create(
        name="general",
        description="Minimal, flexible schema for general-purpose agents",
        learning_categories=[
            "task_patterns",
            "error_recovery",
            "tool_usage",
            "efficiency_patterns",
            "user_preferences",
            "context_switching",
        ],
    )

    # Entity types (minimal but extensible)
    schema.add_entity_type(
        name="task",
        description="A unit of work to be completed",
        attributes=["title", "status", "priority", "category"],
    )
    schema.add_entity_type(
        name="resource",
        description="A resource used or created",
        attributes=["type", "path", "status", "metadata"],
    )
    schema.add_entity_type(
        name="goal",
        description="An objective or target",
        attributes=["description", "status", "deadline", "progress"],
    )
    schema.add_entity_type(
        name="context",
        description="A context or environment state",
        attributes=["name", "state", "active"],
    )

    # Relationships (minimal)
    schema.add_relationship_type(
        name="achieves",
        description="Task contributes to a goal",
        source_type="task",
        target_type="goal",
    )
    schema.add_relationship_type(
        name="uses",
        description="Task uses a resource",
        source_type="task",
        target_type="resource",
    )
    schema.add_relationship_type(
        name="requires",
        description="Task requires a context",
        source_type="task",
        target_type="context",
    )

    return schema


def get_customer_support_schema() -> DomainSchema:
    """
    Pre-built schema for customer support workflows.

    Suitable for: Ticket handling, escalation, knowledge base management.
    """
    schema = DomainSchema.create(
        name="customer_support",
        description="Memory schema for customer support workflows",
        learning_categories=[
            "issue_classification",
            "resolution_patterns",
            "escalation_criteria",
            "customer_sentiment",
            "knowledge_retrieval",
            "follow_up_patterns",
            "edge_case_handling",
        ],
    )

    # Entity types
    schema.add_entity_type(
        name="ticket",
        description="A customer support ticket",
        attributes=["status", "priority", "category", "customer_id", "resolution"],
    )
    schema.add_entity_type(
        name="article",
        description="A knowledge base article",
        attributes=["title", "content", "category", "views", "helpful_votes"],
    )
    schema.add_entity_type(
        name="customer",
        description="A customer profile",
        attributes=["tier", "history", "sentiment", "preferences"],
    )
    schema.add_entity_type(
        name="issue",
        description="A known issue or problem",
        attributes=["description", "status", "workaround", "affected_customers"],
    )

    # Relationships
    schema.add_relationship_type(
        name="resolves",
        description="Article resolves a ticket",
        source_type="article",
        target_type="ticket",
    )
    schema.add_relationship_type(
        name="submitted_by",
        description="Ticket submitted by customer",
        source_type="ticket",
        target_type="customer",
    )
    schema.add_relationship_type(
        name="related_to",
        description="Ticket related to a known issue",
        source_type="ticket",
        target_type="issue",
    )

    return schema


def get_content_creation_schema() -> DomainSchema:
    """
    Pre-built schema for content creation workflows.

    Suitable for: Blog writing, social media, marketing content.
    """
    schema = DomainSchema.create(
        name="content_creation",
        description="Memory schema for content creation workflows",
        learning_categories=[
            "writing_patterns",
            "engagement_optimization",
            "audience_targeting",
            "seo_strategies",
            "content_formatting",
            "voice_and_tone",
            "visual_content_patterns",
        ],
    )

    # Entity types
    schema.add_entity_type(
        name="content",
        description="A piece of content",
        attributes=["type", "title", "status", "platform", "performance_metrics"],
    )
    schema.add_entity_type(
        name="audience",
        description="A target audience segment",
        attributes=["name", "demographics", "interests", "pain_points"],
    )
    schema.add_entity_type(
        name="campaign",
        description="A content campaign",
        attributes=["name", "goal", "start_date", "end_date", "budget"],
    )
    schema.add_entity_type(
        name="template",
        description="A content template",
        attributes=["type", "structure", "usage_count", "effectiveness"],
    )

    # Relationships
    schema.add_relationship_type(
        name="targets",
        description="Content targets an audience",
        source_type="content",
        target_type="audience",
    )
    schema.add_relationship_type(
        name="part_of",
        description="Content is part of a campaign",
        source_type="content",
        target_type="campaign",
    )
    schema.add_relationship_type(
        name="uses",
        description="Content uses a template",
        source_type="content",
        target_type="template",
    )

    return schema
