"""
ALMA Domain Memory Module.

Provides domain-agnostic memory schemas and factory pattern
for creating domain-specific ALMA instances.
"""

from alma.domains.types import (
    DomainSchema,
    EntityType,
    RelationshipType,
)
from alma.domains.factory import DomainMemoryFactory
from alma.domains.schemas import (
    get_coding_schema,
    get_research_schema,
    get_sales_schema,
    get_general_schema,
)

__all__ = [
    "DomainSchema",
    "EntityType",
    "RelationshipType",
    "DomainMemoryFactory",
    "get_coding_schema",
    "get_research_schema",
    "get_sales_schema",
    "get_general_schema",
]
