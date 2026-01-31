"""
ALMA Storage Constants.

Canonical naming conventions for memory types across all storage backends.
This ensures consistency for:
- Data migration between backends
- Backend-agnostic code
- Documentation consistency
"""

from typing import Dict


class MemoryType:
    """
    Canonical memory type identifiers.

    These are the internal names used consistently across all backends.
    Each backend may add a prefix or transform these for their specific
    storage format, but the canonical names remain constant.
    """

    HEURISTICS = "heuristics"
    OUTCOMES = "outcomes"
    PREFERENCES = "preferences"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    ANTI_PATTERNS = "anti_patterns"

    # All memory types as a tuple for iteration
    ALL = (
        HEURISTICS,
        OUTCOMES,
        PREFERENCES,
        DOMAIN_KNOWLEDGE,
        ANTI_PATTERNS,
    )

    # Memory types that support embeddings/vector search
    VECTOR_ENABLED = (
        HEURISTICS,
        OUTCOMES,
        DOMAIN_KNOWLEDGE,
        ANTI_PATTERNS,
    )


def get_table_name(memory_type: str, prefix: str = "") -> str:
    """
    Get the table/container name for a memory type.

    Args:
        memory_type: One of the MemoryType constants
        prefix: Optional prefix to add (e.g., "alma_" for PostgreSQL)

    Returns:
        The formatted table/container name

    Example:
        >>> get_table_name(MemoryType.HEURISTICS, "alma_")
        'alma_heuristics'
        >>> get_table_name(MemoryType.DOMAIN_KNOWLEDGE)
        'domain_knowledge'
    """
    if memory_type not in MemoryType.ALL:
        raise ValueError(f"Unknown memory type: {memory_type}")
    return f"{prefix}{memory_type}"


def get_table_names(prefix: str = "") -> Dict[str, str]:
    """
    Get all table/container names with an optional prefix.

    Args:
        prefix: Optional prefix to add (e.g., "alma_" for PostgreSQL)

    Returns:
        Dict mapping canonical memory type to table/container name

    Example:
        >>> get_table_names("alma_")
        {
            'heuristics': 'alma_heuristics',
            'outcomes': 'alma_outcomes',
            'preferences': 'alma_preferences',
            'domain_knowledge': 'alma_domain_knowledge',
            'anti_patterns': 'alma_anti_patterns',
        }
    """
    return {mt: get_table_name(mt, prefix) for mt in MemoryType.ALL}


# Pre-computed table name mappings for each backend
# These are the canonical mappings that should be used

# PostgreSQL uses alma_ prefix with underscores
POSTGRESQL_TABLE_NAMES = get_table_names("alma_")

# SQLite uses no prefix (local file-based, no collision risk)
SQLITE_TABLE_NAMES = get_table_names("")

# Azure Cosmos uses alma_ prefix with underscores (standardized)
# Note: Previously used hyphens, now standardized to underscores
AZURE_COSMOS_CONTAINER_NAMES = get_table_names("alma_")
