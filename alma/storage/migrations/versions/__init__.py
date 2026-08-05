"""
ALMA Schema Migrations - Version Definitions.

Each version file contains migrations for that schema version.
Migrations are automatically registered when imported.
"""

# Import all version modules to register migrations
from alma.storage.migrations.versions import (
    v1_0_0,
    v1_1_0_workflow_context,
    v1_2_0_atlas_gaps,
)

__all__ = ["v1_0_0", "v1_1_0_workflow_context", "v1_2_0_atlas_gaps"]
