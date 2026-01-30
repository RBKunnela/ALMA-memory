"""
ALMA Schema Migrations - Version Definitions.

Each version file contains migrations for that schema version.
Migrations are automatically registered when imported.
"""

# Import all version modules to register migrations
from alma.storage.migrations.versions import v1_0_0
from alma.storage.migrations.versions import v1_1_0_workflow_context

__all__ = ["v1_0_0", "v1_1_0_workflow_context"]
