"""
ALMA Test Fixtures.

Pre-defined test data for integration and E2E tests.

Usage:
    from tests.fixtures import (
        HELENA_TASKS,
        VICTOR_TASKS,
        seed_helena_memories,
        seed_victor_memories,
    )
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# Re-export seeding functions (at top to satisfy E402)
from tests.fixtures.seed_memories import (
    create_failure_pattern,
    create_learning_progression,
    seed_all_memories,
    seed_helena_memories,
    seed_victor_memories,
)

# Load task definitions from JSON files
_fixtures_dir = Path(__file__).parent


def _load_json(filename: str) -> List[Dict[str, Any]]:
    """Load a JSON fixture file."""
    file_path = _fixtures_dir / filename
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return []


# Pre-loaded task definitions
HELENA_TASKS: List[Dict[str, Any]] = _load_json("helena_tasks.json")
VICTOR_TASKS: List[Dict[str, Any]] = _load_json("victor_tasks.json")


__all__ = [
    "HELENA_TASKS",
    "VICTOR_TASKS",
    "seed_helena_memories",
    "seed_victor_memories",
    "seed_all_memories",
    "create_learning_progression",
    "create_failure_pattern",
]
