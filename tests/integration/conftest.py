"""
Integration Test Fixtures.

Specific fixtures for integration testing with real storage backends.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Load task definitions for parameterized tests
@pytest.fixture
def helena_tasks() -> List[Dict[str, Any]]:
    """Load Helena task definitions."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    with open(fixtures_dir / "helena_tasks.json") as f:
        return json.load(f)


@pytest.fixture
def victor_tasks() -> List[Dict[str, Any]]:
    """Load Victor task definitions."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    with open(fixtures_dir / "victor_tasks.json") as f:
        return json.load(f)


@pytest.fixture
def integration_project_id() -> str:
    """Project ID for integration tests."""
    return "integration-test-project"
