"""
E2E Test Fixtures.

Fixtures specific to end-to-end testing scenarios.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def e2e_storage_dir():
    """Create a dedicated storage directory for E2E tests."""
    temp_dir = tempfile.mkdtemp(prefix="alma_e2e_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def e2e_project_id():
    """Project ID for E2E tests."""
    return "e2e-test-project"
