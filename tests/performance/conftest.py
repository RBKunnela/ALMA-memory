"""
Performance Test Fixtures.

Provides fixtures optimized for benchmark testing.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def perf_storage_dir():
    """Create a dedicated storage directory for performance tests."""
    temp_dir = tempfile.mkdtemp(prefix="alma_perf_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def perf_project_id():
    """Project ID for performance tests."""
    return "perf-test-project"
