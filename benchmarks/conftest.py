"""
Shared fixtures for ALMA benchmarks.

Provides common test infrastructure for benchmark suites.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def benchmark_tmp_dir():
    """Create a temporary directory for benchmark data and artifacts."""
    with tempfile.TemporaryDirectory(prefix="alma_bench_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def benchmark_data_dir():
    """
    Return the benchmark data directory.

    Uses ALMA_BENCHMARK_DATA env var if set, otherwise defaults to
    /tmp/alma-benchmark-data.
    """
    data_dir = Path(os.environ.get("ALMA_BENCHMARK_DATA", "/tmp/alma-benchmark-data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
