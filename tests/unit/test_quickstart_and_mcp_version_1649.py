"""Regression: quickstart classmethod + MCP version from package (Chefe 1649).

Replaces stale PR #33 value without carrying the old branch.
"""

from __future__ import annotations

import inspect

from alma import __version__
from alma.core import ALMA
from alma.mcp.server import ALMAMCPServer


def test_quickstart_is_single_classmethod_and_callable() -> None:
    """Double @classmethod made quickstart unusable on some Python versions."""
    assert callable(ALMA.quickstart)
    # Source must not stack two @classmethod decorators.
    src = inspect.getsource(ALMA.quickstart)
    assert src.count("@classmethod") == 1


def test_mcp_server_version_defaults_to_package_version() -> None:
    """MCP server_version must track alma.__version__, not a hardcoded 0.6.0."""
    param = inspect.signature(ALMAMCPServer.__init__).parameters["server_version"]
    assert param.default == __version__
    assert param.default != "0.6.0"
