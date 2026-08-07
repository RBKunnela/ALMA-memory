"""Security close-out tests (Chefe 1770 / Sentinel ALMA dogfood)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alma.learning.write_guard import check_write_guard, write_guard_fail_closed
from alma.mcp.server import ALMAMCPServer


def test_fail_closed_default_on(monkeypatch):
    monkeypatch.delenv("ALMA_ANTI_PATTERN_WRITE_GUARD_FAIL_CLOSED", raising=False)
    assert write_guard_fail_closed() is True


def test_fail_closed_blocks_missing_getter(monkeypatch):
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD", "1")
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD_FAIL_CLOSED", "1")
    storage = MagicMock(spec=[])  # no get_anti_patterns
    r = check_write_guard(storage, "p1", "a", ["anything"])
    assert r.blocked is True
    assert "fail_closed" in r.reason


def test_fail_open_legacy(monkeypatch):
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD", "1")
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD_FAIL_CLOSED", "0")
    storage = MagicMock(spec=[])
    r = check_write_guard(storage, "p1", "a", ["anything"])
    assert r.blocked is False


@pytest.mark.asyncio
async def test_http_refuses_bind_all_without_token():
    alma = MagicMock()
    server = ALMAMCPServer(alma=alma)
    with pytest.raises(ValueError, match="ALMA_MCP_TOKEN"):
        await server.run_http(host="0.0.0.0", port=18765, auth_token=None)


def test_headers_file_present():
    p = Path(__file__).resolve().parents[2] / "site-docs" / "_headers"
    assert p.is_file()
    text = p.read_text()
    assert "Strict-Transport-Security" in text
    assert "Content-Security-Policy" in text
    assert "X-Frame-Options" in text
