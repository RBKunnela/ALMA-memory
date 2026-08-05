"""
Tests for Agent Memory Atlas gaps (Chefe 561 / Code-Hub 1624).

G1 persist verification · G2 write guard · G4 forget audit · G3 schema columns
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from alma.exceptions import ScopeViolationError
from alma.learning.write_guard import (
    check_write_guard,
    text_matches_anti_pattern,
    write_guard_enabled,
)
from alma.retrieval.verification import (
    Verification,
    VerificationMethod,
    VerificationStatus,
    VerifiedRetriever,
)
from alma.storage.sqlite_local import SQLiteStorage
from alma.types import AntiPattern, Outcome


@pytest.fixture()
def storage(tmp_path: Path):
    db = tmp_path / "alma_atlas.db"
    return SQLiteStorage(str(db))


def test_write_guard_env_default_on(monkeypatch):
    monkeypatch.delenv("ALMA_ANTI_PATTERN_WRITE_GUARD", raising=False)
    assert write_guard_enabled() is True
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD", "0")
    assert write_guard_enabled() is False


def test_text_matches_anti_pattern_substring():
    assert text_matches_anti_pattern(
        "deploy with rolling updates to production",
        "rolling updates",
    )


def test_write_guard_blocks_matching_learn(storage, monkeypatch):
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD", "1")
    from datetime import datetime, timezone

    storage.save_anti_pattern(
        AntiPattern(
            id="ap_roll",
            agent="dev",
            project_id="p1",
            pattern="rolling updates",
            why_bad="caused incidents",
            better_alternative="blue-green",
            occurrence_count=2,
            last_seen=datetime.now(timezone.utc),
        )
    )
    result = check_write_guard(
        storage,
        project_id="p1",
        agent="dev",
        texts=["Deploy service using rolling updates"],
    )
    assert result.blocked is True
    assert result.anti_pattern_id == "ap_roll"


def test_write_guard_allows_unrelated(storage, monkeypatch):
    monkeypatch.setenv("ALMA_ANTI_PATTERN_WRITE_GUARD", "1")
    from datetime import datetime, timezone

    storage.save_anti_pattern(
        AntiPattern(
            id="ap_roll",
            agent="dev",
            project_id="p1",
            pattern="rolling updates",
            why_bad="bad",
            better_alternative="blue-green",
            occurrence_count=1,
            last_seen=datetime.now(timezone.utc),
        )
    )
    result = check_write_guard(
        storage,
        project_id="p1",
        agent="dev",
        texts=["Use blue-green deployment with health checks"],
    )
    assert result.blocked is False


def test_persist_verification_on_outcome(storage):
    o = Outcome(
        id="out_test1",
        agent="dev",
        project_id="p1",
        task_type="deploy",
        task_description="deploy auth",
        success=True,
        strategy_used="blue-green",
    )
    storage.save_outcome(o)
    v = Verification(
        status=VerificationStatus.CONTRADICTED,
        confidence=0.2,
        reason="conflicts with known incident log",
        method=VerificationMethod.CONFIDENCE,
        contradicting_source="out_old",
    )
    from alma.storage.verification_store import persist_verification

    assert persist_verification(storage, o, v) is True
    rows = storage.list_by_verification_status(
        project_id="p1",
        verification_status="contradicted",
        memory_type="outcomes",
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "out_test1"
    assert rows[0]["verification_status"] == "contradicted"


def test_forget_audit_recorded(storage):
    o = Outcome(
        id="out_del1",
        agent="dev",
        project_id="p1",
        task_type="t",
        task_description="x",
        success=False,
        strategy_used="y",
    )
    storage.save_outcome(o)
    aid = storage.record_forget_audit(
        project_id="p1",
        memory_type="outcome",
        memory_id="out_del1",
        agent="dev",
        reason="stale",
        strategy="age_prune",
    )
    assert aid.startswith("fga_")
    import sqlite3

    with storage._get_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM alma_forget_audit").fetchone()[0]
    assert n == 1


def test_verified_retriever_persists(storage, monkeypatch):
    """VerifiedRetriever writes verification when storage is wired."""
    o = Outcome(
        id="out_vr1",
        agent="dev",
        project_id="p1",
        task_type="t",
        task_description="rate limits",
        success=True,
        strategy_used="retry with backoff",
    )
    storage.save_outcome(o)

    class FakeEngine:
        def retrieve(self, query, **kwargs):
            class Slice:
                heuristics = []
                outcomes = [o]
                knowledge = []
                anti_patterns = []
                preferences = []

            return Slice()

    retriever = VerifiedRetriever(
        retrieval_engine=FakeEngine(),
        storage=storage,
        persist_verification=True,
    )
    results = retriever.retrieve_verified(
        query="rate limits",
        agent="dev",
        project_id="p1",
        top_k=5,
    )
    assert results.total_count >= 1
    rows = storage.list_by_verification_status(
        project_id="p1",
        verification_status=results.verified[0].status.value
        if results.verified
        else (
            results.uncertain[0].status.value
            if results.uncertain
            else results.unverifiable[0].status.value
            if results.unverifiable
            else results.contradicted[0].status.value
        ),
        memory_type="outcomes",
    )
    assert any(r["id"] == "out_vr1" for r in rows)


def test_schema_parity_sqlite_columns(storage):
    """G3: verification columns exist on all memory tables."""
    import sqlite3

    with storage._get_connection() as conn:
        for table in (
            "heuristics",
            "outcomes",
            "domain_knowledge",
            "preferences",
            "anti_patterns",
        ):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            assert "verification_status" in cols, table
            assert "verified_at" in cols, table
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "alma_forget_audit" in names
