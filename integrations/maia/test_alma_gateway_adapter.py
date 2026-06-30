"""Unit tests for the Maia <-> ALMA gateway adapter.

These tests use a FAKE in-memory ALMA so they run WITHOUT the embeddings model
or any network (`sentence-transformers` need not be installed). They validate
the RPC contract surface, not ALMA's internals.

Run:
    python -m pytest integrations/maia/ -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from integrations.maia.alma_gateway_adapter import (
    handle_memory_query,
    handle_memory_store,
)

# --- Fake ALMA -------------------------------------------------------------


@dataclass
class _FakeFact:
    fact: str


@dataclass
class _FakeSlice:
    """Mimics alma.types.MemorySlice (only the fields the adapter reads)."""

    domain_knowledge: List[_FakeFact] = field(default_factory=list)
    heuristics: List[Any] = field(default_factory=list)
    preferences: List[Any] = field(default_factory=list)
    anti_patterns: List[Any] = field(default_factory=list)


class FakeALMA:
    """In-memory stand-in for alma.core.ALMA, matching the real signatures."""

    def __init__(self) -> None:
        self.facts: List[Dict[str, Any]] = []

    def retrieve(
        self,
        task: str,
        agent: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> _FakeSlice:
        matches = [
            _FakeFact(fact=f["fact"])
            for f in self.facts
            if f["agent"] == agent
        ][:top_k]
        return _FakeSlice(domain_knowledge=matches)

    def add_domain_knowledge(
        self,
        agent: str,
        domain: str,
        fact: str,
        source: str = "user_stated",
    ) -> None:
        self.facts.append(
            {"agent": agent, "domain": domain, "fact": fact, "source": source}
        )


class ExplodingALMA:
    """Raises on every call — proves handlers never leak exceptions."""

    def retrieve(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("boom")

    def add_domain_knowledge(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("boom")


# --- query -----------------------------------------------------------------


def test_query_returns_contract_shape():
    alma = FakeALMA()
    alma.add_domain_knowledge(agent="maia", domain="d", fact="sky is blue")
    out = handle_memory_query({"query": "sky", "limit": 5}, alma=alma)
    assert out["ok"] is True
    assert out["payload"]["results"] == [{"content": "sky is blue"}]


def test_query_empty_memory_is_success_not_error():
    out = handle_memory_query({"query": "anything"}, alma=FakeALMA())
    assert out == {"ok": True, "payload": {"results": []}}


def test_query_respects_limit():
    alma = FakeALMA()
    for i in range(5):
        alma.add_domain_knowledge(agent="maia", domain="d", fact=f"fact {i}")
    out = handle_memory_query({"query": "x", "limit": 2}, alma=alma)
    assert len(out["payload"]["results"]) == 2


def test_query_persona_routes_to_agent_scope():
    alma = FakeALMA()
    alma.add_domain_knowledge(agent="alice", domain="d", fact="alice fact")
    alma.add_domain_knowledge(agent="maia", domain="d", fact="maia fact")
    out = handle_memory_query(
        {"query": "x", "metadata": {"persona": "alice"}}, alma=alma
    )
    assert out["payload"]["results"] == [{"content": "alice fact"}]


def test_query_bad_params_returns_not_ok():
    assert handle_memory_query({}, alma=FakeALMA())["ok"] is False
    assert handle_memory_query({"query": ""}, alma=FakeALMA())["ok"] is False
    assert handle_memory_query({"query": 123}, alma=FakeALMA())["ok"] is False


def test_query_never_raises_on_backend_failure():
    out = handle_memory_query({"query": "x"}, alma=ExplodingALMA())
    assert out["ok"] is False
    assert "error" in out


# --- store -----------------------------------------------------------------


def test_store_ok_and_persists():
    alma = FakeALMA()
    out = handle_memory_store({"content": "remember this", "metadata": {}}, alma=alma)
    assert out == {"ok": True}
    assert alma.facts[0]["fact"] == "remember this"
    assert alma.facts[0]["agent"] == "maia"


def test_store_persona_selects_agent():
    alma = FakeALMA()
    handle_memory_store(
        {"content": "x", "metadata": {"persona": "bob"}}, alma=alma
    )
    assert alma.facts[0]["agent"] == "bob"


def test_store_bad_params_returns_not_ok():
    alma = FakeALMA()
    assert handle_memory_store({}, alma=alma)["ok"] is False
    assert handle_memory_store({"content": ""}, alma=alma)["ok"] is False
    assert handle_memory_store({"content": "ok", "metadata": "nope"}, alma=alma)[
        "ok"
    ] is False
    assert alma.facts == []  # nothing persisted on bad params


def test_store_never_raises_on_backend_failure():
    out = handle_memory_store({"content": "x"}, alma=ExplodingALMA())
    assert out["ok"] is False
    assert "error" in out


# --- round trip ------------------------------------------------------------


def test_store_then_query_round_trip():
    alma = FakeALMA()
    handle_memory_store({"content": "the eiffel tower is in paris"}, alma=alma)
    out = handle_memory_query({"query": "eiffel", "limit": 10}, alma=alma)
    assert {"content": "the eiffel tower is in paris"} in out["payload"]["results"]
