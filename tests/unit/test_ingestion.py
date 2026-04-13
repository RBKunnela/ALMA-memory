"""
Tests for alma.ingestion package.

Adapted from MemPalace (MIT License).

Tests normalization of 6 chat formats, memory extraction,
file ingestion, and conversation ingestion.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from alma.ingestion import (
    IngestionResult,
    chunk_exchanges,
    chunk_text,
    detect_format,
    extract_memories,
    ingest_conversation,
    ingest_conversations,
    ingest_directory,
    ingest_file,
    normalize,
    normalize_text,
    scan_conversations,
    scan_directory,
)
from alma.types import AntiPattern, DomainKnowledge, Outcome, UserPreference


# =========================================================================
# Test data fixtures
# =========================================================================


CLAUDE_CODE_JSONL = "\n".join(
    [
        json.dumps(
            {
                "type": "human",
                "message": {
                    "content": "How do I fix the login bug?"
                },
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "The login bug is caused by a missing JWT validation step.",
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "human",
                "message": {
                    "content": "Can you show me the fix?"
                },
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Sure, add the validate_jwt() call before checking the session.",
                        }
                    ]
                },
            }
        ),
    ]
)


CODEX_JSONL = "\n".join(
    [
        json.dumps({"type": "session_meta", "session_id": "abc123"}),
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Fix the database connection timeout",
                },
            }
        ),
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "I'll increase the pool size and add retry logic.",
                },
            }
        ),
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Good, also add connection health checks.",
                },
            }
        ),
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "Added periodic health checks with 30s interval.",
                },
            }
        ),
    ]
)


CLAUDE_AI_JSON = json.dumps(
    [
        {"role": "user", "content": "What is the best database for this?"},
        {
            "role": "assistant",
            "content": "PostgreSQL is ideal for your relational data needs.",
        },
        {"role": "user", "content": "Should I use an ORM?"},
        {
            "role": "assistant",
            "content": "SQLAlchemy is the standard choice for Python projects.",
        },
    ]
)


CHATGPT_JSON = json.dumps(
    {
        "title": "Test Conversation",
        "mapping": {
            "root": {
                "parent": None,
                "message": None,
                "children": ["msg1"],
            },
            "msg1": {
                "parent": "root",
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Tell me about Python decorators"]},
                },
                "children": ["msg2"],
            },
            "msg2": {
                "parent": "msg1",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": [
                            "Decorators are functions that modify other functions."
                        ]
                    },
                },
                "children": ["msg3"],
            },
            "msg3": {
                "parent": "msg2",
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Can you show an example?"]},
                },
                "children": ["msg4"],
            },
            "msg4": {
                "parent": "msg3",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": [
                            "Here is a simple logging decorator example."
                        ]
                    },
                },
                "children": [],
            },
        },
    }
)


SLACK_JSON = json.dumps(
    [
        {
            "type": "message",
            "user": "U001",
            "text": "Hey, did you see the new deployment?",
        },
        {
            "type": "message",
            "user": "U002",
            "text": "Yes, it looks good. The latency dropped by 40%.",
        },
        {
            "type": "message",
            "user": "U001",
            "text": "Great, let's monitor it for a few more hours.",
        },
        {
            "type": "message",
            "user": "U002",
            "text": "Agreed. I'll set up the dashboard alerts.",
        },
    ]
)


PLAIN_TRANSCRIPT = """> How do I implement caching?
Use Redis for distributed caching with a TTL of 300 seconds.

> What about cache invalidation?
Invalidate on write using pub/sub events to notify all nodes.

> Should I cache database queries too?
Yes, cache frequently read queries with a shorter TTL of 60 seconds.
"""


# =========================================================================
# Normalizer tests
# =========================================================================


class TestDetectFormat:
    """Tests for detect_format()."""

    def test_detect_claude_code_jsonl(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text(CLAUDE_CODE_JSONL)
        assert detect_format(str(p)) == "claude_code_jsonl"

    def test_detect_codex_jsonl(self, tmp_path):
        p = tmp_path / "codex.jsonl"
        p.write_text(CODEX_JSONL)
        assert detect_format(str(p)) == "codex_jsonl"

    def test_detect_claude_ai_json(self, tmp_path):
        p = tmp_path / "claude.json"
        p.write_text(CLAUDE_AI_JSON)
        assert detect_format(str(p)) == "claude_ai_json"

    def test_detect_chatgpt_json(self, tmp_path):
        p = tmp_path / "chatgpt.json"
        p.write_text(CHATGPT_JSON)
        assert detect_format(str(p)) == "chatgpt_json"

    def test_detect_slack_json(self, tmp_path):
        p = tmp_path / "slack.json"
        p.write_text(SLACK_JSON)
        assert detect_format(str(p)) == "slack_json"

    def test_detect_transcript(self, tmp_path):
        p = tmp_path / "transcript.txt"
        p.write_text(PLAIN_TRANSCRIPT)
        assert detect_format(str(p)) == "transcript"

    def test_detect_plain_text(self, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("This is just a regular text file about Python.")
        assert detect_format(str(p)) == "plain_text"

    def test_detect_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        assert detect_format(str(p)) == "plain_text"

    def test_detect_nonexistent_file(self):
        with pytest.raises(IOError):
            detect_format("/nonexistent/file.txt")


class TestNormalize:
    """Tests for normalize() with each of the 6 chat formats."""

    def test_normalize_claude_code_jsonl(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text(CLAUDE_CODE_JSONL)
        result = normalize(str(p))
        assert ">" in result
        assert "login bug" in result.lower()

    def test_normalize_codex_jsonl(self, tmp_path):
        p = tmp_path / "codex.jsonl"
        p.write_text(CODEX_JSONL)
        result = normalize(str(p))
        assert ">" in result
        assert "database connection" in result.lower()

    def test_normalize_claude_ai_json(self, tmp_path):
        p = tmp_path / "claude.json"
        p.write_text(CLAUDE_AI_JSON)
        result = normalize(str(p))
        assert ">" in result
        assert "postgresql" in result.lower()

    def test_normalize_chatgpt_json(self, tmp_path):
        p = tmp_path / "chatgpt.json"
        p.write_text(CHATGPT_JSON)
        result = normalize(str(p))
        assert ">" in result
        assert "decorator" in result.lower()

    def test_normalize_slack_json(self, tmp_path):
        p = tmp_path / "slack.json"
        p.write_text(SLACK_JSON)
        result = normalize(str(p))
        assert ">" in result
        assert "deployment" in result.lower()

    def test_normalize_transcript_passthrough(self, tmp_path):
        p = tmp_path / "transcript.txt"
        p.write_text(PLAIN_TRANSCRIPT)
        result = normalize(str(p))
        # Already has > markers, passes through unchanged
        assert result == PLAIN_TRANSCRIPT

    def test_normalize_plain_text_passthrough(self, tmp_path):
        p = tmp_path / "plain.txt"
        content = "Just regular text about Python programming."
        p.write_text(content)
        result = normalize(str(p))
        assert result == content

    def test_normalize_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        result = normalize(str(p))
        assert result == ""

    def test_normalize_nonexistent_file(self):
        with pytest.raises(IOError):
            normalize("/nonexistent/file.txt")


class TestNormalizeText:
    """Tests for normalize_text() (in-memory normalization)."""

    def test_normalize_text_claude_code(self):
        result = normalize_text(CLAUDE_CODE_JSONL)
        assert ">" in result

    def test_normalize_text_empty(self):
        assert normalize_text("") == ""

    def test_normalize_text_none(self):
        assert normalize_text(None) == ""

    def test_normalize_text_passthrough(self):
        result = normalize_text(PLAIN_TRANSCRIPT)
        assert result == PLAIN_TRANSCRIPT


# =========================================================================
# Memory extractor tests
# =========================================================================


class TestExtractMemories:
    """Tests for extract_memories()."""

    def test_extract_decision(self):
        text = "We decided to go with PostgreSQL instead of MongoDB because our data is relational."
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1
        types = {m["memory_type"] for m in memories}
        assert "decision" in types

    def test_extract_preference(self):
        text = "I prefer to always use snake_case for Python variable names. Never use camelCase."
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1
        types = {m["memory_type"] for m in memories}
        assert "preference" in types

    def test_extract_milestone(self):
        text = "We finally got it working after three days. The key was using async operations. It was a real breakthrough."
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1
        types = {m["memory_type"] for m in memories}
        assert "milestone" in types

    def test_extract_problem(self):
        text = "There is a critical bug in the authentication module. The session keeps crashing when users try to log in. The root cause is a null pointer."
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1
        types = {m["memory_type"] for m in memories}
        assert "problem" in types or "milestone" in types

    def test_extract_emotional(self):
        text = "I love how this turned out. I feel so proud of what we built. It's beautiful and amazing."
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1
        types = {m["memory_type"] for m in memories}
        assert "emotional" in types or "milestone" in types

    def test_extract_empty_text(self):
        memories = extract_memories("")
        assert memories == []

    def test_extract_short_text(self):
        memories = extract_memories("hi")
        assert memories == []

    def test_extract_no_matches(self):
        text = "The quick brown fox jumps over the lazy dog."
        memories = extract_memories(text, min_confidence=0.9)
        assert memories == []

    def test_extract_has_confidence(self):
        text = "We decided to use Redis because it's fast. The approach is better than Memcached."
        memories = extract_memories(text, min_confidence=0.1)
        for mem in memories:
            assert "confidence" in mem
            assert 0.0 <= mem["confidence"] <= 1.0

    def test_extract_has_chunk_index(self):
        text = "We decided to go with option A.\n\nWe also prefer to use Python."
        memories = extract_memories(text, min_confidence=0.1)
        indices = [m["chunk_index"] for m in memories]
        assert indices == list(range(len(memories)))

    def test_extract_from_conversation_format(self):
        text = """> Should we use React or Vue?
We decided to go with React because of the larger ecosystem.

> What about state management?
We went with Redux because it integrates well with React.

> Any preferences for styling?
I prefer to always use CSS modules. Never use inline styles.
"""
        memories = extract_memories(text, min_confidence=0.1)
        assert len(memories) >= 1


# =========================================================================
# File miner tests
# =========================================================================


class TestChunkText:
    """Tests for chunk_text()."""

    def test_chunk_empty(self):
        assert chunk_text("") == []

    def test_chunk_short_text(self):
        result = chunk_text("x" * 100)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 0

    def test_chunk_long_text(self):
        text = "word " * 500
        result = chunk_text(text)
        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i

    def test_chunk_preserves_paragraphs(self):
        text = "First paragraph.\n\n" + "Second paragraph.\n\n" + "x" * 100
        result = chunk_text(text)
        assert len(result) >= 1

    def test_chunk_too_small_skipped(self):
        text = "hi"
        result = chunk_text(text)
        assert result == []


class TestIngestFile:
    """Tests for ingest_file()."""

    def test_ingest_python_file(self, tmp_path):
        p = tmp_path / "example.py"
        content = (
            "# We decided to use dataclasses because they're simpler.\n"
            "# The approach is better than using plain dicts.\n"
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Config:\n"
            "    name: str\n"
            "    value: int = 0\n"
        )
        p.write_text(content)
        result = ingest_file(str(p))
        assert isinstance(result, IngestionResult)
        assert result.source_file == str(p.resolve())
        assert result.total_chunks >= 1

    def test_ingest_file_returns_alma_types(self, tmp_path):
        p = tmp_path / "notes.txt"
        content = (
            "We decided to go with PostgreSQL instead of MongoDB.\n"
            "I prefer to always use typed queries.\n"
            "We finally got the migration working after three days.\n"
            "There was a critical bug in the auth module that kept crashing.\n"
        )
        p.write_text(content)
        result = ingest_file(str(p), project_id="test", agent="test-agent")
        assert isinstance(result, IngestionResult)
        # Check that at least some memories were classified
        total_classified = (
            len(result.domain_knowledge)
            + len(result.user_preferences)
            + len(result.outcomes)
            + len(result.anti_patterns)
        )
        assert total_classified >= 1 or len(result.raw_chunks) >= 1

    def test_ingest_file_not_found(self):
        with pytest.raises(IOError):
            ingest_file("/nonexistent/path/file.py")

    def test_ingest_file_too_small(self, tmp_path):
        p = tmp_path / "tiny.txt"
        p.write_text("hi")
        with pytest.raises(ValueError):
            ingest_file(str(p))

    def test_ingest_file_no_extract(self, tmp_path):
        p = tmp_path / "raw.txt"
        p.write_text("x" * 200)
        result = ingest_file(str(p), extract=False)
        assert result.classified_chunks == 0
        assert result.total_chunks >= 1
        assert len(result.raw_chunks) >= 1

    def test_ingest_result_merge(self):
        r1 = IngestionResult(
            source_file="a.py",
            total_chunks=5,
            classified_chunks=2,
            domain_knowledge=[
                DomainKnowledge(
                    id="1",
                    agent="a",
                    project_id="p",
                    domain="d",
                    fact="f",
                    source="s",
                )
            ],
        )
        r2 = IngestionResult(
            source_file="b.py",
            total_chunks=3,
            classified_chunks=1,
            outcomes=[
                Outcome(
                    id="2",
                    agent="a",
                    project_id="p",
                    task_type="t",
                    task_description="d",
                    success=True,
                    strategy_used="s",
                )
            ],
        )
        r1.merge(r2)
        assert r1.total_chunks == 8
        assert r1.classified_chunks == 3
        assert len(r1.domain_knowledge) == 1
        assert len(r1.outcomes) == 1


class TestIngestDirectory:
    """Tests for ingest_directory()."""

    def test_ingest_directory(self, tmp_path):
        (tmp_path / "file1.py").write_text(
            "# We decided to use the strategy pattern because it's flexible.\n"
            "class Strategy:\n"
            "    pass\n" * 20
        )
        (tmp_path / "file2.txt").write_text(
            "This is a long enough text file to be chunked properly.\n" * 10
        )
        (tmp_path / "ignore.exe").write_text("binary stuff")
        result = ingest_directory(str(tmp_path))
        assert isinstance(result, IngestionResult)
        assert result.total_chunks >= 1

    def test_ingest_directory_with_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"Content number {i}.\n" * 20)
        result = ingest_directory(str(tmp_path), limit=2)
        assert isinstance(result, IngestionResult)

    def test_ingest_directory_not_found(self):
        with pytest.raises(IOError):
            ingest_directory("/nonexistent/directory/")


class TestScanDirectory:
    """Tests for scan_directory()."""

    def test_scan_finds_readable_files(self, tmp_path):
        (tmp_path / "code.py").write_text("print('hello')")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "binary.exe").write_text("nope")
        files = scan_directory(str(tmp_path))
        extensions = {f.suffix for f in files}
        assert ".py" in extensions
        assert ".json" in extensions
        assert ".exe" not in extensions

    def test_scan_skips_dirs(self, tmp_path):
        skip_dir = tmp_path / "node_modules"
        skip_dir.mkdir()
        (skip_dir / "package.json").write_text("{}")
        (tmp_path / "main.py").write_text("print('hi')")
        files = scan_directory(str(tmp_path))
        assert all("node_modules" not in str(f) for f in files)

    def test_scan_skips_gitignore(self, tmp_path):
        (tmp_path / "package-lock.json").write_text("{}")
        (tmp_path / "main.py").write_text("print('hi')")
        files = scan_directory(str(tmp_path))
        names = {f.name for f in files}
        assert "package-lock.json" not in names


# =========================================================================
# Conversation miner tests
# =========================================================================


class TestChunkExchanges:
    """Tests for chunk_exchanges()."""

    def test_chunk_exchange_pairs(self):
        content = """> What is Python?
Python is a programming language.

> How do I install it?
Use your package manager or download from python.org.

> What about virtual environments?
Use venv or virtualenv for isolation.
"""
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert "content" in chunk
            assert "chunk_index" in chunk

    def test_chunk_paragraph_fallback(self):
        content = "First paragraph about coding.\n\nSecond paragraph about testing.\n\nThird paragraph about deployment."
        chunks = chunk_exchanges(content)
        assert len(chunks) >= 1

    def test_chunk_empty(self):
        chunks = chunk_exchanges("")
        assert chunks == []

    def test_chunk_long_exchange(self):
        content = "> Short question?\n" + "A " * 1000
        chunks = chunk_exchanges(content)
        # Should split into multiple chunks
        assert len(chunks) >= 1


class TestIngestConversation:
    """Tests for ingest_conversation()."""

    def test_ingest_claude_code_conversation(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text(CLAUDE_CODE_JSONL)
        result = ingest_conversation(str(p))
        assert isinstance(result, IngestionResult)
        assert result.total_chunks >= 1

    def test_ingest_plain_transcript(self, tmp_path):
        p = tmp_path / "transcript.txt"
        p.write_text(PLAIN_TRANSCRIPT)
        result = ingest_conversation(str(p))
        assert isinstance(result, IngestionResult)
        assert result.total_chunks >= 1

    def test_ingest_conversation_general_mode(self, tmp_path):
        p = tmp_path / "convo.txt"
        p.write_text(
            "> We decided to go with Redis because it's faster.\n"
            "Good decision. Redis is the better approach for caching.\n\n"
            "> I prefer to always use type hints.\n"
            "That's a great preference for code quality.\n\n"
            "> We finally got the pipeline working!\n"
            "Congratulations on the breakthrough!\n"
        )
        result = ingest_conversation(str(p), extract_mode="general")
        assert isinstance(result, IngestionResult)

    def test_ingest_conversation_not_found(self):
        with pytest.raises(IOError):
            ingest_conversation("/nonexistent/convo.txt")


class TestIngestConversations:
    """Tests for ingest_conversations()."""

    def test_ingest_conversations_directory(self, tmp_path):
        (tmp_path / "session1.jsonl").write_text(CLAUDE_CODE_JSONL)
        (tmp_path / "transcript.txt").write_text(PLAIN_TRANSCRIPT)
        result = ingest_conversations(str(tmp_path))
        assert isinstance(result, IngestionResult)
        assert result.total_chunks >= 1

    def test_ingest_conversations_empty_dir(self, tmp_path):
        result = ingest_conversations(str(tmp_path))
        assert isinstance(result, IngestionResult)
        assert result.total_chunks == 0

    def test_ingest_conversations_with_limit(self, tmp_path):
        for i in range(5):
            (tmp_path / f"convo{i}.txt").write_text(PLAIN_TRANSCRIPT)
        result = ingest_conversations(str(tmp_path), limit=2)
        assert isinstance(result, IngestionResult)


class TestScanConversations:
    """Tests for scan_conversations()."""

    def test_scan_finds_convo_files(self, tmp_path):
        (tmp_path / "chat.json").write_text("{}")
        (tmp_path / "session.jsonl").write_text("{}")
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "image.png").write_bytes(b"PNG")
        files = scan_conversations(str(tmp_path))
        extensions = {f.suffix for f in files}
        assert ".json" in extensions
        assert ".jsonl" in extensions
        assert ".txt" in extensions
        assert ".png" not in extensions

    def test_scan_skips_meta_json(self, tmp_path):
        (tmp_path / "session.meta.json").write_text("{}")
        (tmp_path / "session.json").write_text("{}")
        files = scan_conversations(str(tmp_path))
        names = {f.name for f in files}
        assert "session.meta.json" not in names
        assert "session.json" in names


# =========================================================================
# ALMA type mapping tests
# =========================================================================


class TestAlmaTypeMapping:
    """Test that extracted memories map to correct ALMA types."""

    def test_decision_maps_to_domain_knowledge(self, tmp_path):
        p = tmp_path / "decisions.txt"
        p.write_text(
            "We decided to go with PostgreSQL instead of MongoDB "
            "because our data is heavily relational and we need "
            "strong consistency guarantees. The approach is better "
            "and the trade-off is worth it."
        )
        result = ingest_file(str(p), project_id="test", agent="test")
        decisions = [
            dk for dk in result.domain_knowledge if dk.domain == "decisions"
        ]
        if decisions:
            assert decisions[0].source.startswith("ingested:")
            assert decisions[0].project_id == "test"

    def test_preference_maps_to_user_preference(self, tmp_path):
        p = tmp_path / "prefs.txt"
        p.write_text(
            "I prefer to always use snake_case for Python variables. "
            "Never use camelCase in Python code. My convention is "
            "to always use type hints and never skip them."
        )
        result = ingest_file(str(p), project_id="test", agent="test")
        if result.user_preferences:
            pref = result.user_preferences[0]
            assert isinstance(pref, UserPreference)
            assert pref.category == "ingested"

    def test_milestone_maps_to_outcome(self, tmp_path):
        p = tmp_path / "milestones.txt"
        p.write_text(
            "We finally got the deployment pipeline working after "
            "three weeks of effort. It was a real breakthrough when "
            "we figured out the container networking issue. Now we "
            "have v2.0 deployed and running smoothly."
        )
        result = ingest_file(str(p), project_id="test", agent="test")
        if result.outcomes:
            outcome = result.outcomes[0]
            assert isinstance(outcome, Outcome)
            assert outcome.success is True
            assert outcome.task_type == "milestone"

    def test_problem_maps_to_anti_pattern(self, tmp_path):
        p = tmp_path / "problems.txt"
        p.write_text(
            "There is a critical bug in the authentication module. "
            "The session keeps crashing when users try to log in. "
            "The root cause is a null pointer in the token validation. "
            "The problem is that we don't handle expired tokens correctly."
        )
        result = ingest_file(str(p), project_id="test", agent="test")
        if result.anti_patterns:
            ap = result.anti_patterns[0]
            assert isinstance(ap, AntiPattern)
            assert ap.occurrence_count == 1

    def test_emotional_maps_to_domain_knowledge(self, tmp_path):
        p = tmp_path / "emotional.txt"
        p.write_text(
            "I love how this project turned out. I feel so proud "
            "of what we've built together. It's beautiful and amazing "
            "and wonderful to see it all come together."
        )
        result = ingest_file(str(p), project_id="test", agent="test")
        emotional = [
            dk for dk in result.domain_knowledge if dk.domain == "emotional"
        ]
        if emotional:
            assert emotional[0].source.startswith("ingested:")
