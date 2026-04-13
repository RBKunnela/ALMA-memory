"""
ALMA Ingestion — Normalizer.

Adapted from MemPalace (MIT License).

Converts any chat export format to a standard transcript format.

Supported formats:
    - Plain text with > markers (pass through)
    - Claude.ai JSON export
    - ChatGPT conversations.json
    - Claude Code JSONL (with tool_use/tool_result block capture)
    - OpenAI Codex CLI JSONL
    - Slack JSON export
    - Plain text (pass through for paragraph chunking)

No API key. No internet. Everything local.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from alma.observability.logging import get_logger

logger = get_logger(__name__)


def detect_format(filepath: str) -> str:
    """Detect the chat export format of a file.

    Args:
        filepath: Path to the file to detect.

    Returns:
        One of: "claude_code_jsonl", "codex_jsonl", "claude_ai_json",
        "chatgpt_json", "slack_json", "transcript", "plain_text".

    Raises:
        IOError: If the file cannot be read or is too large.
    """
    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}") from e
    if file_size > 500 * 1024 * 1024:
        raise IOError(f"File too large ({file_size // (1024 * 1024)} MB): {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}") from e

    if not content.strip():
        return "plain_text"

    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return "transcript"

    ext = Path(filepath).suffix.lower()
    if ext in (".json", ".jsonl") or content.strip()[:1] in ("{", "["):
        fmt = _detect_json_format(content)
        if fmt:
            return fmt

    return "plain_text"


def _detect_json_format(content: str) -> Optional[str]:
    """Detect which JSON chat format a string matches."""
    if _is_claude_code_jsonl(content):
        return "claude_code_jsonl"
    if _is_codex_jsonl(content):
        return "codex_jsonl"

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    if _is_claude_ai_json(data):
        return "claude_ai_json"
    if _is_chatgpt_json(data):
        return "chatgpt_json"
    if _is_slack_json(data):
        return "slack_json"

    return None


def _is_claude_code_jsonl(content: str) -> bool:
    """Check if content looks like Claude Code JSONL."""
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    msg_types = set()
    for line in lines[:20]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            msg_types.add(entry.get("type", ""))
    return bool(msg_types & {"human", "user", "assistant"})


def _is_codex_jsonl(content: str) -> bool:
    """Check if content looks like OpenAI Codex JSONL."""
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    for line in lines[:20]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("type") == "session_meta":
            return True
    return False


def _is_claude_ai_json(data) -> bool:
    """Check if data looks like Claude.ai JSON export."""
    if isinstance(data, dict):
        if "messages" in data or "chat_messages" in data:
            return True
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "chat_messages" in first or "messages" in first:
                return True
            if "role" in first or "sender" in first:
                return True
    return False


def _is_chatgpt_json(data) -> bool:
    """Check if data looks like ChatGPT conversations.json."""
    return isinstance(data, dict) and "mapping" in data


def _is_slack_json(data) -> bool:
    """Check if data looks like Slack JSON export."""
    if not isinstance(data, list):
        return False
    for item in data[:5]:
        if isinstance(item, dict) and item.get("type") == "message":
            return True
    return False


def normalize(filepath: str) -> str:
    """Load a file and normalize to transcript format.

    Converts supported chat export formats to a standard transcript
    format with > markers for user turns. Plain text files pass
    through unchanged.

    Args:
        filepath: Path to the file to normalize.

    Returns:
        Normalized transcript string.

    Raises:
        IOError: If the file cannot be read or is too large.
    """
    try:
        file_size = os.path.getsize(filepath)
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}") from e
    if file_size > 500 * 1024 * 1024:
        raise IOError(f"File too large ({file_size // (1024 * 1024)} MB): {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}") from e

    if not content.strip():
        return content

    # Already has > markers -- pass through
    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return content

    # Try JSON normalization
    ext = Path(filepath).suffix.lower()
    if ext in (".json", ".jsonl") or content.strip()[:1] in ("{", "["):
        normalized = _try_normalize_json(content)
        if normalized:
            return normalized

    return content


def normalize_text(content: str) -> str:
    """Normalize chat text content without reading from file.

    Args:
        content: Raw text content to normalize.

    Returns:
        Normalized transcript string.
    """
    if not content or not content.strip():
        return content or ""

    lines = content.split("\n")
    if sum(1 for line in lines if line.strip().startswith(">")) >= 3:
        return content

    if content.strip()[:1] in ("{", "["):
        normalized = _try_normalize_json(content)
        if normalized:
            return normalized

    return content


def _try_normalize_json(content: str) -> Optional[str]:
    """Try all known JSON chat schemas."""
    normalized = _try_claude_code_jsonl(content)
    if normalized:
        return normalized

    normalized = _try_codex_jsonl(content)
    if normalized:
        return normalized

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    for parser in (_try_claude_ai_json, _try_chatgpt_json, _try_slack_json):
        normalized = parser(data)
        if normalized:
            return normalized

    return None


def _try_claude_code_jsonl(content: str) -> Optional[str]:
    """Claude Code JSONL sessions."""
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages: List[Tuple[str, str]] = []
    tool_use_map: dict = {}

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type", "")
        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        msg_content = message.get("content", "")

        # Build tool_use_map from assistant messages
        if msg_type == "assistant" and isinstance(msg_content, list):
            for block in msg_content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    if tool_id:
                        tool_use_map[tool_id] = block.get("name", "Unknown")

        if msg_type in ("human", "user"):
            is_tool_only = isinstance(msg_content, list) and all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in msg_content
            )
            text = _extract_content(msg_content, tool_use_map=tool_use_map)
            if text:
                if is_tool_only and messages and messages[-1][0] == "assistant":
                    prev_role, prev_text = messages[-1]
                    messages[-1] = (prev_role, prev_text + "\n" + text)
                elif not is_tool_only:
                    messages.append(("user", text))
        elif msg_type == "assistant":
            text = _extract_content(msg_content, tool_use_map=tool_use_map)
            if text:
                if messages and messages[-1][0] == "assistant":
                    prev_role, prev_text = messages[-1]
                    messages[-1] = (prev_role, prev_text + "\n" + text)
                else:
                    messages.append(("assistant", text))

    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_codex_jsonl(content: str) -> Optional[str]:
    """OpenAI Codex CLI sessions.

    Uses only event_msg entries (user_message / agent_message) which
    represent the canonical conversation turns.
    """
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    messages: List[Tuple[str, str]] = []
    has_session_meta = False

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry_type == "session_meta":
            has_session_meta = True
            continue

        if entry_type != "event_msg":
            continue

        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            continue

        payload_type = payload.get("type", "")
        msg = payload.get("message")
        if not isinstance(msg, str):
            continue
        text = msg.strip()
        if not text:
            continue

        if payload_type == "user_message":
            messages.append(("user", text))
        elif payload_type == "agent_message":
            messages.append(("assistant", text))

    if len(messages) >= 2 and has_session_meta:
        return _messages_to_transcript(messages)
    return None


def _try_claude_ai_json(data) -> Optional[str]:
    """Claude.ai JSON export: flat messages list or privacy export."""
    if isinstance(data, dict):
        data = data.get("messages", data.get("chat_messages", []))
    if not isinstance(data, list):
        return None

    # Privacy export: array of conversation objects
    if (
        data
        and isinstance(data[0], dict)
        and ("chat_messages" in data[0] or "messages" in data[0])
    ):
        transcripts = []
        for convo in data:
            if not isinstance(convo, dict):
                continue
            chat_msgs = convo.get("chat_messages") or convo.get("messages", [])
            messages = _collect_claude_messages(chat_msgs)
            if len(messages) >= 2:
                transcripts.append(_messages_to_transcript(messages))
        if transcripts:
            return "\n\n".join(transcripts)
        return None

    # Flat messages list
    messages = _collect_claude_messages(data)
    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _collect_claude_messages(items) -> List[Tuple[str, str]]:
    """Extract (role, text) pairs from a Claude.ai message list."""
    messages: List[Tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("sender", "")
        text = (
            _extract_content(item.get("content", ""))
            or (item.get("text") or "").strip()
        )
        if role in ("user", "human") and text:
            messages.append(("user", text))
        elif role in ("assistant", "ai") and text:
            messages.append(("assistant", text))
    return messages


def _try_chatgpt_json(data) -> Optional[str]:
    """ChatGPT conversations.json with mapping tree."""
    if not isinstance(data, dict) or "mapping" not in data:
        return None
    mapping = data["mapping"]
    messages: List[Tuple[str, str]] = []

    root_id = None
    fallback_root = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            if node.get("message") is None:
                root_id = node_id
                break
            elif fallback_root is None:
                fallback_root = node_id
    if not root_id:
        root_id = fallback_root

    if root_id:
        current_id = root_id
        visited: set = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id, {})
            msg = node.get("message")
            if msg:
                role = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = " ".join(
                    str(p) for p in parts if isinstance(p, str) and p
                ).strip()
                if role == "user" and text:
                    messages.append(("user", text))
                elif role == "assistant" and text:
                    messages.append(("assistant", text))
            children = node.get("children", [])
            current_id = children[0] if children else None

    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _try_slack_json(data) -> Optional[str]:
    """Slack channel export.

    Optimized for 2-person DMs. In channels with 3+ people, alternating
    speakers are labeled user/assistant to preserve exchange structure.
    """
    if not isinstance(data, list):
        return None
    messages: List[Tuple[str, str]] = []
    seen_users: dict = {}
    last_role = None

    for item in data:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        user_id = item.get("user", item.get("username", ""))
        text = item.get("text", "").strip()
        if not text or not user_id:
            continue
        if user_id not in seen_users:
            if not seen_users:
                seen_users[user_id] = "user"
            elif last_role == "user":
                seen_users[user_id] = "assistant"
            else:
                seen_users[user_id] = "user"
        last_role = seen_users[user_id]
        messages.append((seen_users[user_id], text))

    if len(messages) >= 2:
        return _messages_to_transcript(messages)
    return None


def _extract_content(content, tool_use_map: Optional[dict] = None) -> str:
    """Pull text from content -- handles str, list of blocks, or dict.

    Args:
        content: Message content -- string, list of content blocks,
            or dict.
        tool_use_map: Optional mapping of tool_use_id to tool_name.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    parts.append(item.get("text", ""))
                elif block_type == "tool_use":
                    parts.append(_format_tool_use(item))
                elif block_type == "tool_result":
                    tid = item.get("tool_use_id", "")
                    tname = (tool_use_map or {}).get(tid, "Unknown")
                    result_content = item.get("content", "")
                    formatted = _format_tool_result(result_content, tname)
                    if formatted:
                        parts.append(formatted)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return content.get("text", "").strip()
    return ""


def _format_tool_use(block: dict) -> str:
    """Format a tool_use block into a human-readable one-liner."""
    name = block.get("name", "Unknown")
    inp = block.get("input", {})

    if name == "Bash":
        cmd = inp.get("command", "")
        if len(cmd) > 200:
            cmd = cmd[:200] + "..."
        return f"[Bash] {cmd}"

    if name == "Read":
        path = inp.get("file_path", "?")
        offset = inp.get("offset")
        limit = inp.get("limit")
        if offset is not None and limit is not None:
            try:
                return f"[Read {path}:{offset}-{int(offset) + int(limit)}]"
            except (ValueError, TypeError):
                return f"[Read {path}:{offset}+{limit}]"
        return f"[Read {path}]"

    if name == "Grep":
        pattern = inp.get("pattern", "")
        target = inp.get("path") or inp.get("glob") or ""
        return f"[Grep] {pattern} in {target}"

    if name == "Glob":
        pattern = inp.get("pattern", "")
        return f"[Glob] {pattern}"

    if name in ("Edit", "Write"):
        path = inp.get("file_path", "?")
        return f"[{name} {path}]"

    # Unknown tool -- serialize input, truncate
    summary = json.dumps(inp, separators=(",", ":"))
    if len(summary) > 200:
        summary = summary[:200] + "..."
    return f"[{name}] {summary}"


_TOOL_RESULT_MAX_LINES_BASH = 20
_TOOL_RESULT_MAX_MATCHES = 20
_TOOL_RESULT_MAX_BYTES = 2048


def _format_tool_result(content, tool_name: str) -> str:
    """Format a tool_result based on the originating tool's type.

    Args:
        content: Result text (str) or list of content blocks.
        tool_name: Name of the tool that produced this result.

    Returns:
        Formatted string prefixed with ``-> ``, or empty string.
    """
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts)
    else:
        text = str(content) if content else ""

    text = text.strip()
    if not text:
        return ""

    # Read/Edit/Write -- omit result
    if tool_name in ("Read", "Edit", "Write"):
        return ""

    lines = text.split("\n")

    # Bash -- head + tail
    if tool_name == "Bash":
        n = _TOOL_RESULT_MAX_LINES_BASH
        if len(lines) <= n * 2:
            return "-> " + "\n-> ".join(lines)
        head = lines[:n]
        tail = lines[-n:]
        omitted = len(lines) - 2 * n
        return (
            "-> "
            + "\n-> ".join(head)
            + f"\n-> ... [{omitted} lines omitted] ..."
            + "\n-> "
            + "\n-> ".join(tail)
        )

    # Grep/Glob -- cap matches
    if tool_name in ("Grep", "Glob"):
        cap = _TOOL_RESULT_MAX_MATCHES
        if len(lines) <= cap:
            return "-> " + "\n-> ".join(lines)
        kept = lines[:cap]
        remaining = len(lines) - cap
        return "-> " + "\n-> ".join(kept) + f"\n-> ... [{remaining} more matches]"

    # Unknown -- byte cap
    if len(text) > _TOOL_RESULT_MAX_BYTES:
        return (
            "-> "
            + text[:_TOOL_RESULT_MAX_BYTES]
            + f"... [truncated, {len(text)} chars]"
        )
    return "-> " + text


def _messages_to_transcript(messages: List[Tuple[str, str]]) -> str:
    """Convert [(role, text), ...] to transcript format with > markers."""
    lines = []
    i = 0
    while i < len(messages):
        role, text = messages[i]
        if role == "user":
            lines.append(f"> {text}")
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                lines.append(messages[i + 1][1])
                i += 2
            else:
                i += 1
        else:
            lines.append(text)
            i += 1
        lines.append("")
    return "\n".join(lines)
