"""
ALMA Ingestion — Conversation Miner.

Adapted from MemPalace (MIT License).

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), and returns
ALMA memory types. No ChromaDB dependency.
"""

import os
from pathlib import Path
from typing import Dict, List, Set

from alma.observability.logging import get_logger

from .file_miner import IngestionResult, _memories_to_alma_types
from .memory_extractor import extract_memories
from .normalizer import normalize

logger = get_logger(__name__)


# File types that might contain conversations
CONVO_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SKIP_DIRS: Set[str] = {
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    ".tox",
}


def chunk_exchanges(content: str) -> List[Dict]:
    """Chunk by exchange pair: one > turn + AI response = one unit.

    Falls back to paragraph chunking if no > markers are found.

    Args:
        content: Normalized transcript content.

    Returns:
        List of dicts with "content" and "chunk_index" keys.
    """
    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines)
    else:
        return _chunk_by_paragraph(content)


def _chunk_by_exchange(lines: List[str]) -> List[Dict]:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim. When the combined
    user-turn + response exceeds CHUNK_SIZE the response is split
    across consecutive chunks.
    """
    chunks: List[Dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines: List[str] = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith(
                    "---"
                ):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines)
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            # Split into multiple chunks when exchange exceeds CHUNK_SIZE
            if len(content) > CHUNK_SIZE:
                first_part = content[:CHUNK_SIZE]
                if len(first_part.strip()) > MIN_CHUNK_SIZE:
                    chunks.append(
                        {
                            "content": first_part,
                            "chunk_index": len(chunks),
                        }
                    )
                remainder = content[CHUNK_SIZE:]
                while remainder:
                    part = remainder[:CHUNK_SIZE]
                    remainder = remainder[CHUNK_SIZE:]
                    if len(part.strip()) > MIN_CHUNK_SIZE:
                        chunks.append(
                            {
                                "content": part,
                                "chunk_index": len(chunks),
                            }
                        )
            elif len(content.strip()) > MIN_CHUNK_SIZE:
                chunks.append(
                    {
                        "content": content,
                        "chunk_index": len(chunks),
                    }
                )
        else:
            i += 1

    return chunks


def _chunk_by_paragraph(content: str) -> List[Dict]:
    """Fallback: chunk by paragraph breaks."""
    chunks: List[Dict] = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > 20:
        lines = content.split("\n")
        for i in range(0, len(lines), 25):
            group = "\n".join(lines[i : i + 25]).strip()
            if len(group) > MIN_CHUNK_SIZE:
                chunks.append({"content": group, "chunk_index": len(chunks)})
        return chunks

    for para in paragraphs:
        if len(para) > MIN_CHUNK_SIZE:
            chunks.append({"content": para, "chunk_index": len(chunks)})

    return chunks


def scan_conversations(convo_dir: str) -> List[Path]:
    """Find all potential conversation files in a directory.

    Args:
        convo_dir: Directory path to scan for conversation files.

    Returns:
        List of file paths found.
    """
    convo_path = Path(convo_dir).resolve()
    if not convo_path.is_dir():
        raise IOError(f"Directory not found: {convo_dir}")

    files: List[Path] = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                if filepath.is_symlink():
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


def ingest_conversation(
    filepath: str,
    project_id: str = "",
    agent: str = "ingestion",
    extract_mode: str = "exchange",
) -> IngestionResult:
    """Ingest a single conversation file.

    Args:
        filepath: Path to the conversation file.
        project_id: ALMA project ID.
        agent: Agent name for memories.
        extract_mode: "exchange" for Q+A chunking, "general" for
            memory type extraction.

    Returns:
        IngestionResult with ALMA-typed memories.

    Raises:
        IOError: If the file cannot be read.
        ValueError: If the file is empty or cannot be normalized.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise IOError(f"File not found: {filepath}")

    # Normalize format
    try:
        content = normalize(str(path))
    except (OSError, ValueError) as e:
        raise IOError(f"Could not normalize {filepath}: {e}") from e

    if not content or len(content.strip()) < MIN_CHUNK_SIZE:
        raise ValueError(f"Normalized content too small: {filepath}")

    # Chunk
    if extract_mode == "general":
        memories = extract_memories(content)
        raw_chunks = [
            {"content": m["content"], "chunk_index": m["chunk_index"]} for m in memories
        ]
    else:
        raw_chunks = chunk_exchanges(content)
        memories = extract_memories(content)

    return _memories_to_alma_types(
        memories=memories,
        raw_chunks=raw_chunks,
        source_file=str(path),
        project_id=project_id,
        agent=agent,
    )


def ingest_conversations(
    convo_dir: str,
    project_id: str = "",
    agent: str = "ingestion",
    extract_mode: str = "exchange",
    limit: int = 0,
) -> IngestionResult:
    """Ingest all conversation files in a directory.

    Args:
        convo_dir: Directory with conversation files.
        project_id: ALMA project ID.
        agent: Agent name for memories.
        extract_mode: "exchange" for Q+A chunking, "general" for
            memory type extraction.
        limit: Maximum files to process (0 = no limit).

    Returns:
        IngestionResult with all memories merged.
    """
    files = scan_conversations(convo_dir)

    if limit > 0:
        files = files[:limit]

    combined = IngestionResult(source_file=convo_dir)

    for filepath in files:
        try:
            result = ingest_conversation(
                str(filepath),
                project_id=project_id,
                agent=agent,
                extract_mode=extract_mode,
            )
            combined.merge(result)
        except (IOError, ValueError) as e:
            logger.warning(
                "skipping_conversation",
                extra={"file": str(filepath), "error": str(e)},
            )
            continue

    return combined
