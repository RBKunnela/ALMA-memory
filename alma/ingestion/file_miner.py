"""
ALMA Ingestion — File Miner.

Adapted from MemPalace (MIT License).

Reads project files, chunks them, and returns ALMA memory types.
No ChromaDB dependency. No storage -- returns results for the
caller to store.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from alma.observability.logging import get_logger
from alma.types import AntiPattern, DomainKnowledge, Outcome, UserPreference

from .memory_extractor import extract_memories

logger = get_logger(__name__)


READABLE_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".sh",
    ".csv",
    ".sql",
    ".toml",
}

SKIP_FILENAMES: Set[str] = {
    ".gitignore",
    "package-lock.json",
}

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
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass
class IngestionResult:
    """Result of ingesting one or more files into ALMA memory types.

    Attributes:
        domain_knowledge: Extracted domain knowledge facts.
        user_preferences: Extracted user preferences.
        outcomes: Extracted milestones/outcomes.
        anti_patterns: Extracted problems/anti-patterns.
        raw_chunks: Unclassified content as DomainKnowledge.
        source_file: Source file path (or directory path).
        total_chunks: Total number of chunks produced.
        classified_chunks: Number of chunks that matched a memory type.
    """

    domain_knowledge: List[DomainKnowledge] = field(default_factory=list)
    user_preferences: List[UserPreference] = field(default_factory=list)
    outcomes: List[Outcome] = field(default_factory=list)
    anti_patterns: List[AntiPattern] = field(default_factory=list)
    raw_chunks: List[DomainKnowledge] = field(default_factory=list)
    source_file: str = ""
    total_chunks: int = 0
    classified_chunks: int = 0

    def merge(self, other: "IngestionResult") -> "IngestionResult":
        """Merge another IngestionResult into this one.

        Args:
            other: Another IngestionResult to merge.

        Returns:
            Self, for chaining.
        """
        self.domain_knowledge.extend(other.domain_knowledge)
        self.user_preferences.extend(other.user_preferences)
        self.outcomes.extend(other.outcomes)
        self.anti_patterns.extend(other.anti_patterns)
        self.raw_chunks.extend(other.raw_chunks)
        self.total_chunks += other.total_chunks
        self.classified_chunks += other.classified_chunks
        return self


def chunk_text(content: str) -> List[dict]:
    """Split content into chunks with paragraph-aware boundaries.

    Args:
        content: Text content to chunk.

    Returns:
        List of dicts with "content" and "chunk_index" keys.
    """
    content = content.strip()
    if not content:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))

        # Try to break at paragraph boundary
        if end < len(content):
            newline_pos = content.rfind("\n\n", start, end)
            if newline_pos > start + CHUNK_SIZE // 2:
                end = newline_pos
            else:
                newline_pos = content.rfind("\n", start, end)
                if newline_pos > start + CHUNK_SIZE // 2:
                    end = newline_pos

        chunk = content[start:end].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(
                {
                    "content": chunk,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        start = end - CHUNK_OVERLAP if end < len(content) else end

    return chunks


def _memories_to_alma_types(
    memories: List[dict],
    raw_chunks: List[dict],
    source_file: str,
    project_id: str = "",
    agent: str = "ingestion",
) -> IngestionResult:
    """Convert extracted memory dicts to ALMA types.

    Args:
        memories: List of classified memory dicts from extract_memories.
        raw_chunks: List of all raw chunks (unclassified).
        source_file: Source file path for metadata.
        project_id: ALMA project ID.
        agent: Agent name for ALMA types.

    Returns:
        IngestionResult with ALMA-typed memories.
    """
    import uuid
    from datetime import datetime, timezone

    result = IngestionResult(
        source_file=source_file,
        total_chunks=len(raw_chunks),
        classified_chunks=len(memories),
    )

    classified_contents: set = set()

    for mem in memories:
        content = mem["content"]
        mem_type = mem["memory_type"]
        confidence = mem.get("confidence", 0.5)
        classified_contents.add(content)

        now = datetime.now(timezone.utc)
        uid = str(uuid.uuid4())

        if mem_type == "decision":
            result.domain_knowledge.append(
                DomainKnowledge(
                    id=uid,
                    agent=agent,
                    project_id=project_id,
                    domain="decisions",
                    fact=content,
                    source=f"ingested:{source_file}",
                    confidence=confidence,
                    last_verified=now,
                )
            )
        elif mem_type == "preference":
            result.user_preferences.append(
                UserPreference(
                    id=uid,
                    user_id=agent,
                    category="ingested",
                    preference=content,
                    source=f"ingested:{source_file}",
                    confidence=confidence,
                    timestamp=now,
                )
            )
        elif mem_type == "milestone":
            result.outcomes.append(
                Outcome(
                    id=uid,
                    agent=agent,
                    project_id=project_id,
                    task_type="milestone",
                    task_description=content,
                    success=True,
                    strategy_used="extracted_from_conversation",
                    timestamp=now,
                )
            )
        elif mem_type == "problem":
            result.anti_patterns.append(
                AntiPattern(
                    id=uid,
                    agent=agent,
                    project_id=project_id,
                    pattern=content,
                    why_bad="extracted from conversation",
                    better_alternative="",
                    occurrence_count=1,
                    last_seen=now,
                    created_at=now,
                )
            )
        elif mem_type == "emotional":
            result.domain_knowledge.append(
                DomainKnowledge(
                    id=uid,
                    agent=agent,
                    project_id=project_id,
                    domain="emotional",
                    fact=content,
                    source=f"ingested:{source_file}",
                    confidence=confidence,
                    last_verified=now,
                )
            )

    # Add unclassified chunks as raw DomainKnowledge
    for chunk in raw_chunks:
        chunk_content = chunk["content"]
        if chunk_content not in classified_contents:
            result.raw_chunks.append(
                DomainKnowledge(
                    id=str(uuid.uuid4()),
                    agent=agent,
                    project_id=project_id,
                    domain="raw",
                    fact=chunk_content,
                    source=f"ingested:{source_file}",
                    confidence=0.3,
                    last_verified=datetime.now(timezone.utc),
                )
            )

    return result


def ingest_file(
    filepath: str,
    project_id: str = "",
    agent: str = "ingestion",
    extract: bool = True,
) -> IngestionResult:
    """Ingest a single file and return ALMA memory types.

    Reads the file, chunks it, optionally extracts classified memories,
    and returns an IngestionResult with ALMA types.

    Args:
        filepath: Path to the file to ingest.
        project_id: ALMA project ID for the resulting memories.
        agent: Agent name for the resulting memories.
        extract: If True, run memory extraction on chunks. If False,
            return only raw chunks.

    Returns:
        IngestionResult with classified and raw memories.

    Raises:
        IOError: If the file cannot be read.
        ValueError: If the file is empty or too small.
    """
    path = Path(filepath).resolve()

    if not path.is_file():
        raise IOError(f"File not found: {filepath}")

    try:
        file_size = path.stat().st_size
    except OSError as e:
        raise IOError(f"Cannot stat {filepath}: {e}") from e

    if file_size > MAX_FILE_SIZE:
        raise IOError(f"File too large ({file_size // (1024 * 1024)} MB): {filepath}")

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise IOError(f"Could not read {filepath}: {e}") from e

    content = content.strip()
    if len(content) < MIN_CHUNK_SIZE:
        raise ValueError(f"File content too small ({len(content)} chars): {filepath}")

    raw_chunks = chunk_text(content)

    if extract:
        memories = extract_memories(content)
    else:
        memories = []

    return _memories_to_alma_types(
        memories=memories,
        raw_chunks=raw_chunks,
        source_file=str(path),
        project_id=project_id,
        agent=agent,
    )


def scan_directory(
    directory: str,
    respect_gitignore: bool = True,
    extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """Scan a directory for readable files.

    Args:
        directory: Directory path to scan.
        respect_gitignore: If True, skip .gitignored files (basic).
        extensions: Set of file extensions to include. Defaults to
            READABLE_EXTENSIONS.

    Returns:
        List of file paths found.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise IOError(f"Directory not found: {directory}")

    allowed_ext = extensions or READABLE_EXTENSIONS
    files: List[Path] = []

    for root, dirs, filenames in os.walk(dir_path):
        root_path = Path(root)

        # Skip known directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = root_path / filename

            if filename in SKIP_FILENAMES:
                continue
            if filepath.suffix.lower() not in allowed_ext:
                continue
            if filepath.is_symlink():
                continue
            try:
                if filepath.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            files.append(filepath)

    return files


def ingest_directory(
    directory: str,
    project_id: str = "",
    agent: str = "ingestion",
    extract: bool = True,
    extensions: Optional[Set[str]] = None,
    limit: int = 0,
) -> IngestionResult:
    """Ingest all readable files in a directory.

    Args:
        directory: Directory path to scan and ingest.
        project_id: ALMA project ID for the resulting memories.
        agent: Agent name for the resulting memories.
        extract: If True, run memory extraction. If False, raw only.
        extensions: Optional set of file extensions to include.
        limit: Maximum number of files to process (0 = no limit).

    Returns:
        IngestionResult with all memories merged.
    """
    files = scan_directory(directory, extensions=extensions)

    if limit > 0:
        files = files[:limit]

    combined = IngestionResult(source_file=directory)

    for filepath in files:
        try:
            result = ingest_file(
                str(filepath),
                project_id=project_id,
                agent=agent,
                extract=extract,
            )
            combined.merge(result)
        except (IOError, ValueError) as e:
            logger.warning(
                "skipping_file",
                extra={"file": str(filepath), "error": str(e)},
            )
            continue

    return combined
