# Ingestion Package

**Module:** `alma/ingestion/`
**Since:** v0.9.0
**Origin:** Adapted from MemPalace (MIT License)

## Overview

The ingestion package converts files and conversation exports into ALMA memory types. It supports 6 chat export formats, automatic memory type classification via regex heuristics, and configurable chunking. No LLM required -- everything runs locally.

## Supported Formats

| Format | File Type | Auto-Detected By |
|--------|-----------|------------------|
| Claude Code | `.jsonl` | JSONL with `type` field containing `assistant`/`human` |
| OpenAI Codex CLI | `.jsonl` | JSONL with `role` field containing `user`/`assistant` |
| Claude.ai JSON | `.json` | JSON with `chat_messages` array |
| ChatGPT JSON | `.json` | JSON array with `mapping` keys |
| Slack JSON | `.json` | JSON array with `ts` and `text` fields |
| Plain text / Transcript | `.txt`, `.md` | Lines starting with `>` markers, or fallback |

Format detection is automatic via `detect_format()`.

## File Ingestion

Ingest source code, documentation, or any readable text files.

```python
from alma.ingestion import ingest_file, ingest_directory

# Ingest a single file
result = ingest_file("path/to/file.py")
print(f"Found {len(result.domain_knowledge)} knowledge items")
print(f"Found {len(result.outcomes)} outcomes")
print(f"Found {len(result.preferences)} preferences")

# Ingest an entire directory (recursive)
result = ingest_directory("path/to/project/")
for dk in result.domain_knowledge:
    print(f"  {dk.fact[:80]}...")
```

### Supported File Extensions

`.txt`, `.md`, `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.json`, `.yaml`, `.yml`,
`.html`, `.css`, `.java`, `.go`, `.rs`, `.rb`, `.sh`, `.csv`, `.sql`, `.toml`

### Skipped Directories

`__pycache__`, `.git`, `node_modules`, `.venv`, `venv`, `dist`, `build`, and others.

### Chunking

Files are chunked with configurable parameters:
- **Chunk size:** 800 characters (default)
- **Overlap:** 100 characters
- **Minimum chunk size:** 50 characters
- **Maximum file size:** 10 MB

## Conversation Ingestion

Ingest chat exports from AI assistants and messaging platforms.

```python
from alma.ingestion import ingest_conversations, ingest_conversation

# Ingest all conversation files in a directory
result = ingest_conversations("path/to/chat-exports/")
print(f"Extracted {len(result.domain_knowledge)} knowledge memories")
print(f"Extracted {len(result.preferences)} preference memories")

# Ingest a single conversation file
result = ingest_conversation("path/to/claude-code-session.jsonl")
```

### Conversation Chunking

Conversations are chunked by exchange pairs (one user turn + AI response = one chunk). Falls back to paragraph chunking if no turn markers are detected.

## Memory Type Mapping

The memory extractor classifies text chunks into 5 categories using regex heuristics:

| Extracted Type | ALMA Memory Type | Trigger Patterns |
|---------------|-----------------|------------------|
| DECISIONS | `DomainKnowledge` | "we went with X", "decided to", "because", "trade-off" |
| PREFERENCES | `UserPreference` | "I prefer", "always use", "never do", "my rule is" |
| MILESTONES | `Outcome` | "it works", "fixed", "breakthrough", "figured out" |
| PROBLEMS | `AntiPattern` | "broke", "root cause", "bug", "failed" |
| EMOTIONAL | `DomainKnowledge` | Feelings, relationships, vulnerability context |

```python
from alma.ingestion import extract_memories

memories = extract_memories("We decided to use PostgreSQL because of its pgvector support.")
# Returns: [{"type": "DECISIONS", "content": "...", "confidence": 0.8}]
```

## Normalization

Convert any supported format to a standard transcript before processing.

```python
from alma.ingestion import detect_format, normalize

# Detect format
fmt = detect_format("session.jsonl")
# Returns: "claude_code_jsonl"

# Normalize to standard transcript
transcript = normalize("session.jsonl")
# Returns plain text with > markers for user turns
```

## IngestionResult

All ingestion functions return an `IngestionResult` dataclass:

```python
@dataclass
class IngestionResult:
    domain_knowledge: List[DomainKnowledge]
    outcomes: List[Outcome]
    preferences: List[UserPreference]
    anti_patterns: List[AntiPattern]
    chunks_processed: int
    files_processed: int
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 800 | Characters per chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `max_file_size` | 10 MB | Skip files larger than this |
| `min_chunk_size` | 50 | Discard chunks smaller than this |
