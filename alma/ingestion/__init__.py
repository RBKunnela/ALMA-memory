"""
ALMA Ingestion Package.

Adapted from MemPalace (MIT License).

Provides file and conversation ingestion with automatic memory type
extraction. Converts 6 chat export formats to standard transcripts
and classifies content into ALMA memory types (DomainKnowledge,
UserPreference, Outcome, AntiPattern).

Usage::

    from alma.ingestion import ingest_file, ingest_directory
    from alma.ingestion import ingest_conversations
    from alma.ingestion import normalize, detect_format
    from alma.ingestion import extract_memories

    # Ingest a single file
    result = ingest_file("path/to/file.py")
    print(result.domain_knowledge)

    # Ingest a directory of conversation exports
    result = ingest_conversations("path/to/conversations/")
    print(result.outcomes)
"""

from alma.ingestion.conversation_miner import (
    chunk_exchanges,
    ingest_conversation,
    ingest_conversations,
    scan_conversations,
)
from alma.ingestion.file_miner import (
    IngestionResult,
    chunk_text,
    ingest_directory,
    ingest_file,
    scan_directory,
)
from alma.ingestion.memory_extractor import extract_memories
from alma.ingestion.normalizer import detect_format, normalize, normalize_text

__all__ = [
    # File ingestion
    "ingest_file",
    "ingest_directory",
    "scan_directory",
    "chunk_text",
    "IngestionResult",
    # Conversation ingestion
    "ingest_conversations",
    "ingest_conversation",
    "scan_conversations",
    "chunk_exchanges",
    # Normalization
    "normalize",
    "normalize_text",
    "detect_format",
    # Extraction
    "extract_memories",
]
