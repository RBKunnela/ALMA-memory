"""
ALMA Memory Compression Pipeline.

Provides intelligent compression of verbose content into structured memories.
Based on Memory Wall principles: "Do not upload 40 pages hoping the AI extracts
what matters. You need to do the compression work."

Key features:
- Multiple compression levels (NONE, LIGHT, MEDIUM, AGGRESSIVE)
- LLM-based intelligent extraction
- Rule-based fallback without LLM
- Key fact and constraint extraction
- Heuristic extraction from multiple experiences
- Deduplication against existing knowledge
"""

from alma.compression.pipeline import (
    CompressedMemory,
    CompressionConfig,
    CompressionLevel,
    CompressionResult,
    MemoryCompressor,
    create_compressor,
)

__all__ = [
    "CompressionLevel",
    "CompressedMemory",
    "CompressionResult",
    "CompressionConfig",
    "MemoryCompressor",
    "create_compressor",
]
