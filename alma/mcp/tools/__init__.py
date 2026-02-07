"""
ALMA MCP Tool Definitions.

Provides the tool functions that can be called via MCP protocol.
Each tool corresponds to an ALMA operation.

Both sync and async versions are provided:
- Sync tools: alma_retrieve, alma_learn, etc.
- Async tools: async_alma_retrieve, async_alma_learn, etc.

The async tools use ALMA's async API for better concurrency in
async MCP server implementations.

This package splits tools into modules by category:
- _common: Shared serialization helpers
- retrieval: Memory retrieval tools (standard, mode-aware, scoped, verified, etc.)
- learning: Memory recording tools (learn, preferences, knowledge, forgetting, etc.)
- workflow: Workflow tools (checkpoints, resume, merge, artifacts, etc.)
- admin: Admin tools (stats, health, trust patterns, etc.)
"""

# Shared utility
from alma.mcp.tools._common import _serialize_memory_slice

# Admin tools
from alma.mcp.tools.admin import (
    alma_get_trust_warnings,
    alma_health,
    alma_stats,
    alma_store_trust_pattern,
    async_alma_get_trust_warnings,
    async_alma_health,
    async_alma_stats,
    async_alma_store_trust_pattern,
)

# Learning tools
from alma.mcp.tools.learning import (
    alma_add_knowledge,
    alma_add_preference,
    alma_compress_and_learn,
    alma_consolidate,
    alma_extract_heuristic,
    alma_forget,
    alma_get_weak_memories,
    alma_learn,
    alma_reinforce,
    alma_smart_forget,
    async_alma_add_knowledge,
    async_alma_add_preference,
    async_alma_compress_and_learn,
    async_alma_extract_heuristic,
    async_alma_forget,
    async_alma_get_weak_memories,
    async_alma_learn,
    async_alma_reinforce,
    async_alma_smart_forget,
)

# Retrieval tools
from alma.mcp.tools.retrieval import (
    alma_get_memory_full,
    alma_retrieve,
    alma_retrieve_for_mode,
    alma_retrieve_progressive,
    alma_retrieve_scoped,
    alma_retrieve_smart,
    alma_retrieve_verified,
    alma_retrieve_with_budget,
    alma_retrieve_with_trust,
    async_alma_get_memory_full,
    async_alma_retrieve,
    async_alma_retrieve_for_mode,
    async_alma_retrieve_progressive,
    async_alma_retrieve_scoped,
    async_alma_retrieve_smart,
    async_alma_retrieve_verified,
    async_alma_retrieve_with_budget,
    async_alma_retrieve_with_trust,
)

# Workflow tools
from alma.mcp.tools.workflow import (
    alma_checkpoint,
    alma_cleanup_checkpoints,
    alma_get_artifacts,
    alma_link_artifact,
    alma_merge_states,
    alma_resume,
    alma_workflow_learn,
    async_alma_checkpoint,
    async_alma_link_artifact,
    async_alma_resume,
    async_alma_workflow_learn,
)

__all__ = [
    # Shared utility
    "_serialize_memory_slice",
    # Sync retrieval tools
    "alma_retrieve",
    "alma_retrieve_for_mode",
    "alma_retrieve_smart",
    "alma_retrieve_scoped",
    "alma_retrieve_verified",
    "alma_retrieve_with_trust",
    "alma_retrieve_with_budget",
    "alma_retrieve_progressive",
    "alma_get_memory_full",
    # Sync learning tools
    "alma_learn",
    "alma_add_preference",
    "alma_add_knowledge",
    "alma_forget",
    "alma_consolidate",
    "alma_reinforce",
    "alma_get_weak_memories",
    "alma_smart_forget",
    "alma_compress_and_learn",
    "alma_extract_heuristic",
    # Sync workflow tools
    "alma_checkpoint",
    "alma_resume",
    "alma_merge_states",
    "alma_workflow_learn",
    "alma_link_artifact",
    "alma_get_artifacts",
    "alma_cleanup_checkpoints",
    # Sync admin tools
    "alma_stats",
    "alma_health",
    "alma_store_trust_pattern",
    "alma_get_trust_warnings",
    # Async retrieval tools
    "async_alma_retrieve",
    "async_alma_retrieve_for_mode",
    "async_alma_retrieve_smart",
    "async_alma_retrieve_scoped",
    "async_alma_retrieve_verified",
    "async_alma_retrieve_with_trust",
    "async_alma_retrieve_with_budget",
    "async_alma_retrieve_progressive",
    "async_alma_get_memory_full",
    # Async learning tools
    "async_alma_learn",
    "async_alma_add_preference",
    "async_alma_add_knowledge",
    "async_alma_forget",
    "async_alma_reinforce",
    "async_alma_get_weak_memories",
    "async_alma_smart_forget",
    "async_alma_compress_and_learn",
    "async_alma_extract_heuristic",
    # Async workflow tools
    "async_alma_checkpoint",
    "async_alma_resume",
    "async_alma_workflow_learn",
    "async_alma_link_artifact",
    # Async admin tools
    "async_alma_stats",
    "async_alma_health",
    "async_alma_store_trust_pattern",
    "async_alma_get_trust_warnings",
]
