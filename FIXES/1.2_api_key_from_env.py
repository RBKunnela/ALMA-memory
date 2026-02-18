# Fix 1.2: API Key from Environment Variable Only
# File: alma/consolidation/core.py
# Impact: Prevents credential leaks in git history

import os
from alma.consolidation.llm_interface import ConsolidationError


def get_llm_api_key() -> str:
    """
    Get LLM API key from environment variable.

    Returns:
        LLM API key

    Raises:
        ConsolidationError: If LLM_API_KEY env var not set

    Note:
        - Key must be in environment, never in config files
        - In deployment: export LLM_API_KEY=sk-...
        - Never commit API keys to git!
    """
    api_key = os.environ.get('LLM_API_KEY')

    if not api_key:
        raise ConsolidationError(
            "LLM_API_KEY environment variable not set. "
            "Set it before running: export LLM_API_KEY=sk-..."
        )

    if len(api_key) < 10:
        raise ConsolidationError(
            "LLM_API_KEY appears invalid (too short). "
            "Check the environment variable is set correctly."
        )

    return api_key


# Usage in core.py:
def consolidate_memories(memories, ...):
    """Consolidate memories using LLM."""
    api_key = get_llm_api_key()

    # Configure OpenAI with API key
    import openai
    openai.api_key = api_key

    # ... rest of consolidation logic
