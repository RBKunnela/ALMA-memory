"""
ALMA Consolidation Prompts.

LLM prompts for intelligently merging similar memories.
"""

# Prompt for merging multiple similar heuristics into one
MERGE_HEURISTICS_PROMPT = """You are a memory consolidation agent. Given these similar heuristics that have been identified as near-duplicates based on semantic similarity, create a single consolidated heuristic that captures the essence of all.

Similar Heuristics:
{heuristics}

Create a consolidated heuristic that:
1. Generalizes the condition to cover all cases
2. Combines the strategies into a comprehensive approach
3. Preserves any unique insights from individual heuristics

Output a JSON object with exactly these fields:
{{
    "condition": "The generalized condition that triggers this heuristic",
    "strategy": "The merged strategy combining all approaches",
    "confidence": <average confidence as a float between 0 and 1>
}}

Only output the JSON object, no other text."""

# Prompt for merging similar domain knowledge
MERGE_DOMAIN_KNOWLEDGE_PROMPT = """You are a memory consolidation agent. Given these similar domain knowledge facts that have been identified as near-duplicates, create a single consolidated fact that captures all the information.

Similar Domain Knowledge:
{knowledge_items}

Create a consolidated fact that:
1. Combines all unique information
2. Removes redundancy
3. Maintains accuracy

Output a JSON object with exactly these fields:
{{
    "fact": "The consolidated fact combining all information",
    "confidence": <average confidence as a float between 0 and 1>
}}

Only output the JSON object, no other text."""

# Prompt for merging anti-patterns
MERGE_ANTI_PATTERNS_PROMPT = """You are a memory consolidation agent. Given these similar anti-patterns that have been identified as near-duplicates, create a single consolidated anti-pattern.

Similar Anti-Patterns:
{anti_patterns}

Create a consolidated anti-pattern that:
1. Generalizes the pattern description
2. Combines all reasons why it's bad
3. Provides a comprehensive alternative

Output a JSON object with exactly these fields:
{{
    "pattern": "The generalized pattern to avoid",
    "why_bad": "Combined explanation of why this pattern is problematic",
    "better_alternative": "The recommended alternative approach"
}}

Only output the JSON object, no other text."""

# Prompt for merging outcomes (typically used for summarization rather than true merge)
MERGE_OUTCOMES_PROMPT = """You are a memory consolidation agent. Given these similar task outcomes, create a summary that captures the key learnings.

Similar Outcomes:
{outcomes}

Create a summary that:
1. Identifies the common task type
2. Notes the overall success/failure pattern
3. Highlights effective strategies

Output a JSON object with exactly these fields:
{{
    "task_type": "The common task type",
    "summary": "Summary of the outcomes and learnings",
    "recommended_strategy": "The most effective strategy based on the outcomes"
}}

Only output the JSON object, no other text."""
