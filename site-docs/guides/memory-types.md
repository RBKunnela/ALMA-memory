# Memory Types

ALMA supports five distinct memory types, each serving a specific purpose in agent learning.

## Overview

| Type | Purpose | Persistence | Example |
|------|---------|-------------|---------|
| **Heuristic** | Learned strategies that work | Permanent | "Test forms incrementally" |
| **Outcome** | Raw task results | Time-decaying | "OAuth test passed in 45s" |
| **Preference** | User constraints | Permanent | "No emojis in docs" |
| **Domain Knowledge** | Factual information | Permanent | "API uses JWT with 24h expiry" |
| **Anti-pattern** | What NOT to do | Permanent | "Don't use sleep() for waits" |

## Heuristics

Heuristics are learned strategies that have proven successful. They're formed automatically when similar outcomes repeat.

```python
# Heuristics are created automatically from repeated successful outcomes
alma.learn(
    agent="helena",
    task="Test login form",
    outcome="success",
    strategy_used="Tested empty fields first, then invalid formats"
)

# After 3+ similar successes, ALMA creates a heuristic:
# "When testing forms, validate empty fields before format checks"
```

**Configuration:**
```yaml
agents:
  helena:
    min_occurrences_for_heuristic: 3  # Require 3 successes
```

## Outcomes

Raw records of task completions. Used to form heuristics and track agent performance.

```python
alma.learn(
    agent="helena",
    task="Test checkout flow",
    outcome="success",  # or "failure"
    strategy_used="Used guest checkout path",
    duration_ms=30000,
    error_message=None,  # For failures
    feedback="User confirmed working"
)
```

## User Preferences

Persistent user constraints and preferences that apply across sessions.

```python
alma.add_preference(
    user_id="user-123",
    category="code_style",
    preference="Always use TypeScript strict mode",
    source="explicit_instruction"  # or "inferred_from_feedback"
)
```

**Categories:**
- `code_style` - Coding preferences
- `communication` - Output format, verbosity
- `workflow` - Process preferences
- `custom` - Any other category

## Domain Knowledge

Factual information about the project, APIs, or domain.

```python
alma.add_knowledge(
    agent="victor",
    domain="authentication",
    fact="The API uses OAuth 2.0 with PKCE for mobile clients",
    source="documentation"  # or "code_analysis", "user_stated"
)
```

## Anti-patterns

Explicit records of what NOT to do, with explanations and alternatives.

```python
alma.add_anti_pattern(
    agent="helena",
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests, wastes execution time",
    better_alternative="Use explicit waits with element conditions"
)
```

## Memory Retrieval

When retrieving, ALMA returns all relevant memory types:

```python
memories = alma.retrieve(
    task="Test the payment form",
    agent="helena",
    top_k=5
)

# Access by type
print(memories.heuristics)      # List of relevant strategies
print(memories.outcomes)        # Recent similar outcomes
print(memories.preferences)     # User preferences
print(memories.domain_knowledge) # Relevant facts
print(memories.anti_patterns)   # What to avoid

# Or get formatted for prompt injection
print(memories.to_prompt())
```

## Prompt Injection Format

The `to_prompt()` method formats memories for LLM consumption:

```
## Learned Strategies
- When testing forms, validate empty fields before format checks
- Use explicit waits instead of sleep()

## What to Avoid
- DON'T: Use sleep() for async waits
  WHY: Causes flaky tests, wastes execution time
  INSTEAD: Use explicit waits with element conditions

## User Preferences
- Always use TypeScript strict mode
- Prefer verbose test output

## Domain Knowledge
- The API uses OAuth 2.0 with PKCE for mobile clients
- JWT tokens expire after 24 hours
```
