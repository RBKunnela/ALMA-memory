# Quick Start

Build your first memory-powered AI agent in 5 minutes.

## 1. Create Configuration

Create a `.alma/config.yaml` file:

```yaml
alma:
  project_id: "my-project"
  storage: sqlite
  embedding_provider: local
  storage_dir: .alma
  db_name: alma.db
  embedding_dim: 384

  agents:
    my-agent:
      domain: general
      can_learn:
        - strategies
        - patterns
      cannot_learn: []
      min_occurrences_for_heuristic: 3
```

## 2. Initialize ALMA

=== "Python"

    ```python
    from alma import ALMA

    # Load from config
    alma = ALMA.from_config(".alma/config.yaml")

    # Or create programmatically
    alma = ALMA(
        storage=my_storage,
        retrieval_engine=my_retrieval,
        learning_protocol=my_learning,
        project_id="my-project"
    )
    ```

=== "TypeScript"

    ```typescript
    import { ALMA } from '@rbkunnela/alma-memory';

    const alma = new ALMA({
      baseUrl: 'http://localhost:8765',  // MCP server URL
      projectId: 'my-project'
    });
    ```

## 3. Retrieve Memories Before a Task

```python
# Get relevant memories for the current task
memories = alma.retrieve(
    task="Implement user authentication",
    agent="my-agent",
    top_k=5
)

# Inject into your LLM prompt
prompt = f"""
You are an AI assistant. Use the following knowledge from past tasks:

{memories.to_prompt()}

Now complete this task: Implement user authentication
"""
```

## 4. Learn from Task Outcomes

```python
# After the task completes, record the outcome
alma.learn(
    agent="my-agent",
    task="Implement user authentication",
    outcome="success",  # or "failure"
    strategy_used="Used JWT with refresh tokens and PKCE flow",
    duration_ms=45000,
)
```

## 5. Add Domain Knowledge

```python
# Store factual knowledge
alma.add_knowledge(
    agent="my-agent",
    domain="authentication",
    fact="The API uses OAuth 2.0 with PKCE for mobile clients",
    source="documentation"
)
```

## 6. Add User Preferences

```python
# Store user preferences that persist across sessions
alma.add_preference(
    user_id="user-123",
    category="code_style",
    preference="Always use TypeScript strict mode",
    source="explicit_instruction"
)
```

## 7. Record Anti-Patterns

```python
# Learn what NOT to do
alma.add_anti_pattern(
    agent="my-agent",
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests and unpredictable timing",
    better_alternative="Use explicit waits with conditions"
)
```

## Complete Example

```python
from alma import ALMA

# Initialize
alma = ALMA.from_config(".alma/config.yaml")

async def run_task(task: str, agent: str):
    # 1. Retrieve relevant memories
    memories = alma.retrieve(task=task, agent=agent, top_k=5)

    # 2. Build prompt with memories
    prompt = f"""
    ## Past Knowledge
    {memories.to_prompt()}

    ## Current Task
    {task}

    Complete the task using the knowledge above.
    """

    # 3. Run your LLM (example with OpenAI)
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content

    # 4. Learn from outcome
    alma.learn(
        agent=agent,
        task=task,
        outcome="success",
        strategy_used=result[:200]  # First 200 chars as summary
    )

    return result

# Run it
result = await run_task("Test the login form", "qa-agent")
```

## Next Steps

- [Configuration Guide](configuration.md) - Configure storage backends and agent scopes
- [Memory Types](../guides/memory-types.md) - Learn about the five memory types
- [Multi-Agent Sharing](../guides/multi-agent-sharing.md) - Share knowledge between agents
