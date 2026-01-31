# Why I Built ALMA

## The Problem

I was building AI agents for automated testing - Helena for frontend QA, Victor for backend verification. They worked great... until they didn't.

**The same mistakes kept happening:**

- Helena would use `sleep(5000)` for waits, causing flaky tests
- Victor would forget that the API uses JWT with 24-hour expiry
- Both agents would repeat failed strategies session after session

Every conversation started fresh. No memory. No learning. Just an expensive LLM making the same mistakes I'd already corrected.

## The Search for Solutions

I tried **Mem0**. It stores memories, but:

- No way to scope what an agent can learn
- No anti-pattern tracking ("don't do this")
- No multi-agent knowledge sharing
- Helena could "learn" database queries she'd never use

I looked at **LangChain Memory**. It's for conversation context, not long-term learning. Different problem.

**Nothing fit.**

I needed:

1. **Scoped learning** - Helena learns testing, not backend logic
2. **Anti-patterns** - Remember what NOT to do
3. **Multi-agent sharing** - Senior agents share knowledge with juniors
4. **Workflow context** - Resume complex tasks after failures
5. **MCP integration** - Work natively with Claude Code

## Building ALMA

So I built it. **Agent Learning Memory Architecture.**

The core insight: AI agents don't need to modify their weights to "learn." They need **smart prompts** built from **relevant past experiences.**

```
+---------------------------------------------------------------------+
|  BEFORE TASK: Retrieve relevant memories                            |
|  +-- "Last time you tested forms, incremental validation worked"    |
|  +-- "User prefers verbose output"                                  |
|  +-- "Don't use sleep() - causes flaky tests"                       |
+---------------------------------------------------------------------+
|  DURING TASK: Agent executes with injected knowledge                |
+---------------------------------------------------------------------+
|  AFTER TASK: Learn from outcome                                     |
|  +-- Success? -> New heuristic. Failure? -> Anti-pattern.           |
+---------------------------------------------------------------------+
```

**No fine-tuning. No model changes. Just smarter prompts.**

## What Makes ALMA Different

### Scoped Learning

Helena can only learn what she needs:

```yaml
agents:
  helena:
    can_learn:
      - testing_strategies
      - selector_patterns
    cannot_learn:
      - backend_logic
      - database_queries
```

### Anti-Pattern Tracking

When something fails, record WHY and WHAT TO DO INSTEAD:

```python
alma.add_anti_pattern(
    agent="helena",
    pattern="Using sleep() for async waits",
    why_bad="Causes flaky tests, wastes time",
    better_alternative="Use explicit waits with conditions"
)
```

### Multi-Agent Sharing

Senior agents teach juniors:

```yaml
agents:
  senior_architect:
    share_with: [junior_dev, qa_agent]

  junior_dev:
    inherit_from: [senior_architect]
```

### Workflow Context (v0.6.0)

Complex tasks can checkpoint and resume:

```python
# Save state mid-workflow
alma.checkpoint(workflow_id="deploy-v2", state=current_state)

# Resume after failure
alma.resume(workflow_id="deploy-v2")
```

## The Result

Helena and Victor now:

- Remember what worked across sessions
- Avoid strategies that failed before
- Share knowledge with each other
- Pick up complex tasks where they left off

They're not smarter. They're **better informed.**

## Open Source

ALMA is MIT licensed. Use it, modify it, contribute to it.

- **GitHub:** [github.com/RBKunnela/ALMA-memory](https://github.com/RBKunnela/ALMA-memory)
- **PyPI:** `pip install alma-memory`
- **npm:** `@rbkunnela/alma-memory`

---

*If your AI agents keep making the same mistakes, they don't have a memory problem. They have an ALMA problem.*

[Get Started](../getting-started/installation.md){ .md-button .md-button--primary }
