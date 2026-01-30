# ALMA - Video Talking Points

> For Professor's video about ALMA

## One-Line Pitch

**"ALMA is memory that learns, not just remembers - it helps AI agents get better at their jobs over time."**

---

## The Problem (30 seconds)

Current AI memory systems like Mem0 store facts:
- "User prefers Python"
- "User is vegetarian"

But they do not learn strategies:
- "Incremental validation works for complex forms"
- "Never use sleep() for async waits - causes flaky tests"

AI agents repeat the same mistakes because they cannot learn from experience.

---

## The Solution (60 seconds)

ALMA has 5 memory types:

| Type | What It Stores | Example |
|------|----------------|---------|
| **Heuristics** | Strategies that work | "For forms with 5+ fields, test incrementally" |
| **Anti-patterns** | What NOT to do | "Don't use sleep() - causes flaky tests" |
| **Outcomes** | Task results | "Login test succeeded with JWT strategy" |
| **Preferences** | User constraints | "User prefers verbose output" |
| **Domain Knowledge** | Accumulated facts | "Login uses OAuth 2.0" |

Key differentiator: **Scoped Learning**
- Helena (frontend agent) cannot learn backend patterns
- Victor (backend agent) cannot learn UI patterns
- Prevents knowledge bleed and confusion

---

## Demo Script (2 minutes)

```bash
# Run the interactive demo
cd ALMA-memory
python demo.py
```

The demo shows:
1. Helena running 5 test tasks
2. Two successes with "incremental validation" → Creates HEURISTIC
3. Two failures with "sleep()" → Creates ANTI-PATTERN
4. New task retrieves relevant memories
5. Prompt injection with learned knowledge

---

## Why It Matters (30 seconds)

**For AI Agent Developers:**
- Agents that improve over time without retraining
- Reduced hallucination through grounded experience
- Scoped learning prevents confusion

**Compared to Mem0:**
- Mem0 = Memory for chatbots (user preferences)
- ALMA = Learning for agents (operational intelligence)

---

## Call to Action

1. **Star the repo**: https://github.com/RBKunnela/ALMA-memory
2. **Try the demo**: `pip install alma-memory && python -c "from alma import ALMA; print('ALMA works!')"`
3. **Contribute**: Look for "good-first-issue" labels
4. **Join waitlist**: [Coming soon - ALMA Cloud]

---

## Technical Highlights for Developer Audience

```python
# Automatic learning from conversations
from alma.extraction import AutoLearner

auto_learner = AutoLearner(alma)
results = auto_learner.learn_from_conversation(messages, agent="helena")
# Extracts heuristics, anti-patterns, knowledge automatically

# Graph memory for entity relationships
from alma.graph import create_graph_store, EntityExtractor

graph = create_graph_store("neo4j", ...)
entities, relationships = extractor.extract("Alice from Acme reviewed Bob's PR")
# Captures: Alice -[WORKS_AT]-> Acme, Alice -[REVIEWED]-> PR, Bob -[SUBMITTED]-> PR

# Scoped learning (unique to ALMA)
alma.learn(
    agent="helena",  # Frontend specialist
    task="Test login form",
    outcome="success",
    strategy="Incremental validation",
)
# Helena cannot learn backend patterns - enforced by scope
```

---

## FAQ

**Q: How is this different from RAG?**
A: RAG retrieves documents. ALMA learns strategies from outcomes. RAG is passive, ALMA actively improves.

**Q: How is this different from Mem0?**
A: Mem0 stores user facts ("likes Python"). ALMA learns operational knowledge ("incremental validation works"). Also, ALMA has scoped learning and anti-patterns.

**Q: Does it require fine-tuning?**
A: No. ALMA works through prompt injection - no model weight updates needed.

**Q: What LLMs does it support?**
A: Any LLM for the agent itself. For fact extraction: OpenAI, Anthropic, or rule-based (free/offline).

---

## Project Stats (Update Before Video)

- GitHub: https://github.com/RBKunnela/ALMA-memory
- PyPI: `pip install alma-memory`
- License: MIT (fully open source)
- Status: Active development, seeking contributors

---

## Contact

- Maintainer: Renata Baldissara-Kunnela
- GitHub: @RBKunnela
- LinkedIn: [Your LinkedIn]
- Email: renata@jurevo.io
