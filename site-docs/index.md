---
hide:
  - navigation
  - toc
---

<!-- ALMA Landing Page — design system per docs/design-system/DESIGN-alma.md -->

<section class="alma-hero-section" markdown="0">
  <div class="alma-aurora" aria-hidden="true">
    <span class="alma-aurora-orb alma-aurora-orb--1"></span>
    <span class="alma-aurora-orb alma-aurora-orb--2"></span>
    <span class="alma-aurora-orb alma-aurora-orb--3"></span>
  </div>

  <div class="alma-hero-inner">
    <span class="alma-eyebrow alma-reveal">Agent Learning Memory Architecture</span>

    <h1 class="alma-display-hero alma-reveal" style="--delay: 0.08s">
      Your AI forgets everything.<br>
      <span class="alma-display-accent">ALMA fixes that.</span>
    </h1>

    <p class="alma-display-lead alma-reveal" style="--delay: 0.16s">
      Give any AI agent permanent memory that learns and improves over time.
      One memory layer. Every AI. Never start from zero.
    </p>

    <div class="alma-hero-ctas alma-reveal" style="--delay: 0.24s">
      <a class="alma-btn alma-btn--primary" href="getting-started/installation/">
        Install ALMA
      </a>
      <a class="alma-btn alma-btn--ghost" href="https://github.com/RBKunnela/ALMA-memory" target="_blank" rel="noopener">
        Star on GitHub ↗
      </a>
    </div>

    <div class="alma-hero-install alma-reveal" style="--delay: 0.32s">
      <code class="alma-cli-line">
        <span class="alma-cli-prompt">$</span>
        pip install alma-memory
      </code>
      <span class="alma-cli-caption">5 minutes to persistent memory · Free forever on SQLite</span>
    </div>
  </div>
</section>


<section class="alma-benchmark-section alma-reveal" markdown="0">
  <div class="alma-benchmark-hero">
    <div class="alma-benchmark-label">Benchmarked on LongMemEval · ICLR 2025 · 500 questions</div>
    <div class="alma-benchmark-value" data-counter="0.964" data-decimals="3">0.000</div>
    <div class="alma-benchmark-caption">R@5 — #1 on the open-source leaderboard</div>
  </div>
  <p class="alma-benchmark-note">
    ALMA doesn't make your AI remember. <strong>It makes your AI learn.</strong>
    Built-in memory stores preferences. ALMA tracks outcomes — what worked, what failed,
    and why — and auto-creates reusable strategies from real experience.
  </p>
</section>


<section class="alma-problem-section" markdown="0">
  <h2 class="alma-section-heading alma-reveal">The problem with every AI tool you already use</h2>
  <div class="alma-problem-grid">

    <div class="alma-problem-card alma-reveal" style="--delay: 0.05s">
      <div class="alma-problem-number">01</div>
      <h3>Fragmented context</h3>
      <p>Claude doesn't know what ChatGPT learned. Every new AI restart starts from zero.</p>
    </div>

    <div class="alma-problem-card alma-reveal" style="--delay: 0.1s">
      <div class="alma-problem-number">02</div>
      <h3>Notepads, not learning</h3>
      <p>Built-in memory remembers facts. It doesn't remember what worked vs what failed.</p>
    </div>

    <div class="alma-problem-card alma-reveal" style="--delay: 0.15s">
      <div class="alma-problem-number">03</div>
      <h3>Locked in walled gardens</h3>
      <p>Your working context lives on someone else's servers, fragmented across accounts.</p>
    </div>

  </div>
</section>


<section class="alma-capabilities-section" markdown="0">
  <h2 class="alma-section-heading alma-reveal">What ALMA actually does</h2>
  <p class="alma-section-lead alma-reveal" style="--delay: 0.05s">
    Six capabilities that turn memory into a learning system. Compose them,
    or use them independently.
  </p>

  <div class="alma-capability-grid">

    <a class="alma-capability-card alma-reveal" style="--delay: 0.05s" href="guides/memory-types/">
      <div class="alma-capability-icon">🧠</div>
      <h3>Learn from outcomes</h3>
      <p>Agents track what strategies worked and failed. Success auto-compounds into heuristics; failures become named anti-patterns with <code>why_bad</code> and <code>better_alternative</code>.</p>
    </a>

    <a class="alma-capability-card alma-reveal" style="--delay: 0.1s" href="guides/multi-agent-sharing/">
      <div class="alma-capability-icon">🔗</div>
      <h3>Multi-agent sharing</h3>
      <p>Hierarchical knowledge flows with <code>inherit_from</code> and <code>share_with</code> scopes. Junior agents inherit from senior agents — no copy-paste, no drift.</p>
    </a>

    <a class="alma-capability-card alma-reveal" style="--delay: 0.15s" href="guides/veritas-trust-layer/">
      <div class="alma-capability-icon">🛡️</div>
      <h3>Veritas trust layer</h3>
      <p>Built-in trust scoring. Contradictions caught before agents act on bad data. The moment silent fallbacks stop being invisible.</p>
    </a>

    <a class="alma-capability-card alma-reveal" style="--delay: 0.05s" href="guides/storage-backends/">
      <div class="alma-capability-icon">💾</div>
      <h3>Your data, your database</h3>
      <p>SQLite, PostgreSQL, Qdrant, Pinecone, Chroma, Azure Cosmos. Your infra, your control, your backup strategy.</p>
    </a>

    <a class="alma-capability-card alma-reveal" style="--delay: 0.1s" href="guides/mcp-integration/">
      <div class="alma-capability-icon">🔌</div>
      <h3>MCP-native</h3>
      <p>Plug into any MCP-compatible AI. Your working context becomes portable across Claude, ChatGPT, Gemini, and whatever ships next.</p>
    </a>

    <a class="alma-capability-card alma-reveal" style="--delay: 0.15s" href="guides/events-webhooks/">
      <div class="alma-capability-icon">📡</div>
      <h3>Event system</h3>
      <p>Webhooks + callbacks on every memory event. Wire ALMA into your existing observability, audit, or automation stack.</p>
    </a>

  </div>
</section>


<section class="alma-compare-section" markdown="0">
  <h2 class="alma-section-heading alma-reveal">Built-in memory is a notepad. ALMA is a learning system.</h2>

  <div class="alma-compare-table alma-reveal" style="--delay: 0.05s">

| | Built-in memory <br><span class="alma-compare-meta">Claude · ChatGPT · OpenClaw</span> | **ALMA** |
|---|---|---|
| **What it stores** | Facts and preferences | Outcomes — what strategies worked, failed, and why |
| **Does it learn?** | No. Remembers what you told it. | **Yes.** After 3+ similar outcomes, auto-creates reusable strategies. |
| **Warns you?** | No | **Yes.** Anti-patterns track what NOT to do, with `why_bad` + `better_alternative`. |
| **Cross-platform?** | No | **Yes.** One memory layer across every AI tool. |
| **Multi-agent?** | No | **Yes.** Junior agents inherit from senior agents. |
| **Scoring?** | Basic relevance | 4-factor: similarity + recency + success rate + confidence |
| **Lifecycle?** | Grows until you delete | Automatic decay · compression · consolidation · archival |
| **Benchmarked?** | Not measured | **R@5 = 0.964** on LongMemEval |
| **Your data?** | Their servers | **Your database.** SQLite · Postgres · Qdrant |

  </div>

  <div class="alma-compare-ctas alma-reveal" style="--delay: 0.1s">
    <a class="alma-btn alma-btn--ghost" href="comparison/mem0-vs-alma/">ALMA vs Mem0 →</a>
    <a class="alma-btn alma-btn--ghost" href="comparison/langchain-memory-vs-alma/">ALMA vs LangChain →</a>
  </div>
</section>


<section class="alma-quickstart-section" markdown="0">
  <h2 class="alma-section-heading alma-reveal">Three lines to persistent memory</h2>
  <p class="alma-section-lead alma-reveal" style="--delay: 0.05s">
    No complex setup. No separate service. Works offline on SQLite by default.
  </p>

  <div class="alma-code-showcase alma-reveal" style="--delay: 0.1s">

```python
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")

# Before task: What strategies worked for this type of problem?
memories = alma.retrieve(task="Deploy auth service", agent="backend-dev")

# After task: Record what happened so next time is better
alma.learn(agent="backend-dev", task="Deploy auth service",
           outcome="success", strategy_used="Blue-green deployment")
```

  </div>

  <p class="alma-code-caption alma-reveal" style="--delay: 0.15s">
    Next time the backend agent deploys — on Claude, ChatGPT, or any platform —
    it already knows blue-green works and rolling updates don't.
  </p>
</section>


<section class="alma-cta-section alma-reveal" markdown="0">
  <div class="alma-cta-inner">
    <h2>Memory has replaced models as the moat of 2026.</h2>
    <p>Free forever on SQLite. MIT-licensed. Anonymized quality telemetry keeps ALMA honest and improving.</p>
    <div class="alma-hero-ctas">
      <a class="alma-btn alma-btn--primary" href="getting-started/installation/">Get Started</a>
      <a class="alma-btn alma-btn--ghost" href="https://github.com/RBKunnela/ALMA-memory" target="_blank" rel="noopener">View on GitHub ↗</a>
    </div>
  </div>
</section>


## How it works

Every time your agent runs, ALMA retrieves what worked before and learns from new outcomes. No manual prompt engineering. No copy-pasting from past conversations. **The memory compounds automatically.**

![How ALMA Works — The Learning Cycle](assets/diagrams/alma-how-it-works.jpeg){ .alma-reveal }

---

## Language SDKs

=== "Python"

    ```bash
    pip install alma-memory
    ```

    ```python
    from alma import ALMA

    # Initialize
    alma = ALMA.from_config(".alma/config.yaml")

    # Before task: Get relevant memories
    memories = alma.retrieve(
        task="Test the login form validation",
        agent="qa_tester",
        top_k=5
    )

    # Inject into your prompt
    prompt = f"""
    ## Your Task
    Test the login form validation

    ## Knowledge from Past Runs
    {memories.to_prompt()}
    """

    # After task: Learn from outcome
    alma.learn(
        agent="qa_tester",
        task="Test login form",
        outcome="success",
        strategy_used="Tested empty fields, invalid email, valid submission",
    )
    ```

=== "TypeScript"

    ```bash
    npm install @rbkunnela/alma-memory
    ```

    ```typescript
    import { ALMA } from '@rbkunnela/alma-memory';

    const alma = new ALMA({
      baseUrl: 'http://localhost:8765',
      projectId: 'my-project'
    });

    const memories = await alma.retrieve({
      query: 'authentication flow',
      agent: 'dev-agent',
      topK: 5
    });

    await alma.learn({
      agent: 'dev-agent',
      task: 'Implement OAuth',
      outcome: 'success',
      strategy_used: 'Used Passport.js with JWT strategy'
    });
    ```
