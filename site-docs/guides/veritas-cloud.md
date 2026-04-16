# Veritas Cloud — Coming Soon

!!! info "Early Access"
    Veritas Cloud is being built with design partners. If multi-agent trust is a pain point for your team, [get in touch](mailto:dev@friendlyai.fi?subject=Veritas%20Cloud%20Early%20Access).

## What's Free vs What's Coming

### ALMA + Veritas (Free, Open Source)

Everything you need for single-instance trust:

- **Trust scoring** — per-agent trust profiles, 5 behavioral dimensions, trust decay
- **Verified retrieval** — two-stage verification, 4 statuses, conflict detection
- **Anti-pattern memory** — agents remember what failed and why
- **Retrieval feedback loop** — memories agents use get scored higher
- **7 storage backends**, 4 graph backends, 22 MCP tools

```bash
pip install alma-memory
```

### Veritas Cloud (Pro — Coming Soon)

When you outgrow single-instance trust — dozens of agents, multiple deployments, enterprise clients:

| Feature | What It Solves |
|---------|---------------|
| **Real-time conflict prevention** | Stop Agent B before it acts on data Agent A already invalidated. Not after — before. |
| **Shared trust graph** | One source of truth across all your agent deployments. Trust scores, provenance, conflict history — unified. |
| **Trust dashboard** | Conflicts/week, trust trends per agent, resolution success rate. One screen, whole fleet. |
| **Provenance chain API** | Full audit trail: who created this memory, who verified it, who used it. SOC2-ready. |
| **Monthly value report** | Automated: "Veritas prevented X conflicts, resolved Y automatically, estimated $Z saved." |

## Does This Sound Like Your Team?

If you answer yes to any of these, Veritas Cloud is being built for you.

### 1. "Our agents contradict each other and we find out when a customer complains."

Veritas Cloud catches conflicts in real-time, before agents act on contradicting data. The open-source version detects contradictions on retrieval. The cloud version **prevents** them before the agent acts.

### 2. "We can't prove to our clients that our AI agents are making trustworthy decisions."

The provenance chain API gives you a full audit trail. Show clients exactly how every decision was made — which memories were used, which were verified, which were flagged.

### 3. "We run 50+ agent workflows and have no idea how many conflicts happen per week."

The trust dashboard shows trust violations, resolution rates, and memory accuracy across your entire fleet. No more flying blind.

### 4. "Trust built in one workflow doesn't carry over to another."

Shared trust graph — agent trust scores travel with the agent across deployments. An agent that proved itself in workflow A starts trusted in workflow B.

### 5. "Enterprise clients are asking about AI compliance and audit trails."

SOC2-ready audit exports, per-tenant trust isolation, and SLA guarantees. When procurement asks "show me your AI decision audit trail" — one API call.

## Early Access

We're building Veritas Cloud with design partners — teams running multi-agent systems in production who feel these pain points today.

**What you get as a design partner:**

- Direct access to the team building Veritas Cloud
- Input on which features ship first
- Team tier pricing ($149/mo) with hands-on onboarding
- 30-day pilot with full refund if the value isn't obvious

**What we ask in return:**

- Tell us about your agent setup (how many agents, what they do, how they coordinate)
- Share which of the 5 questions above resonates most
- Give honest feedback on whether the product delivers value

[Request Early Access](mailto:dev@friendlyai.fi?subject=Veritas%20Cloud%20Early%20Access&body=Hi%2C%20I'm%20interested%20in%20Veritas%20Cloud.%0A%0AMy%20team%20runs%20____%20agent%20workflows.%0AThe%20biggest%20trust%20challenge%20we%20face%20is%3A%20____){ .md-button .md-button--primary }
