# Veritas Trust Layer

ALMA includes the **Veritas trust framework** — built-in trust scoring and memory verification so your agents don't act on bad data.

When you run multiple agents, memories can conflict. Agent A says "lead is disqualified." Agent B says "lead is engaged." Which one does your agent trust? Veritas answers that question.

## Trust Scoring

Every agent builds a trust profile over time. Memories from trusted agents rank higher automatically.

```python
from alma.retrieval.trust_scoring import TrustAwareScorer, AgentTrustProfile

scorer = TrustAwareScorer()

# Senior agent — high trust
scorer.set_trust_profile(AgentTrustProfile(
    agent_id="senior-dev",
    sessions_completed=50,
    total_actions=200,
    total_violations=2,
    consecutive_clean_sessions=15,
))

# New agent — lower trust
scorer.set_trust_profile(AgentTrustProfile(
    agent_id="new-intern-bot",
    sessions_completed=3,
    total_actions=10,
    total_violations=4,
))

# Score memories — senior-dev's rank higher automatically
scored = scorer.score_with_trust(memories, agent="senior-dev")
```

### 5 Behavioral Dimensions

Trust scores factor in 5 behavioral dimensions per agent:

| Dimension | What It Measures |
|-----------|-----------------|
| **verification_before_claim** | Does the agent verify facts before asserting them? |
| **loud_failure** | Does the agent report failures openly instead of hiding them? |
| **honest_uncertainty** | Does the agent express uncertainty when it's unsure? |
| **paper_trail** | Does the agent maintain an audit trail? |
| **diligent_execution** | Does the agent execute tasks thoroughly? |

### Trust Decay

Trust decays over time if an agent goes inactive (30-day half-life). An agent that hasn't been validated in 60 days drops to ~50% of its peak trust. This prevents stale agents from being trusted blindly.

### Trust Levels

| Level | Score Range | Meaning |
|-------|------------|---------|
| **FULL** | 1.0 | Completely trusted |
| **HIGH** | 0.85+ | Very reliable |
| **GOOD** | 0.7+ | Generally trustworthy |
| **MODERATE** | 0.5+ | Default for new agents |
| **LOW** | 0.4+ | Needs improvement |
| **MINIMAL** | 0.2+ | Mostly untrusted |
| **UNTRUSTED** | 0.0 | Do not trust |

## Verified Retrieval

For high-stakes decisions, ALMA can verify memories before your agent uses them.

```python
from alma.retrieval.verification import VerifiedRetriever, VerificationConfig

retriever = VerifiedRetriever(
    retrieval_engine=alma.retrieval_engine,
    llm_client=my_llm,  # Optional — works without LLM too
    config=VerificationConfig(
        enabled=True,
        default_method="cross_verify",
        confidence_threshold=0.7,
    )
)

results = retriever.retrieve_verified(
    query="What's the status of lead #1234?",
    agent="voice-agent",
    project_id="my-project",
)

# Only use memories you can trust
for memory in results.verified:
    print(f"Safe: {memory.memory}")

for memory in results.contradicted:
    print(f"CONFLICT: {memory.verification.reason}")

print(results.summary())
# {'verified': 3, 'uncertain': 1, 'contradicted': 1,
#  'usable_ratio': 0.8, 'verification_time_ms': 45}
```

### Verification Statuses

Every retrieved memory gets a verification status:

| Status | Meaning | Should your agent use it? |
|--------|---------|--------------------------|
| **VERIFIED** | Confirmed accurate against ground truth or other memories | Yes |
| **UNCERTAIN** | No conflicting evidence, but unconfirmed | Yes, with caution |
| **CONTRADICTED** | Conflicts with other memories detected | No — review needed |
| **UNVERIFIABLE** | Can't be verified (no other sources) | Use your judgment |

### Verification Methods

| Method | How It Works | Requires LLM? |
|--------|-------------|----------------|
| **Ground Truth** | Verify against authoritative sources you provide | Yes |
| **Cross-Verify** | Check memory against other memories for contradictions | Yes |
| **Confidence** | Score-based fallback using existing confidence values | No |

## Why This Matters

Without trust verification, your voice agent might call a lead that your email agent already disqualified — because both agents stored conflicting memories about the same person. With Veritas:

1. The voice agent retrieves memories about the lead
2. Veritas cross-verifies and detects the contradiction
3. The contradicted memory is flagged — the agent sees `CONTRADICTED` status
4. The agent skips the call or escalates to a human

**127 tests** cover the trust scoring and verification modules. All passing.
