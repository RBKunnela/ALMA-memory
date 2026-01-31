# Multi-Agent Memory Sharing Guide

ALMA supports hierarchical memory sharing between agents, enabling knowledge transfer without duplicating data.

## Overview

Multi-agent sharing allows:
- **Senior agents** to share knowledge with juniors
- **Specialized agents** to contribute domain expertise
- **Team leads** to propagate best practices
- **Cross-functional teams** to collaborate

## Configuration

### share_with

Makes your memories readable by other agents.

```yaml
agents:
  senior_developer:
    domain: coding
    can_learn:
      - architecture
      - best_practices
      - design_patterns
    share_with:
      - junior_developer
      - qa_engineer
```

### inherit_from

Allows reading memories from other agents.

```yaml
agents:
  junior_developer:
    domain: coding
    can_learn:
      - coding_patterns
      - debugging
    inherit_from:
      - senior_developer
      - tech_lead
```

## Complete Example

```yaml
alma:
  project_id: "my-team"
  storage: sqlite

agents:
  # Tech lead shares with everyone
  tech_lead:
    domain: coding
    can_learn:
      - architecture
      - system_design
      - team_standards
    share_with:
      - senior_developer
      - junior_developer
      - qa_engineer
      - devops_engineer

  # Senior dev shares with juniors, learns from lead
  senior_developer:
    domain: coding
    can_learn:
      - implementation_patterns
      - code_review
    inherit_from:
      - tech_lead
    share_with:
      - junior_developer

  # Junior learns from senior and lead
  junior_developer:
    domain: coding
    can_learn:
      - basic_patterns
      - debugging
    inherit_from:
      - senior_developer
      - tech_lead

  # QA learns from tech lead
  qa_engineer:
    domain: testing
    can_learn:
      - testing_strategies
      - bug_patterns
    inherit_from:
      - tech_lead
```

## Usage in Code

### Retrieving Shared Memories

```python
from alma import ALMA

alma = ALMA.from_config(".alma/config.yaml")

# Junior developer retrieves memories
# This includes their own + senior_developer + tech_lead memories
memories = alma.retrieve(
    task="Implement user authentication",
    agent="junior_developer",
    include_shared=True,  # Enable shared memory retrieval
    top_k=10
)

# Access all heuristics (own + inherited)
for heuristic in memories.heuristics:
    if heuristic.metadata.get('shared_from'):
        # This memory came from another agent
        origin = heuristic.metadata['shared_from']
        print(f"[From {origin}] {heuristic.strategy}")
    else:
        # This is the agent's own memory
        print(f"[Own] {heuristic.strategy}")
```

### Checking Memory Origin

```python
# Filter to only inherited memories
inherited = [
    h for h in memories.heuristics
    if h.metadata.get('shared_from')
]

# Filter to only own memories
own = [
    h for h in memories.heuristics
    if not h.metadata.get('shared_from')
]

print(f"Own memories: {len(own)}")
print(f"Inherited memories: {len(inherited)}")
```

### Learning Still Respects Scope

When an agent learns, the memory is stored under **their own** agent ID, not the inherited agent's:

```python
# Junior learns from their experience
alma.learn(
    agent="junior_developer",
    task="Fixed null pointer bug",
    outcome="success",
    strategy_used="Added null checks before dereferencing"
)
# This memory belongs to junior_developer
# It will be shared with NO ONE (junior has no share_with)
```

## How It Works

### Retrieval Flow

1. Agent requests memories with `include_shared=True`
2. ALMA looks up agent's scope for `inherit_from` list
3. For each inherited agent, ALMA checks their `share_with` includes requesting agent
4. Valid inherited agents are added to the query
5. Single optimized query retrieves from all agents
6. Results are marked with `shared_from` metadata

### Permission Model

Both conditions must be met for sharing:
1. **Receiver** lists source in `inherit_from`
2. **Source** lists receiver in `share_with`

This prevents unauthorized access:

```yaml
agents:
  alice:
    share_with: [bob]  # Alice shares with Bob

  bob:
    inherit_from: [alice]  # Bob can read Alice's memories ✓

  charlie:
    inherit_from: [alice]  # Charlie tries to read Alice
    # DENIED - Alice doesn't have charlie in share_with
```

### Optimized Queries

ALMA uses optimized batch queries for multi-agent retrieval:

```python
# Instead of N separate queries
for agent in agents_to_query:
    results.extend(storage.get_heuristics(agent=agent))

# ALMA uses one query with OR conditions
results = storage.get_heuristics_for_agents(
    agents=agents_to_query,  # Single query for all
    ...
)
```

## Patterns

### Expertise Hierarchy

```yaml
agents:
  domain_expert:
    can_learn: [deep_domain_knowledge]
    share_with: [generalist_1, generalist_2]

  generalist_1:
    inherit_from: [domain_expert]

  generalist_2:
    inherit_from: [domain_expert]
```

### Team Structure

```yaml
agents:
  team_lead:
    share_with: [member_1, member_2, member_3]

  member_1:
    inherit_from: [team_lead]
    share_with: [member_2, member_3]  # Peer sharing

  member_2:
    inherit_from: [team_lead, member_1]
    share_with: [member_1, member_3]
```

### Mentor-Mentee

```yaml
agents:
  mentor:
    share_with: [mentee]

  mentee:
    inherit_from: [mentor]
    # Mentee doesn't share back - one-way knowledge transfer
```

## TypeScript SDK

```typescript
import { ALMA } from 'alma-memory';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-team'
});

// Retrieve with shared memories
const memories = await alma.retrieve({
  query: 'implement authentication',
  agent: 'junior_developer',
  topK: 10,
  includeShared: true
});

// Check origin
memories.heuristics.forEach(h => {
  if (h.metadata?.shared_from) {
    console.log(`Learned from ${h.metadata.shared_from}: ${h.strategy}`);
  }
});
```

## Best Practices

1. **Keep hierarchies shallow** - Deep inheritance chains can be confusing
2. **Be explicit about sharing** - Both share_with and inherit_from are required
3. **Don't over-share** - Only share what's relevant to the receiver
4. **Monitor inherited memory counts** - Too many can dilute relevance
5. **Use meaningful agent names** - Makes shared_from metadata readable

## Troubleshooting

### No inherited memories appearing

Check:
1. `include_shared: true` is set in retrieve call
2. Source agent has receiver in `share_with`
3. Receiver agent has source in `inherit_from`
4. Source agent actually has memories

### Too many inherited memories

- Reduce `top_k` value
- Be more specific in your query
- Review if all inherited agents are necessary

### Performance concerns

Multi-agent queries are optimized, but with many agents:
- Consider caching results
- Use appropriate `top_k` limits
- Review if full inheritance chain is needed
