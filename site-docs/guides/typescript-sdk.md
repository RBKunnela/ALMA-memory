# TypeScript/JavaScript SDK Guide

The ALMA TypeScript SDK provides a type-safe client for interacting with ALMA Memory from JavaScript and TypeScript applications.

## Installation

```bash
npm install alma-memory
# or
yarn add alma-memory
```

## Quick Start

```typescript
import { ALMA } from 'alma-memory';

// Create client
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project'
});

// Retrieve memories
const memories = await alma.retrieve({
  query: 'authentication flow',
  agent: 'dev-agent',
  topK: 5
});

// Learn from outcomes
await alma.learn({
  agent: 'dev-agent',
  task: 'Implement OAuth',
  outcome: 'success',
  strategyUsed: 'Used passport.js middleware'
});
```

## Configuration

### Basic Configuration

```typescript
import { ALMA, ALMAConfig } from 'alma-memory';

const config: ALMAConfig = {
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project'
};

const alma = new ALMA(config);
```

### With Retry Configuration

```typescript
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  retry: {
    maxRetries: 3,
    baseDelay: 1000,      // 1 second
    maxDelay: 30000,      // 30 seconds
    backoffMultiplier: 2
  }
});
```

### With Custom Timeout

```typescript
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  timeout: 30000  // 30 seconds
});
```

## API Reference

### retrieve(options)

Get relevant memories for a task.

```typescript
import { RetrieveOptions, MemorySlice } from 'alma-memory';

const options: RetrieveOptions = {
  query: 'user authentication',
  agent: 'dev-agent',
  topK: 5,
  userId: 'user-123',         // Optional: for user preferences
  includeShared: true         // Optional: include inherited memories
};

const memories: MemorySlice = await alma.retrieve(options);

// Access different memory types
console.log('Heuristics:', memories.heuristics);
console.log('Outcomes:', memories.outcomes);
console.log('Preferences:', memories.preferences);
console.log('Knowledge:', memories.domainKnowledge);
console.log('Anti-patterns:', memories.antiPatterns);
```

### learn(options)

Record task outcome and optionally create a heuristic.

```typescript
import { LearnOptions, LearnResponse } from 'alma-memory';

const options: LearnOptions = {
  agent: 'dev-agent',
  task: 'Implement OAuth 2.0',
  outcome: 'success',           // 'success' | 'failure' | 'partial'
  strategyUsed: 'Used passport.js with Google provider',
  context: {                    // Optional metadata
    duration: '2h',
    complexity: 'medium'
  }
};

const result: LearnResponse = await alma.learn(options);
console.log('Outcome ID:', result.outcomeId);
console.log('Heuristic created:', result.heuristicCreated);
```

### addPreference(options)

Add a user preference.

```typescript
import { AddPreferenceOptions } from 'alma-memory';

await alma.addPreference({
  userId: 'user-123',
  category: 'code_style',
  preference: 'Use TypeScript strict mode',
  confidence: 0.9,           // Optional: 0.0 to 1.0
  source: 'explicit'         // Optional: 'explicit' | 'inferred'
});
```

### addKnowledge(options)

Add domain knowledge.

```typescript
import { AddKnowledgeOptions } from 'alma-memory';

await alma.addKnowledge({
  agent: 'dev-agent',
  domain: 'authentication',
  fact: 'API uses JWT with 24-hour expiry',
  confidence: 0.95,
  source: 'documentation'
});
```

### forget(options)

Remove old or low-confidence memories.

```typescript
import { ForgetOptions, ForgetResponse } from 'alma-memory';

const result: ForgetResponse = await alma.forget({
  agent: 'dev-agent',
  olderThanDays: 90,         // Remove memories older than 90 days
  belowConfidence: 0.3       // Remove memories with confidence < 0.3
});

console.log('Memories removed:', result.removedCount);
```

### stats()

Get memory statistics.

```typescript
const stats = await alma.stats();

console.log('Total memories:', stats.totalMemories);
console.log('By type:', stats.byType);
console.log('By agent:', stats.byAgent);
```

### health()

Check server health.

```typescript
const health = await alma.health();

console.log('Status:', health.status);  // 'healthy' | 'degraded' | 'unhealthy'
console.log('Version:', health.version);
```

## Memory Types

### Heuristic

Learned strategies that worked.

```typescript
interface Heuristic {
  id: string;
  agent: string;
  projectId: string;
  category: string;
  strategy: string;
  confidence: number;
  successCount: number;
  failureCount: number;
  metadata: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}
```

### Outcome

Results of completed tasks.

```typescript
interface Outcome {
  id: string;
  agent: string;
  projectId: string;
  task: string;
  outcome: 'success' | 'failure' | 'partial';
  strategyUsed: string;
  metadata: Record<string, unknown>;
  createdAt: Date;
}
```

### UserPreference

User-specific preferences.

```typescript
interface UserPreference {
  id: string;
  userId: string;
  projectId: string;
  category: string;
  preference: string;
  confidence: number;
  source: 'explicit' | 'inferred';
  createdAt: Date;
}
```

### DomainKnowledge

Accumulated facts about the domain.

```typescript
interface DomainKnowledge {
  id: string;
  agent: string;
  projectId: string;
  domain: string;
  fact: string;
  confidence: number;
  source: string;
  metadata: Record<string, unknown>;
  createdAt: Date;
}
```

### AntiPattern

Things to avoid.

```typescript
interface AntiPattern {
  id: string;
  agent: string;
  projectId: string;
  pattern: string;
  whyBad: string;
  betterAlternative: string;
  occurrences: number;
  metadata: Record<string, unknown>;
  createdAt: Date;
}
```

## Error Handling

The SDK provides typed errors for different failure modes.

```typescript
import {
  ALMAError,
  ConnectionError,
  ValidationError,
  NotFoundError,
  ScopeViolationError,
  TimeoutError,
  ServerError,
  isConnectionError,
  isValidationError
} from 'alma-memory';

try {
  await alma.retrieve({ query: 'test', agent: '' });
} catch (error) {
  if (isValidationError(error)) {
    console.error('Invalid input:', error.message);
  } else if (isConnectionError(error)) {
    console.error('Cannot reach server:', error.message);
  } else if (error instanceof ALMAError) {
    console.error('ALMA error:', error.message, error.code);
  } else {
    throw error;
  }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `ConnectionError` | Cannot connect to ALMA server |
| `ValidationError` | Invalid input parameters |
| `NotFoundError` | Requested resource not found |
| `ScopeViolationError` | Agent tried to learn outside scope |
| `TimeoutError` | Request timed out |
| `ServerError` | Internal server error |

## Framework Integration

### React

```typescript
import { useState, useEffect } from 'react';
import { ALMA, MemorySlice } from 'alma-memory';

const alma = new ALMA({
  baseUrl: process.env.NEXT_PUBLIC_ALMA_URL!,
  projectId: 'my-app'
});

function useMemories(query: string, agent: string) {
  const [memories, setMemories] = useState<MemorySlice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setLoading(true);
    alma.retrieve({ query, agent, topK: 5 })
      .then(setMemories)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [query, agent]);

  return { memories, loading, error };
}
```

### Node.js / Express

```typescript
import express from 'express';
import { ALMA } from 'alma-memory';

const app = express();
const alma = new ALMA({
  baseUrl: process.env.ALMA_URL!,
  projectId: 'api-server'
});

app.post('/api/chat', async (req, res) => {
  const { message, agent } = req.body;

  // Get relevant memories
  const memories = await alma.retrieve({
    query: message,
    agent,
    topK: 5
  });

  // Use memories in your LLM prompt
  const context = memories.heuristics
    .map(h => `Strategy: ${h.strategy}`)
    .join('\n');

  // ... rest of chat logic
});
```

### Next.js API Route

```typescript
// pages/api/memories.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { ALMA } from 'alma-memory';

const alma = new ALMA({
  baseUrl: process.env.ALMA_URL!,
  projectId: 'my-app'
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query, agent } = req.body;

  try {
    const memories = await alma.retrieve({ query, agent, topK: 5 });
    res.status(200).json(memories);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve memories' });
  }
}
```

## Starting the MCP Server

The TypeScript SDK communicates with ALMA via the MCP server. Start it with:

```bash
# Python MCP server
python -m alma.mcp --config .alma/config.yaml --port 8765
```

Or add to your package.json:

```json
{
  "scripts": {
    "alma:server": "python -m alma.mcp --config .alma/config.yaml --port 8765"
  }
}
```

## Best Practices

1. **Singleton Client**: Create one ALMA instance and reuse it
2. **Error Handling**: Always wrap calls in try-catch
3. **Appropriate topK**: Use 5-10 for most cases
4. **Batch Learning**: Record outcomes immediately after tasks
5. **Type Safety**: Use TypeScript for compile-time validation
