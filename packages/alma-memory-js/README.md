# ALMA Memory - TypeScript/JavaScript SDK

Official TypeScript/JavaScript SDK for [ALMA Memory](https://github.com/your-org/ALMA-memory) - Persistent memory for AI agents.

[![npm version](https://badge.fury.io/js/alma-memory.svg)](https://www.npmjs.com/package/alma-memory)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Type-Safe**: Full TypeScript support with comprehensive type definitions
- **Zero Dependencies**: Uses native `fetch` (Node.js 18+)
- **Dual Package**: Supports both CommonJS and ES Modules
- **Retry Logic**: Built-in exponential backoff for reliability
- **Error Handling**: Rich error types for precise error handling
- **JSDoc Comments**: Comprehensive documentation for IDE support

## Installation

```bash
npm install alma-memory
# or
yarn add alma-memory
# or
pnpm add alma-memory
```

## Requirements

- Node.js 18.0.0 or higher (for native `fetch` support)
- A running ALMA MCP server

## Quick Start

```typescript
import { ALMA } from 'alma-memory';

// Create client
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
});

// Retrieve relevant memories for a task
const result = await alma.retrieve({
  query: 'authentication flow',
  agent: 'dev-agent',
  topK: 5,
});

console.log(result.memories?.heuristics);
// Inject memories into your prompt
console.log(result.prompt_injection);
```

## API Reference

### Creating a Client

```typescript
import { ALMA, createClient } from 'alma-memory';

// Using constructor
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  timeout: 30000, // optional, default 30s
  headers: { 'X-Custom': 'value' }, // optional
  retry: {
    maxRetries: 3, // optional, default 3
    baseDelay: 1000, // optional, default 1000ms
    maxDelay: 10000, // optional, default 10000ms
  },
});

// Using helper function
const alma2 = createClient({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
});
```

### Retrieving Memories

Retrieve relevant memories for a task context:

```typescript
const result = await alma.retrieve({
  query: 'form validation testing', // Description of current task
  agent: 'qa-agent', // Agent name
  userId: 'user-123', // Optional: for user preference retrieval
  topK: 10, // Optional: max items per memory type (default: 5)
});

// Check success
if (result.success) {
  // Access memories by type
  console.log('Heuristics:', result.memories?.heuristics);
  console.log('Anti-patterns:', result.memories?.anti_patterns);
  console.log('User preferences:', result.memories?.preferences);
  console.log('Domain knowledge:', result.memories?.domain_knowledge);

  // Use formatted prompt injection
  const prompt = `${result.prompt_injection}\n\nNow perform the task...`;
}
```

### Learning from Outcomes

Record task outcomes to build the memory system:

```typescript
// Record successful outcome
await alma.learn({
  agent: 'dev-agent',
  task: 'Implement user authentication',
  taskType: 'authentication', // Optional: for grouping
  outcome: 'success',
  strategyUsed: 'Used JWT with refresh tokens and PKCE',
  durationMs: 45000, // Optional
});

// Record failure with error details
await alma.learn({
  agent: 'dev-agent',
  task: 'Deploy to production',
  outcome: 'failure',
  strategyUsed: 'Direct push to main branch',
  errorMessage: 'CI tests failed due to missing environment variables',
  feedback: 'Should use feature branches', // Optional user feedback
});
```

### Adding User Preferences

Store user preferences that persist across sessions:

```typescript
await alma.addPreference({
  userId: 'user-123',
  category: 'code_style', // or 'communication', 'workflow', etc.
  preference: 'Always use TypeScript strict mode',
  source: 'explicit_instruction', // or 'inferred_from_feedback'
});
```

### Adding Domain Knowledge

Store factual knowledge within an agent's scope:

```typescript
await alma.addKnowledge({
  agent: 'dev-agent',
  domain: 'authentication',
  fact: 'The API uses OAuth 2.0 with PKCE for mobile clients',
  source: 'documentation', // or 'code_analysis', 'user_stated'
});
```

### Pruning Memories

Clean up old or low-confidence memories:

```typescript
// Prune all agents
const result = await alma.forget({
  olderThanDays: 90, // Remove outcomes older than 90 days
  belowConfidence: 0.3, // Remove heuristics below 30% confidence
});
console.log(`Pruned ${result.pruned_count} memories`);

// Prune specific agent
await alma.forget({
  agent: 'test-agent',
  olderThanDays: 30,
});
```

### Getting Statistics

```typescript
// All agents
const allStats = await alma.stats();
console.log(`Total memories: ${allStats.stats?.total_count}`);
console.log(`Heuristics: ${allStats.stats?.heuristics_count}`);

// Specific agent
const agentStats = await alma.stats('dev-agent');
```

### Health Check

```typescript
const health = await alma.health();

if (health.status === 'healthy') {
  console.log(`Server healthy with ${health.total_memories} memories`);
  console.log(`Registered agents: ${health.registered_agents?.join(', ')}`);
}
```

## Error Handling

The SDK provides specific error types for different failure scenarios:

```typescript
import {
  ALMA,
  ALMAError,
  ConnectionError,
  ValidationError,
  TimeoutError,
  ServerError,
  ScopeViolationError,
  isConnectionError,
  isValidationError,
} from 'alma-memory';

try {
  await alma.retrieve({ query: 'test', agent: 'my-agent' });
} catch (error) {
  if (isConnectionError(error)) {
    console.error(`Cannot connect to server at ${error.url}`);
    // Show retry UI or check server status
  } else if (isValidationError(error)) {
    console.error(`Invalid input: ${error.field} - ${error.message}`);
    // Show validation error to user
  } else if (error instanceof TimeoutError) {
    console.error(`Request timed out after ${error.timeout}ms`);
    // Retry with longer timeout
  } else if (error instanceof ServerError) {
    console.error(`Server error: ${error.serverMessage}`);
    // Log for debugging
  }
}
```

### Error Types

| Error Class | Description | Properties |
|-------------|-------------|------------|
| `ALMAError` | Base error class | `code`, `statusCode`, `cause` |
| `ConnectionError` | Server unreachable | `url` |
| `ValidationError` | Invalid input | `field`, `value` |
| `TimeoutError` | Request timeout | `timeout` |
| `ServerError` | Server returned error | `serverMessage`, `rpcCode` |
| `ScopeViolationError` | Agent scope violation | `agent`, `domain` |
| `NotFoundError` | Resource not found | `resourceType`, `resourceId` |

## TypeScript Types

All types are exported for use in your applications:

```typescript
import type {
  // Configuration
  ALMAConfig,
  RetryConfig,

  // Memory types
  Heuristic,
  Outcome,
  UserPreference,
  DomainKnowledge,
  AntiPattern,
  MemorySlice,
  MemoryScope,
  MemoryType,

  // Request options
  RetrieveOptions,
  LearnOptions,
  AddPreferenceOptions,
  AddKnowledgeOptions,
  ForgetOptions,

  // Response types
  RetrieveResponse,
  LearnResponse,
  StatsResponse,
  HealthResponse,
} from 'alma-memory';
```

## Usage with AI Frameworks

### With LangChain

```typescript
import { ALMA } from 'alma-memory';
import { ChatOpenAI } from '@langchain/openai';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'langchain-project',
});

async function runWithMemory(task: string, agent: string) {
  // Retrieve relevant memories
  const memories = await alma.retrieve({ query: task, agent });

  // Create prompt with memories
  const prompt = `${memories.prompt_injection}

Task: ${task}

Please complete the task using the context above.`;

  // Run LLM
  const llm = new ChatOpenAI();
  const result = await llm.invoke(prompt);

  // Learn from outcome
  await alma.learn({
    agent,
    task,
    outcome: 'success',
    strategyUsed: 'Used memory-augmented prompt',
  });

  return result;
}
```

### With Vercel AI SDK

```typescript
import { ALMA } from 'alma-memory';
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'vercel-ai-project',
});

async function chat(userMessage: string, agent: string) {
  const memories = await alma.retrieve({
    query: userMessage,
    agent,
  });

  const { text } = await generateText({
    model: openai('gpt-4'),
    system: memories.prompt_injection,
    prompt: userMessage,
  });

  return text;
}
```

## Configuration

### Environment Variables

You can configure the client using environment variables:

```typescript
const alma = new ALMA({
  baseUrl: process.env.ALMA_BASE_URL || 'http://localhost:8765',
  projectId: process.env.ALMA_PROJECT_ID || 'default',
  timeout: parseInt(process.env.ALMA_TIMEOUT || '30000'),
});
```

### Custom Headers

Add authentication or tracking headers:

```typescript
const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  headers: {
    'Authorization': `Bearer ${process.env.ALMA_TOKEN}`,
    'X-Request-ID': generateRequestId(),
  },
});
```

## Memory Types

ALMA supports five types of memories:

| Type | Description | Example |
|------|-------------|---------|
| **Heuristic** | Learned strategies that work | "When testing forms, validate happy path first" |
| **Outcome** | Raw task results (success/failure) | "OAuth implementation succeeded in 45s" |
| **UserPreference** | User constraints and preferences | "No emojis in documentation" |
| **DomainKnowledge** | Facts about the domain | "API uses JWT with 24h expiry" |
| **AntiPattern** | What NOT to do | "Don't use sleep() for async waits" |

## Running Tests

```bash
# Run tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

## Building

```bash
# Build all formats (CJS, ESM, types)
npm run build

# Clean build artifacts
npm run clean
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../../CONTRIBUTING.md) for details.

## Related

- [ALMA Memory Python SDK](../../) - The original Python implementation
- [ALMA MCP Server](../../alma/mcp/) - The MCP server this SDK connects to
- [Competitive Analysis](../../docs/architecture/competitive-analysis-mem0.md) - How ALMA compares to alternatives
