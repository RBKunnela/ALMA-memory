# TypeScript API Reference

## Installation

```bash
echo "@rbkunnela:registry=https://npm.pkg.github.com" >> ~/.npmrc
npm install @rbkunnela/alma-memory
```

## ALMA Class

```typescript
import { ALMA } from '@rbkunnela/alma-memory';

const alma = new ALMA({
  baseUrl: 'http://localhost:8765',
  projectId: 'my-project',
  timeout: 30000,           // Optional
  headers: { 'X-Custom': 'value' },  // Optional
  retry: {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000
  }
});
```

## Methods

### `retrieve(options)`
### `learn(options)`
### `addPreference(options)`
### `addKnowledge(options)`
### `forget(options)`
### `stats(agent?)`
### `health()`

### Workflow Methods (v0.6.0)
### `consolidate(options)`
### `checkpoint(options)`
### `resume(options)`
### `mergeStates(options)`
### `workflowLearn(options)`
### `linkArtifact(options)`
### `getArtifacts(options)`
### `cleanupCheckpoints(options)`
### `retrieveScoped(options)`

See [TypeScript SDK Guide](../guides/typescript-sdk.md) for detailed examples.
