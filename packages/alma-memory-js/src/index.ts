/**
 * ALMA Memory TypeScript/JavaScript SDK
 *
 * Persistent memory for AI agents. This SDK provides a type-safe interface
 * for interacting with the ALMA Memory MCP server via HTTP.
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import { ALMA, MemorySlice, RetrieveOptions } from 'alma-memory';
 *
 * // Create client
 * const alma = new ALMA({
 *   baseUrl: 'http://localhost:8765',
 *   projectId: 'my-project'
 * });
 *
 * // Retrieve memories
 * const memories = await alma.retrieve({
 *   query: 'authentication flow',
 *   agent: 'dev-agent',
 *   topK: 5
 * });
 *
 * // Learn from outcomes
 * await alma.learn({
 *   agent: 'dev-agent',
 *   task: 'Implement OAuth',
 *   outcome: 'success',
 *   strategyUsed: 'Used passport.js middleware'
 * });
 *
 * // Add preferences
 * await alma.addPreference({
 *   userId: 'user-123',
 *   category: 'code_style',
 *   preference: 'Use TypeScript strict mode'
 * });
 *
 * // Add domain knowledge
 * await alma.addKnowledge({
 *   agent: 'dev-agent',
 *   domain: 'authentication',
 *   fact: 'API uses JWT with 24h expiry'
 * });
 * ```
 */

// Main client
export { ALMA, createClient } from './client';

// Types - Core memory types
export type {
  MemoryType,
  Heuristic,
  Outcome,
  UserPreference,
  DomainKnowledge,
  AntiPattern,
  MemorySlice,
  MemoryScope,
} from './types';

// Types - Configuration
export type {
  ALMAConfig,
  RetryConfig,
} from './types';

// Types - Operation options
export type {
  RetrieveOptions,
  LearnOptions,
  AddPreferenceOptions,
  AddKnowledgeOptions,
  ForgetOptions,
} from './types';

// Types - Response types
export type {
  RetrieveResponse,
  LearnResponse,
  AddPreferenceResponse,
  AddKnowledgeResponse,
  ForgetResponse,
  StatsResponse,
  HealthResponse,
  MemoryStats,
} from './types';

// Types - MCP protocol types
export type {
  MCPRequest,
  MCPResponse,
} from './types';

// Errors - Classes
export {
  ALMAError,
  ConnectionError,
  ValidationError,
  NotFoundError,
  ScopeViolationError,
  TimeoutError,
  ServerError,
} from './errors';

// Errors - Type guards
export {
  isALMAError,
  isConnectionError,
  isValidationError,
  isNotFoundError,
  isScopeViolationError,
  isTimeoutError,
  isServerError,
} from './errors';

// Version
export const VERSION = '0.5.0';
