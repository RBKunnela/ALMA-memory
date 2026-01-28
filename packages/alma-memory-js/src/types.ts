/**
 * ALMA Memory TypeScript Types
 *
 * These types mirror the Python dataclasses in alma/types.py
 * to ensure consistency between the Python and TypeScript SDKs.
 */

/**
 * Categories of memory that agents can store and retrieve.
 */
export type MemoryType =
  | 'heuristic'
  | 'outcome'
  | 'user_preference'
  | 'domain_knowledge'
  | 'anti_pattern';

/**
 * A learned rule: "When condition X, strategy Y works N% of the time."
 *
 * Heuristics are only created after min_occurrences validations.
 */
export interface Heuristic {
  /** Unique identifier */
  id: string;
  /** Agent that owns this heuristic */
  agent: string;
  /** Project this heuristic belongs to */
  project_id: string;
  /** The condition that triggers this heuristic (e.g., "form with multiple required fields") */
  condition: string;
  /** The strategy to apply (e.g., "test happy path first, then individual validation") */
  strategy: string;
  /** Confidence score from 0.0 to 1.0 */
  confidence: number;
  /** Number of times this heuristic has been observed */
  occurrence_count: number;
  /** Number of successful applications */
  success_count: number;
  /** When this heuristic was last validated */
  last_validated: string;
  /** When this heuristic was created */
  created_at: string;
  /** Success rate calculated from occurrence_count and success_count */
  success_rate?: number;
  /** Optional embedding vector */
  embedding?: number[];
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Record of a task execution - success or failure with context.
 *
 * Outcomes are raw data that can be consolidated into heuristics.
 */
export interface Outcome {
  /** Unique identifier */
  id: string;
  /** Agent that executed the task */
  agent: string;
  /** Project this outcome belongs to */
  project_id: string;
  /** Category of task (e.g., "api_validation", "form_testing") */
  task_type: string;
  /** Description of what was attempted */
  task_description: string;
  /** Whether the task succeeded */
  success: boolean;
  /** The approach that was taken */
  strategy_used: string;
  /** How long the task took in milliseconds */
  duration_ms?: number;
  /** Error details if the task failed */
  error_message?: string;
  /** User feedback if provided */
  user_feedback?: string;
  /** When this outcome was recorded */
  timestamp: string;
  /** Optional embedding vector */
  embedding?: number[];
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * A remembered user constraint or communication preference.
 *
 * Persists across sessions so users don't repeat themselves.
 */
export interface UserPreference {
  /** Unique identifier */
  id: string;
  /** User this preference belongs to */
  user_id: string;
  /** Category (e.g., "communication", "code_style", "workflow") */
  category: string;
  /** The preference text (e.g., "No emojis in documentation") */
  preference: string;
  /** How this preference was learned (e.g., "explicit_instruction", "inferred_from_correction") */
  source: string;
  /** Confidence score (lower for inferred preferences) */
  confidence: number;
  /** When this preference was recorded */
  timestamp: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Accumulated domain-specific facts within agent's scope.
 *
 * Different from heuristics - these are facts, not strategies.
 */
export interface DomainKnowledge {
  /** Unique identifier */
  id: string;
  /** Agent this knowledge belongs to */
  agent: string;
  /** Project this knowledge belongs to */
  project_id: string;
  /** Knowledge domain (e.g., "authentication", "database_schema") */
  domain: string;
  /** The fact to remember (e.g., "Login endpoint uses JWT with 24h expiry") */
  fact: string;
  /** How this was learned (e.g., "code_analysis", "documentation", "user_stated") */
  source: string;
  /** Confidence score from 0.0 to 1.0 */
  confidence: number;
  /** When this fact was last verified */
  last_verified: string;
  /** Optional embedding vector */
  embedding?: number[];
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * What NOT to do - learned from validated failures.
 *
 * Helps agents avoid repeating mistakes.
 */
export interface AntiPattern {
  /** Unique identifier */
  id: string;
  /** Agent this anti-pattern belongs to */
  agent: string;
  /** Project this anti-pattern belongs to */
  project_id: string;
  /** The pattern to avoid (e.g., "Using fixed sleep() for async waits") */
  pattern: string;
  /** Why this pattern is bad (e.g., "Causes flaky tests, doesn't adapt to load") */
  why_bad: string;
  /** What to do instead (e.g., "Use explicit waits with conditions") */
  better_alternative: string;
  /** Number of times this pattern has been observed */
  occurrence_count: number;
  /** When this pattern was last seen */
  last_seen: string;
  /** When this anti-pattern was first recorded */
  created_at: string;
  /** Optional embedding vector */
  embedding?: number[];
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * A compact, relevant subset of memories for injection into context.
 *
 * This is what gets injected per-call - must stay under token budget.
 */
export interface MemorySlice {
  /** Relevant learned strategies */
  heuristics: Heuristic[];
  /** Recent task outcomes */
  outcomes: Outcome[];
  /** User preferences to respect */
  preferences: UserPreference[];
  /** Domain facts to consider */
  domain_knowledge: DomainKnowledge[];
  /** Patterns to avoid */
  anti_patterns: AntiPattern[];
  /** The query used for retrieval */
  query?: string;
  /** The agent that requested this slice */
  agent?: string;
  /** How long retrieval took in milliseconds */
  retrieval_time_ms?: number;
  /** Total number of items in the slice */
  total_items?: number;
}

/**
 * Defines what an agent is allowed to learn and share.
 *
 * Prevents scope creep by explicitly listing allowed and forbidden domains.
 * Supports multi-agent memory sharing through share_with and inherit_from.
 */
export interface MemoryScope {
  /** Name of the agent */
  agent_name: string;
  /** Domains this agent is allowed to learn in (empty means all) */
  can_learn: string[];
  /** Domains this agent is forbidden from learning */
  cannot_learn: string[];
  /** Agents that can read this agent's memories */
  share_with?: string[];
  /** Agents whose memories this agent can read */
  inherit_from?: string[];
  /** Minimum observations before creating a heuristic (default: 3) */
  min_occurrences_for_heuristic?: number;
}

/**
 * Configuration options for the ALMA client.
 */
export interface ALMAConfig {
  /** Base URL of the ALMA MCP server (e.g., "http://localhost:8765") */
  baseUrl: string;
  /** Project identifier for memory isolation */
  projectId: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Custom headers to include in requests */
  headers?: Record<string, string>;
  /** Retry configuration */
  retry?: RetryConfig;
}

/**
 * Configuration for request retries.
 */
export interface RetryConfig {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Base delay between retries in milliseconds (default: 1000) */
  baseDelay?: number;
  /** Maximum delay between retries in milliseconds (default: 10000) */
  maxDelay?: number;
}

/**
 * Options for retrieving memories.
 */
export interface RetrieveOptions {
  /** Description of the task to perform */
  query: string;
  /** Name of the agent requesting memories */
  agent: string;
  /** Optional user ID for preference retrieval */
  userId?: string;
  /** Maximum items per memory type (default: 5) */
  topK?: number;
  /** Minimum confidence score for results (default: 0.0) */
  minConfidence?: number;
  /** Include memories from agents this agent inherits from (default: true) */
  includeShared?: boolean;
}

/**
 * Options for recording a learning outcome.
 */
export interface LearnOptions {
  /** Name of the agent that executed the task */
  agent: string;
  /** Description of the task */
  task: string;
  /** Category of task (for grouping) */
  taskType?: string;
  /** Whether the task succeeded or failed */
  outcome: 'success' | 'failure';
  /** What approach was taken */
  strategyUsed: string;
  /** Error details if failed */
  errorMessage?: string;
  /** How long the task took in milliseconds */
  durationMs?: number;
  /** User feedback if provided */
  feedback?: string;
}

/**
 * Options for adding a user preference.
 */
export interface AddPreferenceOptions {
  /** User identifier */
  userId: string;
  /** Category (e.g., "communication", "code_style", "workflow") */
  category: string;
  /** The preference text */
  preference: string;
  /** How this was learned (default: "explicit_instruction") */
  source?: string;
}

/**
 * Options for adding domain knowledge.
 */
export interface AddKnowledgeOptions {
  /** Agent this knowledge belongs to */
  agent: string;
  /** Knowledge domain */
  domain: string;
  /** The fact to remember */
  fact: string;
  /** How this was learned (default: "user_stated") */
  source?: string;
}

/**
 * Options for forgetting/pruning memories.
 */
export interface ForgetOptions {
  /** Specific agent to prune, or undefined for all */
  agent?: string;
  /** Remove outcomes older than this many days (default: 90) */
  olderThanDays?: number;
  /** Remove heuristics below this confidence (default: 0.3) */
  belowConfidence?: number;
}

/**
 * Response from a retrieve operation.
 */
export interface RetrieveResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** The retrieved memories */
  memories?: MemorySlice;
  /** Formatted text for prompt injection */
  prompt_injection?: string;
  /** Error message if failed */
  error?: string;
}

/**
 * Response from a learn operation.
 */
export interface LearnResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** Whether learning was recorded (false if scope violation) */
  learned?: boolean;
  /** Human-readable message */
  message?: string;
  /** Error message if failed */
  error?: string;
}

/**
 * Response from adding a preference.
 */
export interface AddPreferenceResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** The created preference */
  preference?: {
    id: string;
    user_id: string;
    category: string;
    preference: string;
    source: string;
  };
  /** Error message if failed */
  error?: string;
}

/**
 * Response from adding knowledge.
 */
export interface AddKnowledgeResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** The created knowledge */
  knowledge?: {
    id: string;
    agent: string;
    domain: string;
    fact: string;
    source: string;
  };
  /** Error message if failed */
  error?: string;
}

/**
 * Response from a forget operation.
 */
export interface ForgetResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** Number of memories pruned */
  pruned_count?: number;
  /** Human-readable message */
  message?: string;
  /** Error message if failed */
  error?: string;
}

/**
 * Memory statistics.
 */
export interface MemoryStats {
  /** Total number of heuristics */
  heuristics_count?: number;
  /** Total number of outcomes */
  outcomes_count?: number;
  /** Total number of preferences */
  preferences_count?: number;
  /** Total number of domain knowledge items */
  domain_knowledge_count?: number;
  /** Total number of anti-patterns */
  anti_patterns_count?: number;
  /** Total count of all memories */
  total_count?: number;
  /** Additional statistics */
  [key: string]: number | undefined;
}

/**
 * Response from a stats operation.
 */
export interface StatsResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** Memory statistics */
  stats?: MemoryStats;
  /** Error message if failed */
  error?: string;
}

/**
 * Response from a health check.
 */
export interface HealthResponse {
  /** Whether the operation succeeded */
  success: boolean;
  /** Health status ("healthy" or "unhealthy") */
  status?: 'healthy' | 'unhealthy';
  /** Project ID of the server */
  project_id?: string;
  /** Total number of memories */
  total_memories?: number;
  /** List of registered agents */
  registered_agents?: string[];
  /** Server timestamp */
  timestamp?: string;
  /** Error message if unhealthy */
  error?: string;
}

/**
 * MCP JSON-RPC request format.
 */
export interface MCPRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

/**
 * MCP JSON-RPC response format.
 */
export interface MCPResponse<T = unknown> {
  jsonrpc: '2.0';
  id: number;
  result?: {
    content?: Array<{
      type: string;
      text: string;
    }>;
  } & T;
  error?: {
    code: number;
    message: string;
  };
}
