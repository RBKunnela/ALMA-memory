/**
 * ALMA Memory Client
 *
 * Main client class for interacting with the ALMA MCP server via HTTP.
 * Uses native fetch (Node.js 18+) with no external dependencies.
 */

import type {
  ALMAConfig,
  RetrieveOptions,
  LearnOptions,
  AddPreferenceOptions,
  AddKnowledgeOptions,
  ForgetOptions,
  RetrieveResponse,
  LearnResponse,
  AddPreferenceResponse,
  AddKnowledgeResponse,
  ForgetResponse,
  StatsResponse,
  HealthResponse,
  MCPRequest,
  MCPResponse,
  RetryConfig,
  // v0.6.0 Workflow types
  ConsolidateOptions,
  ConsolidateResponse,
  CheckpointOptions,
  CheckpointResponse,
  ResumeOptions,
  ResumeResponse,
  MergeStatesOptions,
  MergeStatesResponse,
  WorkflowLearnOptions,
  WorkflowLearnResponse,
  LinkArtifactOptions,
  LinkArtifactResponse,
  GetArtifactsOptions,
  GetArtifactsResponse,
  CleanupCheckpointsOptions,
  CleanupCheckpointsResponse,
  RetrieveScopedOptions,
  RetrieveScopedResponse,
} from './types';

import {
  ALMAError,
  ConnectionError,
  ValidationError,
  TimeoutError,
  ServerError,
} from './errors';

/**
 * Default configuration values.
 */
const DEFAULTS = {
  timeout: 30000,
  maxRetries: 3,
  baseDelay: 1000,
  maxDelay: 10000,
} as const;

/**
 * ALMA Memory Client
 *
 * Provides a type-safe interface for interacting with the ALMA Memory MCP server.
 * All methods are async and use native fetch for HTTP communication.
 *
 * @example
 * ```typescript
 * import { ALMA } from 'alma-memory';
 *
 * const alma = new ALMA({
 *   baseUrl: 'http://localhost:8765',
 *   projectId: 'my-project'
 * });
 *
 * // Retrieve memories for a task
 * const memories = await alma.retrieve({
 *   query: 'authentication flow',
 *   agent: 'dev-agent'
 * });
 *
 * // Learn from an outcome
 * await alma.learn({
 *   agent: 'dev-agent',
 *   task: 'Implement OAuth',
 *   outcome: 'success',
 *   strategyUsed: 'Used passport.js middleware'
 * });
 * ```
 */
export class ALMA {
  private readonly config: Required<Omit<ALMAConfig, 'headers' | 'retry'>> & {
    headers: Record<string, string>;
    retry: Required<RetryConfig>;
  };
  private requestId: number = 0;

  /**
   * Create a new ALMA client.
   *
   * @param config - Configuration options
   * @throws {ValidationError} If required configuration is missing
   *
   * @example
   * ```typescript
   * const alma = new ALMA({
   *   baseUrl: 'http://localhost:8765',
   *   projectId: 'my-project',
   *   timeout: 60000, // 60 second timeout
   *   retry: {
   *     maxRetries: 5,
   *     baseDelay: 2000
   *   }
   * });
   * ```
   */
  constructor(config: ALMAConfig) {
    if (!config.baseUrl) {
      throw new ValidationError('baseUrl is required', { field: 'baseUrl' });
    }
    if (!config.projectId) {
      throw new ValidationError('projectId is required', { field: 'projectId' });
    }

    this.config = {
      baseUrl: config.baseUrl.replace(/\/$/, ''), // Remove trailing slash
      projectId: config.projectId,
      timeout: config.timeout ?? DEFAULTS.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...config.headers,
      },
      retry: {
        maxRetries: config.retry?.maxRetries ?? DEFAULTS.maxRetries,
        baseDelay: config.retry?.baseDelay ?? DEFAULTS.baseDelay,
        maxDelay: config.retry?.maxDelay ?? DEFAULTS.maxDelay,
      },
    };
  }

  /**
   * Retrieve relevant memories for a task.
   *
   * Returns heuristics, domain knowledge, anti-patterns, and user preferences
   * that are relevant to the specified task.
   *
   * @param options - Retrieval options
   * @returns Memory slice with relevant memories and formatted prompt injection
   * @throws {ValidationError} If required fields are missing
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * const result = await alma.retrieve({
   *   query: 'form validation testing',
   *   agent: 'qa-agent',
   *   topK: 10,
   *   userId: 'user-123'
   * });
   *
   * // Use the memories
   * console.log(`Retrieved ${result.memories?.total_items} memories`);
   *
   * // Inject into prompt
   * const prompt = `${result.prompt_injection}\n\nNow, test the form...`;
   * ```
   */
  async retrieve(options: RetrieveOptions): Promise<RetrieveResponse> {
    this.validateRequired(options.query, 'query');
    this.validateRequired(options.agent, 'agent');

    const response = await this.callTool<RetrieveResponse>('alma_retrieve', {
      task: options.query,
      agent: options.agent,
      user_id: options.userId,
      top_k: options.topK ?? 5,
    });

    return response;
  }

  /**
   * Record a task outcome for learning.
   *
   * Use this after completing a task to help improve future performance.
   * Outcomes are consolidated into heuristics after enough observations.
   *
   * @param options - Learning options
   * @returns Whether learning was recorded
   * @throws {ValidationError} If required fields are missing
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * // Record a successful outcome
   * await alma.learn({
   *   agent: 'dev-agent',
   *   task: 'Implement user authentication',
   *   taskType: 'authentication',
   *   outcome: 'success',
   *   strategyUsed: 'Used JWT with refresh tokens',
   *   durationMs: 45000
   * });
   *
   * // Record a failure
   * await alma.learn({
   *   agent: 'dev-agent',
   *   task: 'Deploy to production',
   *   outcome: 'failure',
   *   strategyUsed: 'Direct push to main',
   *   errorMessage: 'Tests failed due to missing env vars'
   * });
   * ```
   */
  async learn(options: LearnOptions): Promise<LearnResponse> {
    this.validateRequired(options.agent, 'agent');
    this.validateRequired(options.task, 'task');
    this.validateRequired(options.outcome, 'outcome');
    this.validateRequired(options.strategyUsed, 'strategyUsed');

    if (options.outcome !== 'success' && options.outcome !== 'failure') {
      throw new ValidationError('outcome must be "success" or "failure"', {
        field: 'outcome',
        value: options.outcome,
      });
    }

    const response = await this.callTool<LearnResponse>('alma_learn', {
      agent: options.agent,
      task: options.task,
      task_type: options.taskType,
      outcome: options.outcome,
      strategy_used: options.strategyUsed,
      error_message: options.errorMessage,
      duration_ms: options.durationMs,
      feedback: options.feedback,
    });

    return response;
  }

  /**
   * Add a user preference to memory.
   *
   * Preferences persist across sessions so users don't have to repeat themselves.
   *
   * @param options - Preference options
   * @returns The created preference
   * @throws {ValidationError} If required fields are missing
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * // Add explicit preference
   * await alma.addPreference({
   *   userId: 'user-123',
   *   category: 'code_style',
   *   preference: 'Always use TypeScript strict mode',
   *   source: 'explicit_instruction'
   * });
   *
   * // Add inferred preference
   * await alma.addPreference({
   *   userId: 'user-123',
   *   category: 'communication',
   *   preference: 'Prefers detailed explanations',
   *   source: 'inferred_from_feedback'
   * });
   * ```
   */
  async addPreference(options: AddPreferenceOptions): Promise<AddPreferenceResponse> {
    this.validateRequired(options.userId, 'userId');
    this.validateRequired(options.category, 'category');
    this.validateRequired(options.preference, 'preference');

    const response = await this.callTool<AddPreferenceResponse>('alma_add_preference', {
      user_id: options.userId,
      category: options.category,
      preference: options.preference,
      source: options.source ?? 'explicit_instruction',
    });

    return response;
  }

  /**
   * Add domain knowledge within agent's scope.
   *
   * Knowledge items are facts, not strategies. They represent accumulated
   * understanding about a specific domain.
   *
   * @param options - Knowledge options
   * @returns The created knowledge or error if scope violation
   * @throws {ValidationError} If required fields are missing
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * // Add domain knowledge
   * await alma.addKnowledge({
   *   agent: 'dev-agent',
   *   domain: 'authentication',
   *   fact: 'The API uses OAuth 2.0 with PKCE for mobile clients',
   *   source: 'documentation'
   * });
   *
   * // Add knowledge from code analysis
   * await alma.addKnowledge({
   *   agent: 'dev-agent',
   *   domain: 'database',
   *   fact: 'Users table has soft delete with deleted_at column',
   *   source: 'code_analysis'
   * });
   * ```
   */
  async addKnowledge(options: AddKnowledgeOptions): Promise<AddKnowledgeResponse> {
    this.validateRequired(options.agent, 'agent');
    this.validateRequired(options.domain, 'domain');
    this.validateRequired(options.fact, 'fact');

    const response = await this.callTool<AddKnowledgeResponse>('alma_add_knowledge', {
      agent: options.agent,
      domain: options.domain,
      fact: options.fact,
      source: options.source ?? 'user_stated',
    });

    return response;
  }

  /**
   * Prune stale or low-confidence memories.
   *
   * Use this periodically to keep the memory system clean and efficient.
   *
   * @param options - Forget options (all optional)
   * @returns Number of memories pruned
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * // Prune all agents
   * const result = await alma.forget({
   *   olderThanDays: 90,
   *   belowConfidence: 0.3
   * });
   * console.log(`Pruned ${result.pruned_count} memories`);
   *
   * // Prune specific agent
   * await alma.forget({
   *   agent: 'test-agent',
   *   olderThanDays: 30
   * });
   * ```
   */
  async forget(options: ForgetOptions = {}): Promise<ForgetResponse> {
    const response = await this.callTool<ForgetResponse>('alma_forget', {
      agent: options.agent,
      older_than_days: options.olderThanDays ?? 90,
      below_confidence: options.belowConfidence ?? 0.3,
    });

    return response;
  }

  /**
   * Get memory statistics.
   *
   * @param agent - Specific agent to get stats for, or undefined for all
   * @returns Memory statistics
   * @throws {ConnectionError} If server is unreachable
   * @throws {ServerError} If server returns an error
   *
   * @example
   * ```typescript
   * // Get stats for all agents
   * const allStats = await alma.stats();
   * console.log(`Total memories: ${allStats.stats?.total_count}`);
   *
   * // Get stats for specific agent
   * const agentStats = await alma.stats('dev-agent');
   * console.log(`Agent has ${agentStats.stats?.heuristics_count} heuristics`);
   * ```
   */
  async stats(agent?: string): Promise<StatsResponse> {
    const response = await this.callTool<StatsResponse>('alma_stats', {
      agent,
    });

    return response;
  }

  /**
   * Check server health status.
   *
   * @returns Health status including registered agents and total memories
   * @throws {ConnectionError} If server is unreachable
   *
   * @example
   * ```typescript
   * const health = await alma.health();
   * if (health.status === 'healthy') {
   *   console.log(`Server healthy with ${health.total_memories} memories`);
   *   console.log(`Registered agents: ${health.registered_agents?.join(', ')}`);
   * }
   * ```
   */
  async health(): Promise<HealthResponse> {
    const response = await this.callTool<HealthResponse>('alma_health', {});
    return response;
  }

  // ============================================================
  // v0.6.0 Workflow Context Methods
  // ============================================================

  /**
   * Consolidate similar memories using LLM-powered deduplication.
   *
   * @param options - Consolidation options
   * @returns Consolidation results
   *
   * @example
   * ```typescript
   * // Dry run to preview
   * const preview = await alma.consolidate({
   *   agent: 'dev-agent',
   *   similarityThreshold: 0.85,
   *   dryRun: true
   * });
   * console.log(`Would merge ${preview.memoriesMerged} memories`);
   *
   * // Actual consolidation
   * const result = await alma.consolidate({
   *   agent: 'dev-agent',
   *   similarityThreshold: 0.85
   * });
   * ```
   */
  async consolidate(options: ConsolidateOptions = {}): Promise<ConsolidateResponse> {
    const response = await this.callTool<ConsolidateResponse>('alma_consolidate', {
      agent: options.agent,
      similarity_threshold: options.similarityThreshold ?? 0.85,
      min_group_size: options.minGroupSize ?? 2,
      dry_run: options.dryRun ?? false,
    });
    return response;
  }

  /**
   * Create a checkpoint for crash recovery.
   *
   * @param options - Checkpoint options
   * @returns Created checkpoint info
   *
   * @example
   * ```typescript
   * const result = await alma.checkpoint({
   *   runId: 'run-123',
   *   nodeId: 'step-2',
   *   state: { progress: 50, data: [...] }
   * });
   * console.log(`Checkpoint ${result.checkpointId} created`);
   * ```
   */
  async checkpoint(options: CheckpointOptions): Promise<CheckpointResponse> {
    this.validateRequired(options.runId, 'runId');
    this.validateRequired(options.nodeId, 'nodeId');
    this.validateRequired(options.state, 'state');

    const response = await this.callTool<CheckpointResponse>('alma_checkpoint', {
      run_id: options.runId,
      node_id: options.nodeId,
      state: options.state,
      branch_id: options.branchId,
      metadata: options.metadata,
    });
    return response;
  }

  /**
   * Resume from a checkpoint after crash or restart.
   *
   * @param options - Resume options
   * @returns Restored checkpoint and state
   *
   * @example
   * ```typescript
   * const result = await alma.resume({ runId: 'run-123' });
   * if (result.success && result.state) {
   *   console.log(`Resuming from step ${result.checkpoint?.nodeId}`);
   *   // Continue with restored state
   * }
   * ```
   */
  async resume(options: ResumeOptions): Promise<ResumeResponse> {
    this.validateRequired(options.runId, 'runId');

    const response = await this.callTool<ResumeResponse>('alma_resume', {
      run_id: options.runId,
      checkpoint_id: options.checkpointId,
      branch_id: options.branchId,
    });
    return response;
  }

  /**
   * Merge states from parallel workflow branches.
   *
   * @param options - Merge options
   * @returns Merged state
   *
   * @example
   * ```typescript
   * const result = await alma.mergeStates({
   *   runId: 'run-123',
   *   branchIds: ['branch-a', 'branch-b'],
   *   reducers: {
   *     results: 'append',
   *     counts: 'sum',
   *     status: 'last'
   *   }
   * });
   * console.log('Merged state:', result.mergedState);
   * ```
   */
  async mergeStates(options: MergeStatesOptions): Promise<MergeStatesResponse> {
    this.validateRequired(options.runId, 'runId');
    this.validateRequired(options.branchIds, 'branchIds');

    const response = await this.callTool<MergeStatesResponse>('alma_merge_states', {
      run_id: options.runId,
      branch_ids: options.branchIds,
      reducers: options.reducers,
    });
    return response;
  }

  /**
   * Learn from a completed workflow execution.
   *
   * @param options - Workflow learning options
   * @returns Learning results
   *
   * @example
   * ```typescript
   * await alma.workflowLearn({
   *   context: { workflowId: 'deploy', runId: 'run-123' },
   *   agent: 'deploy-agent',
   *   result: 'success',
   *   summary: 'Deployed v2.0 to production',
   *   strategiesUsed: ['blue-green', 'canary'],
   *   successfulPatterns: ['gradual rollout'],
   *   durationSeconds: 300
   * });
   * ```
   */
  async workflowLearn(options: WorkflowLearnOptions): Promise<WorkflowLearnResponse> {
    this.validateRequired(options.context, 'context');
    this.validateRequired(options.agent, 'agent');
    this.validateRequired(options.result, 'result');

    const response = await this.callTool<WorkflowLearnResponse>('alma_workflow_learn', {
      tenant_id: options.context.tenantId,
      workflow_id: options.context.workflowId,
      run_id: options.context.runId,
      agent: options.agent,
      result: options.result,
      summary: options.summary,
      strategies_used: options.strategiesUsed,
      successful_patterns: options.successfulPatterns,
      failed_patterns: options.failedPatterns,
      duration_seconds: options.durationSeconds,
      node_count: options.nodeCount,
    });
    return response;
  }

  /**
   * Link an external artifact to a memory.
   *
   * @param options - Artifact linking options
   * @returns Created artifact reference
   *
   * @example
   * ```typescript
   * const result = await alma.linkArtifact({
   *   memoryId: 'outcome-123',
   *   artifactType: 'report',
   *   storageUrl: 's3://bucket/reports/test-report.html',
   *   filename: 'test-report.html',
   *   mimeType: 'text/html'
   * });
   * ```
   */
  async linkArtifact(options: LinkArtifactOptions): Promise<LinkArtifactResponse> {
    this.validateRequired(options.memoryId, 'memoryId');
    this.validateRequired(options.artifactType, 'artifactType');
    this.validateRequired(options.storageUrl, 'storageUrl');

    const response = await this.callTool<LinkArtifactResponse>('alma_link_artifact', {
      memory_id: options.memoryId,
      artifact_type: options.artifactType,
      storage_url: options.storageUrl,
      filename: options.filename,
      mime_type: options.mimeType,
      size_bytes: options.sizeBytes,
      checksum: options.checksum,
      metadata: options.metadata,
    });
    return response;
  }

  /**
   * Get artifacts linked to a memory.
   *
   * @param options - Get artifacts options
   * @returns List of artifact references
   *
   * @example
   * ```typescript
   * const result = await alma.getArtifacts({
   *   memoryId: 'outcome-123',
   *   artifactType: 'report'
   * });
   * for (const artifact of result.artifacts ?? []) {
   *   console.log(`${artifact.filename}: ${artifact.storageUrl}`);
   * }
   * ```
   */
  async getArtifacts(options: GetArtifactsOptions): Promise<GetArtifactsResponse> {
    this.validateRequired(options.memoryId, 'memoryId');

    const response = await this.callTool<GetArtifactsResponse>('alma_get_artifacts', {
      memory_id: options.memoryId,
      artifact_type: options.artifactType,
    });
    return response;
  }

  /**
   * Clean up old checkpoints for a run.
   *
   * @param options - Cleanup options
   * @returns Cleanup results
   *
   * @example
   * ```typescript
   * // Keep only the latest checkpoint
   * const result = await alma.cleanupCheckpoints({
   *   runId: 'run-123',
   *   keepLatest: 1
   * });
   * console.log(`Deleted ${result.deleted} old checkpoints`);
   * ```
   */
  async cleanupCheckpoints(options: CleanupCheckpointsOptions): Promise<CleanupCheckpointsResponse> {
    this.validateRequired(options.runId, 'runId');

    const response = await this.callTool<CleanupCheckpointsResponse>('alma_cleanup_checkpoints', {
      run_id: options.runId,
      keep_latest: options.keepLatest ?? 1,
      branch_id: options.branchId,
    });
    return response;
  }

  /**
   * Retrieve memories with workflow context scoping.
   *
   * @param options - Scoped retrieval options
   * @returns Scoped memory slice
   *
   * @example
   * ```typescript
   * // Get memories scoped to current workflow run
   * const result = await alma.retrieveScoped({
   *   query: 'deployment patterns',
   *   agent: 'deploy-agent',
   *   context: { workflowId: 'deploy', runId: 'run-123' },
   *   scope: 'run'
   * });
   * ```
   */
  async retrieveScoped(options: RetrieveScopedOptions): Promise<RetrieveScopedResponse> {
    this.validateRequired(options.query, 'query');
    this.validateRequired(options.agent, 'agent');
    this.validateRequired(options.context, 'context');
    this.validateRequired(options.scope, 'scope');

    const response = await this.callTool<RetrieveScopedResponse>('alma_retrieve_scoped', {
      task: options.query,
      agent: options.agent,
      tenant_id: options.context.tenantId,
      workflow_id: options.context.workflowId,
      run_id: options.context.runId,
      node_id: options.context.nodeId,
      scope: options.scope,
      top_k: options.topK ?? 5,
    });
    return response;
  }

  /**
   * Get the configured project ID.
   */
  get projectId(): string {
    return this.config.projectId;
  }

  /**
   * Get the configured base URL.
   */
  get baseUrl(): string {
    return this.config.baseUrl;
  }

  // ============================================================
  // Private methods
  // ============================================================

  /**
   * Call an MCP tool via HTTP.
   */
  private async callTool<T>(
    toolName: string,
    params: Record<string, unknown>,
  ): Promise<T> {
    const request: MCPRequest = {
      jsonrpc: '2.0',
      id: ++this.requestId,
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: this.filterUndefined(params),
      },
    };

    const response = await this.request<MCPResponse<T>>(request);
    return this.parseResponse<T>(response);
  }

  /**
   * Make an HTTP request with retry logic.
   */
  private async request<T>(body: MCPRequest): Promise<T> {
    const { maxRetries, baseDelay, maxDelay } = this.config.retry;
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.singleRequest<T>(body);
      } catch (error) {
        lastError = error as Error;

        // Don't retry validation errors
        if (error instanceof ValidationError) {
          throw error;
        }

        // Check if we should retry
        if (attempt < maxRetries) {
          const delay = Math.min(
            baseDelay * Math.pow(2, attempt),
            maxDelay,
          );
          await this.sleep(delay);
        }
      }
    }

    throw lastError;
  }

  /**
   * Make a single HTTP request.
   */
  private async singleRequest<T>(body: MCPRequest): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.config.timeout,
    );

    try {
      const response = await fetch(this.config.baseUrl, {
        method: 'POST',
        headers: this.config.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ServerError(
          `HTTP ${response.status}: ${response.statusText}`,
          { statusCode: response.status },
        );
      }

      const data = await response.json() as T;
      return data;
    } catch (error) {
      if (error instanceof ALMAError) {
        throw error;
      }

      const err = error as Error;

      if (err.name === 'AbortError') {
        throw new TimeoutError(
          `Request timed out after ${this.config.timeout}ms`,
          this.config.timeout,
        );
      }

      if (
        err.message.includes('ECONNREFUSED') ||
        err.message.includes('ENOTFOUND') ||
        err.message.includes('fetch failed')
      ) {
        throw new ConnectionError(
          `Failed to connect to ${this.config.baseUrl}: ${err.message}`,
          { url: this.config.baseUrl, cause: err },
        );
      }

      throw new ALMAError(err.message, 'UNKNOWN_ERROR', { cause: err });
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Parse MCP response and extract result.
   */
  private parseResponse<T>(response: MCPResponse<T>): T {
    if (response.error) {
      throw new ServerError(response.error.message, {
        rpcCode: response.error.code,
        serverMessage: response.error.message,
      });
    }

    // MCP returns tool results as content array with text
    if (response.result?.content?.[0]?.text) {
      try {
        return JSON.parse(response.result.content[0].text) as T;
      } catch {
        // If parsing fails, return the raw text wrapped
        return { success: false, error: 'Failed to parse response' } as T;
      }
    }

    // Direct result (for non-tool calls)
    if (response.result) {
      return response.result as T;
    }

    throw new ServerError('Empty response from server');
  }

  /**
   * Validate that a required field is present and non-empty.
   */
  private validateRequired(value: unknown, field: string): void {
    if (value === undefined || value === null || value === '') {
      throw new ValidationError(`${field} is required`, { field });
    }
  }

  /**
   * Remove undefined values from an object.
   */
  private filterUndefined(
    obj: Record<string, unknown>,
  ): Record<string, unknown> {
    return Object.fromEntries(
      Object.entries(obj).filter(([_, v]) => v !== undefined),
    );
  }

  /**
   * Sleep for a specified number of milliseconds.
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create a new ALMA client instance.
 *
 * This is a convenience function that wraps the ALMA constructor.
 *
 * @param config - Configuration options
 * @returns A new ALMA client instance
 *
 * @example
 * ```typescript
 * import { createClient } from 'alma-memory';
 *
 * const alma = createClient({
 *   baseUrl: 'http://localhost:8765',
 *   projectId: 'my-project'
 * });
 * ```
 */
export function createClient(config: ALMAConfig): ALMA {
  return new ALMA(config);
}
