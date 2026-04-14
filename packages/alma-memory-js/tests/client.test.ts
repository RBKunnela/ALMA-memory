/**
 * ALMA Memory Client Tests
 *
 * These tests use mocked HTTP responses to test the client without
 * requiring a running server.
 */

import {
  ALMA,
  createClient,
  ALMAError,
  ConnectionError,
  ValidationError,
  TimeoutError,
  ServerError,
  isALMAError,
  isConnectionError,
  isValidationError,
  isServerError,
  VERSION,
  FeedbackSignal,
} from '../src';

import type {
  RetrievalFeedback,
  FeedbackSummary,
} from '../src';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Helper to create mock MCP response
function createMCPResponse<T>(result: T): object {
  return {
    jsonrpc: '2.0',
    id: 1,
    result: {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result),
        },
      ],
    },
  };
}

// Helper to create mock error response
function createMCPError(code: number, message: string): object {
  return {
    jsonrpc: '2.0',
    id: 1,
    error: {
      code,
      message,
    },
  };
}

// Helper to setup mock fetch response
function setupMockResponse(response: object, options?: { ok?: boolean; status?: number }) {
  mockFetch.mockResolvedValueOnce({
    ok: options?.ok ?? true,
    status: options?.status ?? 200,
    statusText: options?.ok === false ? 'Error' : 'OK',
    json: () => Promise.resolve(response),
  });
}

describe('ALMA Client', () => {
  let alma: ALMA;

  beforeEach(() => {
    jest.clearAllMocks();
    alma = new ALMA({
      baseUrl: 'http://localhost:8765',
      projectId: 'test-project',
    });
  });

  describe('Constructor', () => {
    it('should create client with valid config', () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'my-project',
      });

      expect(client.baseUrl).toBe('http://localhost:8765');
      expect(client.projectId).toBe('my-project');
    });

    it('should remove trailing slash from baseUrl', () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765/',
        projectId: 'my-project',
      });

      expect(client.baseUrl).toBe('http://localhost:8765');
    });

    it('should throw ValidationError if baseUrl is missing', () => {
      expect(() => {
        new ALMA({ baseUrl: '', projectId: 'test' });
      }).toThrow(ValidationError);
    });

    it('should throw ValidationError if projectId is missing', () => {
      expect(() => {
        new ALMA({ baseUrl: 'http://localhost', projectId: '' });
      }).toThrow(ValidationError);
    });

    it('should accept custom timeout', () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        timeout: 60000,
      });

      expect(client).toBeInstanceOf(ALMA);
    });

    it('should accept custom headers', () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        headers: { 'X-Custom': 'value' },
      });

      expect(client).toBeInstanceOf(ALMA);
    });

    it('should accept retry configuration', () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        retry: {
          maxRetries: 5,
          baseDelay: 2000,
          maxDelay: 20000,
        },
      });

      expect(client).toBeInstanceOf(ALMA);
    });
  });

  describe('createClient helper', () => {
    it('should create ALMA instance', () => {
      const client = createClient({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
      });

      expect(client).toBeInstanceOf(ALMA);
    });
  });

  describe('retrieve()', () => {
    it('should retrieve memories successfully', async () => {
      const mockResponse = {
        success: true,
        memories: {
          heuristics: [
            {
              id: 'h1',
              condition: 'Form validation',
              strategy: 'Test happy path first',
              confidence: 0.85,
              occurrence_count: 10,
              success_rate: 0.9,
            },
          ],
          outcomes: [],
          preferences: [],
          domain_knowledge: [],
          anti_patterns: [],
          total_items: 1,
        },
        prompt_injection: '## Relevant Strategies\n- When: Form validation\n  Do: Test happy path first',
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.retrieve({
        query: 'form validation',
        agent: 'test-agent',
      });

      expect(result.success).toBe(true);
      expect(result.memories?.heuristics).toHaveLength(1);
      expect(result.memories?.heuristics[0]?.confidence).toBe(0.85);
      expect(result.prompt_injection).toContain('Relevant Strategies');
    });

    it('should pass optional parameters', async () => {
      setupMockResponse(createMCPResponse({ success: true, memories: {} }));

      await alma.retrieve({
        query: 'test',
        agent: 'agent',
        userId: 'user-123',
        topK: 10,
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8765',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"user_id":"user-123"'),
        }),
      );
    });

    it('should throw ValidationError if query is empty', async () => {
      await expect(
        alma.retrieve({ query: '', agent: 'test' }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError if agent is empty', async () => {
      await expect(
        alma.retrieve({ query: 'test', agent: '' }),
      ).rejects.toThrow(ValidationError);
    });
  });

  describe('learn()', () => {
    it('should record successful outcome', async () => {
      const mockResponse = {
        success: true,
        learned: true,
        message: 'Outcome recorded',
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.learn({
        agent: 'dev-agent',
        task: 'Implement OAuth',
        outcome: 'success',
        strategyUsed: 'Used passport.js',
      });

      expect(result.success).toBe(true);
      expect(result.learned).toBe(true);
    });

    it('should record failure with error message', async () => {
      setupMockResponse(
        createMCPResponse({ success: true, learned: true }),
      );

      await alma.learn({
        agent: 'dev-agent',
        task: 'Deploy',
        outcome: 'failure',
        strategyUsed: 'Direct push',
        errorMessage: 'Tests failed',
        durationMs: 5000,
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8765',
        expect.objectContaining({
          body: expect.stringContaining('"error_message":"Tests failed"'),
        }),
      );
    });

    it('should pass optional parameters', async () => {
      setupMockResponse(createMCPResponse({ success: true }));

      await alma.learn({
        agent: 'agent',
        task: 'task',
        outcome: 'success',
        strategyUsed: 'strategy',
        taskType: 'testing',
        durationMs: 1000,
        feedback: 'good',
      });

      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.params.arguments.task_type).toBe('testing');
      expect(callBody.params.arguments.duration_ms).toBe(1000);
      expect(callBody.params.arguments.feedback).toBe('good');
    });

    it('should throw ValidationError for invalid outcome', async () => {
      await expect(
        alma.learn({
          agent: 'agent',
          task: 'task',
          outcome: 'maybe' as 'success',
          strategyUsed: 'strategy',
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError if required fields missing', async () => {
      await expect(
        alma.learn({
          agent: '',
          task: 'task',
          outcome: 'success',
          strategyUsed: 'strategy',
        }),
      ).rejects.toThrow(ValidationError);
    });
  });

  describe('addPreference()', () => {
    it('should add preference successfully', async () => {
      const mockResponse = {
        success: true,
        preference: {
          id: 'pref-1',
          user_id: 'user-123',
          category: 'code_style',
          preference: 'Use TypeScript',
          source: 'explicit_instruction',
        },
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.addPreference({
        userId: 'user-123',
        category: 'code_style',
        preference: 'Use TypeScript',
      });

      expect(result.success).toBe(true);
      expect(result.preference?.id).toBe('pref-1');
    });

    it('should use default source', async () => {
      setupMockResponse(createMCPResponse({ success: true }));

      await alma.addPreference({
        userId: 'user',
        category: 'test',
        preference: 'pref',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8765',
        expect.objectContaining({
          body: expect.stringContaining('"source":"explicit_instruction"'),
        }),
      );
    });

    it('should throw ValidationError if userId is empty', async () => {
      await expect(
        alma.addPreference({
          userId: '',
          category: 'test',
          preference: 'pref',
        }),
      ).rejects.toThrow(ValidationError);
    });
  });

  describe('addKnowledge()', () => {
    it('should add knowledge successfully', async () => {
      const mockResponse = {
        success: true,
        knowledge: {
          id: 'know-1',
          agent: 'dev-agent',
          domain: 'auth',
          fact: 'Uses JWT',
          source: 'documentation',
        },
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.addKnowledge({
        agent: 'dev-agent',
        domain: 'auth',
        fact: 'Uses JWT',
        source: 'documentation',
      });

      expect(result.success).toBe(true);
      expect(result.knowledge?.domain).toBe('auth');
    });

    it('should use default source', async () => {
      setupMockResponse(createMCPResponse({ success: true }));

      await alma.addKnowledge({
        agent: 'agent',
        domain: 'domain',
        fact: 'fact',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8765',
        expect.objectContaining({
          body: expect.stringContaining('"source":"user_stated"'),
        }),
      );
    });

    it('should handle scope violation response', async () => {
      const mockResponse = {
        success: false,
        error: "Agent 'test' not allowed to learn in domain 'forbidden'",
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.addKnowledge({
        agent: 'test',
        domain: 'forbidden',
        fact: 'fact',
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('not allowed');
    });
  });

  describe('forget()', () => {
    it('should prune memories with defaults', async () => {
      const mockResponse = {
        success: true,
        pruned_count: 42,
        message: 'Pruned 42 stale or low-confidence memories',
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.forget();

      expect(result.success).toBe(true);
      expect(result.pruned_count).toBe(42);
    });

    it('should accept custom options', async () => {
      setupMockResponse(createMCPResponse({ success: true, pruned_count: 0 }));

      await alma.forget({
        agent: 'specific-agent',
        olderThanDays: 30,
        belowConfidence: 0.5,
      });

      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.params.arguments.agent).toBe('specific-agent');
      expect(callBody.params.arguments.older_than_days).toBe(30);
      expect(callBody.params.arguments.below_confidence).toBe(0.5);
    });
  });

  describe('stats()', () => {
    it('should get stats for all agents', async () => {
      const mockResponse = {
        success: true,
        stats: {
          heuristics_count: 10,
          outcomes_count: 50,
          preferences_count: 5,
          domain_knowledge_count: 20,
          anti_patterns_count: 3,
          total_count: 88,
        },
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.stats();

      expect(result.success).toBe(true);
      expect(result.stats?.total_count).toBe(88);
    });

    it('should get stats for specific agent', async () => {
      setupMockResponse(
        createMCPResponse({ success: true, stats: { total_count: 10 } }),
      );

      await alma.stats('dev-agent');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8765',
        expect.objectContaining({
          body: expect.stringContaining('"agent":"dev-agent"'),
        }),
      );
    });
  });

  describe('health()', () => {
    it('should check server health', async () => {
      const mockResponse = {
        success: true,
        status: 'healthy',
        project_id: 'test-project',
        total_memories: 100,
        registered_agents: ['agent1', 'agent2'],
        timestamp: '2026-01-28T10:00:00Z',
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const result = await alma.health();

      expect(result.success).toBe(true);
      expect(result.status).toBe('healthy');
      expect(result.registered_agents).toContain('agent1');
    });
  });

  describe('Error handling', () => {
    it('should throw ServerError on HTTP error', async () => {
      // Mock multiple responses for retry attempts
      setupMockResponse({}, { ok: false, status: 500 });
      setupMockResponse({}, { ok: false, status: 500 });
      setupMockResponse({}, { ok: false, status: 500 });
      setupMockResponse({}, { ok: false, status: 500 });

      await expect(alma.health()).rejects.toThrow(ServerError);
    });

    it('should throw ServerError on MCP error response', async () => {
      setupMockResponse(createMCPError(-32602, 'Unknown tool'));

      await expect(alma.health()).rejects.toThrow(ServerError);
    });

    it('should throw ConnectionError on fetch failure', async () => {
      // Mock multiple rejections for retry attempts
      mockFetch
        .mockRejectedValueOnce(new Error('fetch failed'))
        .mockRejectedValueOnce(new Error('fetch failed'))
        .mockRejectedValueOnce(new Error('fetch failed'))
        .mockRejectedValueOnce(new Error('fetch failed'));

      await expect(alma.health()).rejects.toThrow(ConnectionError);
    });

    it('should throw TimeoutError on abort', async () => {
      const abortError = new Error('Aborted');
      abortError.name = 'AbortError';
      mockFetch.mockRejectedValueOnce(abortError);

      const fastClient = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        timeout: 100,
        retry: { maxRetries: 0 },
      });

      await expect(fastClient.health()).rejects.toThrow(TimeoutError);
    });

    it('should retry on connection failure', async () => {
      // First two attempts fail, third succeeds
      mockFetch
        .mockRejectedValueOnce(new Error('fetch failed'))
        .mockRejectedValueOnce(new Error('fetch failed'))
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(createMCPResponse({ success: true })),
        });

      const clientWithRetry = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        retry: { maxRetries: 3, baseDelay: 10 },
      });

      const result = await clientWithRetry.health();
      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('should not retry validation errors', async () => {
      const client = new ALMA({
        baseUrl: 'http://localhost:8765',
        projectId: 'test',
        retry: { maxRetries: 3 },
      });

      await expect(
        client.retrieve({ query: '', agent: 'test' }),
      ).rejects.toThrow(ValidationError);

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('Type guards', () => {
    it('isALMAError should identify ALMA errors', () => {
      expect(isALMAError(new ALMAError('test'))).toBe(true);
      expect(isALMAError(new ValidationError('test'))).toBe(true);
      expect(isALMAError(new Error('test'))).toBe(false);
    });

    it('isConnectionError should identify connection errors', () => {
      expect(isConnectionError(new ConnectionError('test'))).toBe(true);
      expect(isConnectionError(new ALMAError('test'))).toBe(false);
    });

    it('isValidationError should identify validation errors', () => {
      expect(isValidationError(new ValidationError('test'))).toBe(true);
      expect(isValidationError(new ALMAError('test'))).toBe(false);
    });

    it('isServerError should identify server errors', () => {
      expect(isServerError(new ServerError('test'))).toBe(true);
      expect(isServerError(new ALMAError('test'))).toBe(false);
    });
  });

  describe('VERSION', () => {
    it('should export version', () => {
      expect(VERSION).toBe('0.9.0');
    });
  });
});

describe('FeedbackSignal enum', () => {
  it('should have all expected values', () => {
    expect(FeedbackSignal.USED).toBe('used');
    expect(FeedbackSignal.IGNORED).toBe('ignored');
    expect(FeedbackSignal.THUMBS_UP).toBe('thumbs_up');
    expect(FeedbackSignal.THUMBS_DOWN).toBe('thumbs_down');
  });

  it('should have exactly 4 values', () => {
    const values = Object.values(FeedbackSignal);
    expect(values).toHaveLength(4);
  });
});

describe('Feedback types', () => {
  it('should allow constructing a RetrievalFeedback object', () => {
    const feedback: RetrievalFeedback = {
      id: 'fb-1',
      memoryId: 'heur-1',
      memoryType: 'heuristic',
      query: 'form validation',
      agent: 'dev-agent',
      projectId: 'test-project',
      signal: FeedbackSignal.USED,
      timestamp: '2026-04-14T10:00:00Z',
      metadata: { context: 'testing' },
    };

    expect(feedback.id).toBe('fb-1');
    expect(feedback.signal).toBe(FeedbackSignal.USED);
    expect(feedback.memoryType).toBe('heuristic');
  });

  it('should allow constructing a FeedbackSummary object', () => {
    const summary: FeedbackSummary = {
      memoryId: 'heur-1',
      memoryType: 'heuristic',
      useCount: 10,
      ignoreCount: 2,
      positiveCount: 8,
      negativeCount: 1,
      feedbackScore: 0.75,
    };

    expect(summary.feedbackScore).toBe(0.75);
    expect(summary.useCount).toBe(10);
  });

  it('should allow negative feedbackScore', () => {
    const summary: FeedbackSummary = {
      memoryId: 'ap-1',
      memoryType: 'anti_pattern',
      useCount: 0,
      ignoreCount: 5,
      positiveCount: 0,
      negativeCount: 3,
      feedbackScore: -0.8,
    };

    expect(summary.feedbackScore).toBe(-0.8);
  });
});

describe('ALMA Client - Feedback methods', () => {
  let alma: ALMA;

  beforeEach(() => {
    jest.clearAllMocks();
    alma = new ALMA({
      baseUrl: 'http://localhost:8765',
      projectId: 'test-project',
    });
  });

  describe('recordFeedback()', () => {
    it('should record feedback successfully', async () => {
      const mockResponse = {
        success: true,
        feedbackId: 'fb-123',
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const feedbackId = await alma.recordFeedback({
        memoryId: 'heur-1',
        memoryType: 'heuristic',
        query: 'form validation',
        agent: 'dev-agent',
        signal: FeedbackSignal.USED,
      });

      expect(feedbackId).toBe('fb-123');
    });

    it('should pass all parameters to server', async () => {
      setupMockResponse(createMCPResponse({ success: true, feedbackId: 'fb-1' }));

      await alma.recordFeedback({
        memoryId: 'heur-1',
        memoryType: 'heuristic',
        query: 'test query',
        agent: 'test-agent',
        signal: FeedbackSignal.THUMBS_UP,
        metadata: { reason: 'helpful' },
      });

      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.params.arguments.memory_id).toBe('heur-1');
      expect(callBody.params.arguments.memory_type).toBe('heuristic');
      expect(callBody.params.arguments.query).toBe('test query');
      expect(callBody.params.arguments.agent).toBe('test-agent');
      expect(callBody.params.arguments.project_id).toBe('test-project');
      expect(callBody.params.arguments.signal).toBe('thumbs_up');
      expect(callBody.params.arguments.metadata).toEqual({ reason: 'helpful' });
    });

    it('should throw ValidationError if memoryId is empty', async () => {
      await expect(
        alma.recordFeedback({
          memoryId: '',
          memoryType: 'heuristic',
          query: 'test',
          agent: 'agent',
          signal: FeedbackSignal.USED,
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError if agent is empty', async () => {
      await expect(
        alma.recordFeedback({
          memoryId: 'heur-1',
          memoryType: 'heuristic',
          query: 'test',
          agent: '',
          signal: FeedbackSignal.USED,
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError if query is empty', async () => {
      await expect(
        alma.recordFeedback({
          memoryId: 'heur-1',
          memoryType: 'heuristic',
          query: '',
          agent: 'agent',
          signal: FeedbackSignal.USED,
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ServerError on failed response', async () => {
      setupMockResponse(
        createMCPResponse({ success: false, error: 'Memory not found' }),
      );

      await expect(
        alma.recordFeedback({
          memoryId: 'nonexistent',
          memoryType: 'heuristic',
          query: 'test',
          agent: 'agent',
          signal: FeedbackSignal.USED,
        }),
      ).rejects.toThrow(ServerError);
    });

    it('should support all feedback signals', async () => {
      for (const signal of [
        FeedbackSignal.USED,
        FeedbackSignal.IGNORED,
        FeedbackSignal.THUMBS_UP,
        FeedbackSignal.THUMBS_DOWN,
      ]) {
        jest.clearAllMocks();
        setupMockResponse(createMCPResponse({ success: true, feedbackId: `fb-${signal}` }));

        const id = await alma.recordFeedback({
          memoryId: 'heur-1',
          memoryType: 'heuristic',
          query: 'test',
          agent: 'agent',
          signal,
        });

        expect(id).toBe(`fb-${signal}`);
      }
    });
  });

  describe('recordUsage()', () => {
    it('should record batch usage successfully', async () => {
      const mockResponse = {
        success: true,
        feedbackIds: ['fb-1', 'fb-2', 'fb-3'],
      };

      setupMockResponse(createMCPResponse(mockResponse));

      const ids = await alma.recordUsage({
        memoryIds: ['heur-1', 'heur-2', 'heur-3'],
        memoryType: 'heuristic',
        query: 'form validation',
        agent: 'dev-agent',
      });

      expect(ids).toEqual(['fb-1', 'fb-2', 'fb-3']);
    });

    it('should pass all parameters to server', async () => {
      setupMockResponse(
        createMCPResponse({ success: true, feedbackIds: ['fb-1'] }),
      );

      await alma.recordUsage({
        memoryIds: ['heur-1'],
        memoryType: 'domain_knowledge',
        query: 'auth patterns',
        agent: 'test-agent',
        metadata: { batch: true },
      });

      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.params.arguments.memory_ids).toEqual(['heur-1']);
      expect(callBody.params.arguments.memory_type).toBe('domain_knowledge');
      expect(callBody.params.arguments.query).toBe('auth patterns');
      expect(callBody.params.arguments.agent).toBe('test-agent');
      expect(callBody.params.arguments.project_id).toBe('test-project');
      expect(callBody.params.arguments.metadata).toEqual({ batch: true });
    });

    it('should throw ValidationError if memoryIds is empty', async () => {
      await expect(
        alma.recordUsage({
          memoryIds: [],
          memoryType: 'heuristic',
          query: 'test',
          agent: 'agent',
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ValidationError if agent is empty', async () => {
      await expect(
        alma.recordUsage({
          memoryIds: ['heur-1'],
          memoryType: 'heuristic',
          query: 'test',
          agent: '',
        }),
      ).rejects.toThrow(ValidationError);
    });

    it('should throw ServerError on failed response', async () => {
      setupMockResponse(
        createMCPResponse({ success: false, error: 'Batch failed' }),
      );

      await expect(
        alma.recordUsage({
          memoryIds: ['heur-1'],
          memoryType: 'heuristic',
          query: 'test',
          agent: 'agent',
        }),
      ).rejects.toThrow(ServerError);
    });

    it('should handle single memory ID', async () => {
      setupMockResponse(
        createMCPResponse({ success: true, feedbackIds: ['fb-solo'] }),
      );

      const ids = await alma.recordUsage({
        memoryIds: ['solo-1'],
        memoryType: 'outcome',
        query: 'test',
        agent: 'agent',
      });

      expect(ids).toEqual(['fb-solo']);
    });
  });
});

describe('Error classes', () => {
  describe('ALMAError', () => {
    it('should create error with message and code', () => {
      const error = new ALMAError('Test error', 'TEST_CODE');

      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_CODE');
      expect(error.name).toBe('ALMAError');
    });

    it('should serialize to JSON', () => {
      const error = new ALMAError('Test', 'CODE', { statusCode: 500 });
      const json = error.toJSON();

      expect(json.name).toBe('ALMAError');
      expect(json.message).toBe('Test');
      expect(json.code).toBe('CODE');
      expect(json.statusCode).toBe(500);
    });
  });

  describe('ValidationError', () => {
    it('should include field information', () => {
      const error = new ValidationError('Invalid field', {
        field: 'username',
        value: '',
      });

      expect(error.field).toBe('username');
      expect(error.statusCode).toBe(400);
    });
  });

  describe('ConnectionError', () => {
    it('should include URL information', () => {
      const error = new ConnectionError('Failed', {
        url: 'http://localhost:8765',
      });

      expect(error.url).toBe('http://localhost:8765');
    });
  });

  describe('ServerError', () => {
    it('should include server message and RPC code', () => {
      const error = new ServerError('Server error', {
        serverMessage: 'Internal error',
        rpcCode: -32603,
      });

      expect(error.serverMessage).toBe('Internal error');
      expect(error.rpcCode).toBe(-32603);
    });
  });
});
