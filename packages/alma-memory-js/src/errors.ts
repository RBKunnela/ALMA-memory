/**
 * ALMA Memory Error Classes
 *
 * Custom exceptions for the ALMA TypeScript SDK, providing clear error
 * categorization for connection, validation, and operation errors.
 */

/**
 * Base error class for all ALMA errors.
 *
 * @example
 * ```typescript
 * try {
 *   await alma.retrieve({ query: 'test', agent: 'my-agent' });
 * } catch (error) {
 *   if (error instanceof ALMAError) {
 *     console.error(`ALMA Error [${error.code}]: ${error.message}`);
 *   }
 * }
 * ```
 */
export class ALMAError extends Error {
  /** Error code for programmatic handling */
  public readonly code: string;
  /** HTTP status code if applicable */
  public readonly statusCode?: number;
  /** Original error that caused this error */
  public readonly cause?: Error;

  constructor(
    message: string,
    code: string = 'ALMA_ERROR',
    options?: { statusCode?: number; cause?: Error }
  ) {
    super(message);
    this.name = 'ALMAError';
    this.code = code;
    this.statusCode = options?.statusCode;
    this.cause = options?.cause;

    // Maintains proper stack trace for where error was thrown (V8 engines)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ALMAError);
    }
  }

  /**
   * Convert error to a JSON-serializable object.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
    };
  }
}

/**
 * Raised when connection to the ALMA server fails.
 *
 * Common causes:
 * - Server not running
 * - Wrong URL
 * - Network issues
 * - Timeout
 *
 * @example
 * ```typescript
 * try {
 *   await alma.health();
 * } catch (error) {
 *   if (error instanceof ConnectionError) {
 *     console.error('Failed to connect to ALMA server');
 *     // Maybe show retry UI or check server status
 *   }
 * }
 * ```
 */
export class ConnectionError extends ALMAError {
  /** The URL that failed to connect */
  public readonly url?: string;

  constructor(
    message: string,
    options?: { url?: string; cause?: Error }
  ) {
    super(message, 'CONNECTION_ERROR', { cause: options?.cause });
    this.name = 'ConnectionError';
    this.url = options?.url;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      url: this.url,
    };
  }
}

/**
 * Raised when input validation fails.
 *
 * Common causes:
 * - Missing required fields
 * - Invalid field values
 * - Type mismatches
 *
 * @example
 * ```typescript
 * try {
 *   await alma.learn({ agent: '', task: '', outcome: 'success', strategyUsed: '' });
 * } catch (error) {
 *   if (error instanceof ValidationError) {
 *     console.error(`Invalid input: ${error.field} - ${error.message}`);
 *   }
 * }
 * ```
 */
export class ValidationError extends ALMAError {
  /** The field that failed validation */
  public readonly field?: string;
  /** The value that was invalid */
  public readonly value?: unknown;

  constructor(
    message: string,
    options?: { field?: string; value?: unknown }
  ) {
    super(message, 'VALIDATION_ERROR', { statusCode: 400 });
    this.name = 'ValidationError';
    this.field = options?.field;
    this.value = options?.value;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      field: this.field,
    };
  }
}

/**
 * Raised when a requested resource is not found.
 *
 * Common causes:
 * - Invalid memory ID
 * - Agent not registered
 * - Resource deleted
 *
 * @example
 * ```typescript
 * try {
 *   await alma.stats('unknown-agent');
 * } catch (error) {
 *   if (error instanceof NotFoundError) {
 *     console.error(`Resource not found: ${error.resourceType} ${error.resourceId}`);
 *   }
 * }
 * ```
 */
export class NotFoundError extends ALMAError {
  /** Type of resource that was not found */
  public readonly resourceType?: string;
  /** ID of the resource that was not found */
  public readonly resourceId?: string;

  constructor(
    message: string,
    options?: { resourceType?: string; resourceId?: string }
  ) {
    super(message, 'NOT_FOUND', { statusCode: 404 });
    this.name = 'NotFoundError';
    this.resourceType = options?.resourceType;
    this.resourceId = options?.resourceId;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      resourceType: this.resourceType,
      resourceId: this.resourceId,
    };
  }
}

/**
 * Raised when an agent attempts to learn outside its scope.
 *
 * The MemoryScope system prevents agents from learning in forbidden domains.
 * This error indicates that the operation violated scope constraints.
 *
 * @example
 * ```typescript
 * try {
 *   await alma.addKnowledge({
 *     agent: 'test-agent',
 *     domain: 'forbidden-domain',
 *     fact: 'Some fact'
 *   });
 * } catch (error) {
 *   if (error instanceof ScopeViolationError) {
 *     console.error(`Agent cannot learn in domain: ${error.domain}`);
 *   }
 * }
 * ```
 */
export class ScopeViolationError extends ALMAError {
  /** Agent that attempted the violation */
  public readonly agent?: string;
  /** Domain that was attempted */
  public readonly domain?: string;

  constructor(
    message: string,
    options?: { agent?: string; domain?: string }
  ) {
    super(message, 'SCOPE_VIOLATION', { statusCode: 403 });
    this.name = 'ScopeViolationError';
    this.agent = options?.agent;
    this.domain = options?.domain;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      agent: this.agent,
      domain: this.domain,
    };
  }
}

/**
 * Raised when request times out.
 *
 * @example
 * ```typescript
 * try {
 *   await alma.retrieve({ query: 'test', agent: 'my-agent' });
 * } catch (error) {
 *   if (error instanceof TimeoutError) {
 *     console.error(`Request timed out after ${error.timeout}ms`);
 *   }
 * }
 * ```
 */
export class TimeoutError extends ALMAError {
  /** Timeout value in milliseconds */
  public readonly timeout: number;

  constructor(message: string, timeout: number) {
    super(message, 'TIMEOUT', { statusCode: 408 });
    this.name = 'TimeoutError';
    this.timeout = timeout;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      timeout: this.timeout,
    };
  }
}

/**
 * Raised when the server returns an error response.
 *
 * @example
 * ```typescript
 * try {
 *   await alma.retrieve({ query: 'test', agent: 'my-agent' });
 * } catch (error) {
 *   if (error instanceof ServerError) {
 *     console.error(`Server error: ${error.serverMessage}`);
 *   }
 * }
 * ```
 */
export class ServerError extends ALMAError {
  /** Error message from the server */
  public readonly serverMessage?: string;
  /** JSON-RPC error code */
  public readonly rpcCode?: number;

  constructor(
    message: string,
    options?: { serverMessage?: string; rpcCode?: number; statusCode?: number }
  ) {
    super(message, 'SERVER_ERROR', { statusCode: options?.statusCode ?? 500 });
    this.name = 'ServerError';
    this.serverMessage = options?.serverMessage;
    this.rpcCode = options?.rpcCode;
  }

  toJSON(): Record<string, unknown> {
    return {
      ...super.toJSON(),
      serverMessage: this.serverMessage,
      rpcCode: this.rpcCode,
    };
  }
}

/**
 * Type guard to check if an error is an ALMAError.
 */
export function isALMAError(error: unknown): error is ALMAError {
  return error instanceof ALMAError;
}

/**
 * Type guard to check if an error is a ConnectionError.
 */
export function isConnectionError(error: unknown): error is ConnectionError {
  return error instanceof ConnectionError;
}

/**
 * Type guard to check if an error is a ValidationError.
 */
export function isValidationError(error: unknown): error is ValidationError {
  return error instanceof ValidationError;
}

/**
 * Type guard to check if an error is a NotFoundError.
 */
export function isNotFoundError(error: unknown): error is NotFoundError {
  return error instanceof NotFoundError;
}

/**
 * Type guard to check if an error is a ScopeViolationError.
 */
export function isScopeViolationError(error: unknown): error is ScopeViolationError {
  return error instanceof ScopeViolationError;
}

/**
 * Type guard to check if an error is a TimeoutError.
 */
export function isTimeoutError(error: unknown): error is TimeoutError {
  return error instanceof TimeoutError;
}

/**
 * Type guard to check if an error is a ServerError.
 */
export function isServerError(error: unknown): error is ServerError {
  return error instanceof ServerError;
}
