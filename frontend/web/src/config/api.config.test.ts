/**
 * F-12-001 (PRD audit 2026-04 / Workstream F): regression test.
 *
 * Every endpoint string in apiConfig.endpoints must start with /api/v1/.
 * Pre-fix code used /api/ (no version segment), causing 40+ endpoints to 404
 * because the backend mounts every router under /api/v1/ (see
 * backend/api/main.py:333-348).
 */

import { describe, it, expect } from 'vitest';
import { apiConfig } from './api.config';

const PREFIX_RE = /^\/api\/v1\//;

function collectEndpointStrings(node: unknown, out: string[] = []): string[] {
  if (typeof node === 'string') {
    if (node.startsWith('/api')) {
      out.push(node);
    }
    return out;
  }
  if (typeof node === 'function') {
    // Endpoint factory — invoke with safe placeholders to inspect output.
    try {
      const result = (node as (...args: unknown[]) => string)(
        'TICKER',
        1,
        2,
      );
      if (typeof result === 'string' && result.startsWith('/api')) {
        out.push(result);
      }
    } catch {
      // Some factories may require specific arity; ignore failures here.
    }
    return out;
  }
  if (node && typeof node === 'object') {
    for (const value of Object.values(node as Record<string, unknown>)) {
      collectEndpointStrings(value, out);
    }
  }
  return out;
}

describe('apiConfig endpoints (F-12-001)', () => {
  const endpoints = collectEndpointStrings(apiConfig.endpoints);

  it('produces a non-trivial number of endpoints', () => {
    expect(endpoints.length).toBeGreaterThan(20);
  });

  it('every endpoint begins with /api/v1/', () => {
    const offenders = endpoints.filter((e) => !PREFIX_RE.test(e));
    expect(offenders).toEqual([]);
  });
});
