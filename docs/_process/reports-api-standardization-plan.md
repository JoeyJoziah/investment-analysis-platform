# API Standardization & Hooks Consolidation Plan

**Date**: January 27, 2026
**Project**: Investment Analysis Platform
**Scope**: Hooks configuration consolidation and API response format standardization
**Status**: PLANNING PHASE - No code modifications

---

## Executive Summary

This document outlines the plan for two related standardization efforts:

1. **Hooks Configuration Consolidation** - Unifying scattered hook configurations across `.claude/hooks/hooks.json` and `.claude/settings.json` into a single V3-standard configuration
2. **API Response Standardization** - Standardizing the response format across all 13 backend routers to use a consistent `ApiResponse<T>` wrapper

---

## Part 1: Hooks Configuration Consolidation

### 1.1 Current State Analysis

Hook configurations are currently scattered across multiple files:

| File | Hook Types | Format | Purpose |
|------|------------|--------|---------|
| `.claude/hooks/hooks.json` | 10 types | V2+ custom | Project-specific hooks |
| `.claude/settings.json` | 6 types | V3 regex | Claude Flow integration |
| `.claude/settings.local.json` | N/A | Permissions only | Local permissions |

### 1.2 Detailed Hook Inventory

**Source 1: `.claude/hooks/hooks.json`**

- **PreToolUse (5 hooks)**: Dev server blocker, tmux reminder, git push review, doc blocker, compact suggester
- **PreCompact (1 hook)**: Memory persistence
- **SessionStart (1 hook)**: Load context
- **PostToolUse (5 hooks)**: PR logger, board sync, Prettier, TypeScript check, console.log warning
- **Stop (1 hook)**: Console.log audit
- **SessionEnd (5 hooks)**: Memory persistence, board sync, pattern evaluation, learning pipeline, workflow pause
- **WorkflowPhaseStart/Complete/Checkpoint (4 hooks)**: Phase tracking, quality gates
- **TaskComplete (1 hook)**: Board sync

**Source 2: `.claude/settings.json`**

- **PreToolUse (3 hooks)**: CLI pre-edit, pre-command, pre-task
- **PostToolUse (3 hooks)**: CLI post-edit, post-command, post-task
- **UserPromptSubmit (1 hook)**: Route task
- **SessionStart (1 hook)**: Daemon + session restore
- **Stop (1 hook)**: Return OK
- **Notification (1 hook)**: Store notifications

### 1.3 Consolidation Strategy

**Target**: Single `.claude/config.json` with merged hooks in execution order:
1. V3 CLI hooks first (learning/tracking)
2. Project-specific hooks second

### 1.4 Migration Steps

1. Create backup of current configurations
2. Merge hooks by type, respecting execution order
3. Test each hook individually
4. Remove deprecated files

**Estimated Effort**: 3-4 hours

---

## Part 2: API Response Standardization

### 2.1 Target Standard Interface

```typescript
interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  meta?: {
    total: number
    page: number
    limit: number
  }
}
```

### 2.2 Router Analysis Summary (13 Routers)

| Router | Endpoints | Current Pattern | Key Issues |
|--------|-----------|-----------------|------------|
| stocks.py | 9 | Pydantic models | No wrapper, embedded pagination |
| auth.py | 5 | Mixed (Token, Dict) | Inconsistent structures |
| portfolio.py | 8 | Pydantic + Dict | Dict for add/remove |
| analysis.py | 5 | Pydantic + Dict | No schema for technical |
| watchlist.py | 8 | Pydantic models | 204 for deletes |
| health.py | 2 | Dict only | No Pydantic at all |
| recommendations.py | 8 | Pydantic + Dict | SEC compliance mixed with Dict |
| monitoring.py | 5 | Dict only | No standardization |
| admin.py | 10 | Pydantic + Dict | Security ops use Dict |
| agents.py | 6 | Pydantic + Dict | Batch uses Dict |
| gdpr.py | 9 | Pydantic models | **Most consistent** |
| thesis.py | 6 | Pydantic models | Good, 204 for delete |
| cache_management.py | 6 | Pydantic + Dict | Mixed patterns |

### 2.3 Key Inconsistencies Found

| Category | Occurrences | Impact |
|----------|-------------|--------|
| No response wrapper | 13/13 routers | All responses lack success/error envelope |
| Dict instead of Pydantic | 45+ endpoints | Poor API documentation, no validation |
| Inconsistent pagination | 8 patterns | Frontend must handle multiple formats |
| Delete returns None vs Dict | 3 routers | Inconsistent client handling |
| Error format varies | All | HTTPException detail: str vs dict |

### 2.4 Migration Plan

**Phase 1**: Create `backend/models/api_response.py` with standard models
**Phase 2**: Create `backend/middleware/error_handler.py` for consistent errors
**Phase 3**: Migrate routers in priority order (health → auth → stocks → ... → cache_management)
**Phase 4**: Update frontend API client and types
**Phase 5**: Testing and documentation

**Estimated Total Effort**: 27 hours

### 2.5 Breaking Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| Response wrapper added | Frontend must unwrap `.data` | API versioning (v2/) |
| Pagination in `meta` | Frontend pagination logic changes | Deprecation period |
| Error format standardized | Error handling code changes | Consistent `error` field |

---

## Part 3: Implementation Checklist

### Hooks Consolidation
- [ ] Backup current configurations
- [ ] Create merged `.claude/config.json`
- [ ] Test all hook types
- [ ] Remove deprecated files
- [ ] Update documentation

### API Standardization
- [ ] Create `api_response.py` models
- [ ] Create `error_handler.py` middleware
- [ ] Migrate all 13 routers
- [ ] Update frontend client
- [ ] E2E testing
- [ ] API documentation

---

## Appendix: File Locations

**Hook Configuration Sources**:
- `.claude/hooks/hooks.json`
- `.claude/settings.json`
- `.claude/settings.local.json`

**Backend Routers** (all in `backend/api/routers/`):
- `stocks.py`, `auth.py`, `portfolio.py`, `analysis.py`, `watchlist.py`
- `health.py`, `recommendations.py`, `monitoring.py`, `admin.py`
- `agents.py`, `gdpr.py`, `thesis.py`, `cache_management.py`

**Document Status**: Ready for review - Implementation requires approval
