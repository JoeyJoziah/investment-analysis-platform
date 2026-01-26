---
name: workflow-phase
description: Manage workflow phase execution and control
version: 1.0.0
invocable: true
aliases: [/wp, /phase]
---

# Workflow Phase Management

Control workflow phase execution, checkpoints, and transitions.

## Command Syntax

```
/workflow <action> [phase] [options]
```

## Actions

| Action | Description |
|--------|-------------|
| `pause` | Pause the current workflow |
| `resume` | Resume a paused workflow |
| `approve` | Approve a pending checkpoint |
| `skip <phase>` | Skip a specific phase |
| `abort` | Cancel the current workflow |
| `retry` | Retry the current phase |
| `rollback` | Rollback to previous phase |
| `goto <phase>` | Jump to a specific phase (advanced) |

## Checkpoint Approval

### /workflow approve

Approve the current checkpoint to continue execution.

```
╔══════════════════════════════════════════════════════════════╗
║                    CHECKPOINT APPROVAL                        ║
╠══════════════════════════════════════════════════════════════╣
║ Phase: REVIEW                                                 ║
║ Checkpoint Type: Quality Gate                                 ║
║                                                               ║
║ Summary:                                                      ║
║ • Code review completed by code-review-swarm                  ║
║ • Security scan: PASSED                                       ║
║ • Issues: 0 Critical, 0 High, 2 Medium                       ║
║                                                               ║
║ Medium Issues:                                                ║
║ 1. [MEDIUM] Missing JSDoc in auth/service.ts:42              ║
║ 2. [MEDIUM] Consider caching in portfolio/cache.ts:88        ║
║                                                               ║
║ Recommendation: APPROVE (no blocking issues)                  ║
╠══════════════════════════════════════════════════════════════╣
║ Type '/workflow approve' to continue                          ║
║ Type '/workflow approve --with-notes "..."' to add notes     ║
║ Type '/workflow abort' to cancel workflow                     ║
╚══════════════════════════════════════════════════════════════╝
```

### Approval Options

| Option | Description |
|--------|-------------|
| `--with-notes "..."` | Add approval notes |
| `--defer-medium` | Defer medium issues to separate ticket |
| `--force` | Force approve (use with caution) |

## Phase Control

### /workflow pause

Pause the current workflow execution.

```bash
/workflow pause
# Output: Workflow paused at BUILD phase. Use '/workflow resume' to continue.
```

### /workflow resume

Resume a paused workflow.

```bash
/workflow resume
# Output: Resuming workflow from BUILD phase...
```

### /workflow skip <phase>

Skip a specific phase.

```bash
/workflow skip design
# Output: Skipping DESIGN phase. Proceeding to BUILD...
```

**Restrictions:**
- Cannot skip mandatory checkpoint phases without `--force`
- Cannot skip already completed phases
- Skipping records reason in workflow history

### /workflow abort

Cancel the current workflow.

```bash
/workflow abort
# Output: Aborting workflow. Saving state for potential resume...
```

**Options:**
| Option | Description |
|--------|-------------|
| `--save-state` | Save state for later resume (default) |
| `--cleanup` | Remove all workflow artifacts |
| `--reason "..."` | Record abort reason |

### /workflow retry

Retry the current phase from the beginning.

```bash
/workflow retry
# Output: Retrying BUILD phase...
```

**Options:**
| Option | Description |
|--------|-------------|
| `--fresh` | Clear phase cache and retry |
| `--with-changes` | Apply pending changes before retry |

### /workflow rollback

Rollback to the previous phase.

```bash
/workflow rollback
# Output: Rolling back from BUILD to DESIGN...
```

**Restrictions:**
- Can only rollback one phase at a time
- Approved checkpoints require re-approval
- All phase outputs are preserved

### /workflow goto <phase>

Jump to a specific phase (advanced use only).

```bash
/workflow goto review
# Output: Jumping to REVIEW phase. Marking intermediate phases as skipped.
```

**Restrictions:**
- Requires `--force` flag
- Records skip reason for all intermediate phases
- May cause workflow inconsistencies

## Phase Information

### /workflow phases

List all phases and their configuration.

```
╔══════════════════════════════════════════════════════════════╗
║                    WORKFLOW PHASES                            ║
╠══════════════════════════════════════════════════════════════╣
║ Phase      │ Checkpoint │ Topology    │ Coordinator           ║
╠════════════╪════════════╪═════════════╪═══════════════════════╣
║ 1. INTAKE  │ Required   │ Star        │ team-coordinator      ║
║ 2. DESIGN  │ Conditional│ Hierarchical│ architecture-reviewer ║
║ 3. BUILD   │ Auto       │ Mesh        │ coder                 ║
║ 4. REVIEW  │ Conditional│ Parallel    │ project-quality-swarm ║
║ 5. INTEGRATE│ Required  │ Hierarchical│ github-swarm-coord    ║
║ 6. DEPLOY  │ Required   │ Hierarchical│ release-manager       ║
║ 7. LEARN   │ Auto       │ Star        │ memory-coordinator    ║
║ 8. SYNC    │ Auto       │ Sequential  │ doc-updater           ║
╚══════════════════════════════════════════════════════════════╝
```

### /workflow phase <name>

Get detailed information about a specific phase.

```bash
/workflow phase build
```

```
╔══════════════════════════════════════════════════════════════╗
║                    PHASE: BUILD                               ║
╠══════════════════════════════════════════════════════════════╣
║ Description: Implement code with test-driven development      ║
║ Order: 3/8                                                    ║
║ Topology: Mesh (parallel execution)                           ║
║ Coordinator: coder                                            ║
╠══════════════════════════════════════════════════════════════╣
║ AGENTS                                                        ║
║ • tdd-guide - Enforce test-first methodology                  ║
║ • coder - Implementation specialist                           ║
║ • build-error-resolver - Auto-fix build errors               ║
║ • backend-api-swarm - Backend development                     ║
║ • ui-visualization-swarm - Frontend development               ║
║ • data-ml-pipeline-swarm - Data pipeline development          ║
║ • financial-analysis-swarm - Financial logic                  ║
╠══════════════════════════════════════════════════════════════╣
║ COMMANDS                                                      ║
║ /tdd, /sparc-tdd, /code, /sparc-code, /build-fix             ║
║ /batch-executor, /parallel-execute                            ║
╠══════════════════════════════════════════════════════════════╣
║ QUALITY GATES                                                 ║
║ • Tests must pass                                             ║
║ • Coverage >= 80%                                             ║
║ • Build must succeed                                          ║
╠══════════════════════════════════════════════════════════════╣
║ AUTO-ACTIONS                                                  ║
║ • On build error: Activate build-error-resolver (3 retries)  ║
╚══════════════════════════════════════════════════════════════╝
```

## State Management

Phase state is stored in `.claude/memory/workflow-state.json`:

```json
{
  "workflow_id": "wf_abc123",
  "workflow_type": "feature",
  "task_description": "Implement user authentication",
  "current_phase": "build",
  "phase_index": 3,
  "status": "in_progress",
  "started_at": "2026-01-26T10:30:00Z",
  "phases": {
    "intake": { "status": "completed", "approved_at": "..." },
    "design": { "status": "completed", "approved_at": "..." },
    "build": { "status": "in_progress", "started_at": "..." },
    "review": { "status": "pending" },
    "integrate": { "status": "pending" },
    "deploy": { "status": "pending" },
    "learn": { "status": "pending" },
    "sync": { "status": "pending" }
  },
  "outputs": {},
  "metrics": {}
}
```

## Examples

```bash
# Approve checkpoint
/workflow approve

# Approve with notes
/workflow approve --with-notes "Reviewed security findings, acceptable for staging"

# Pause workflow
/workflow pause

# Resume workflow
/workflow resume

# Skip design phase
/workflow skip design

# Retry current phase
/workflow retry --fresh

# Abort with reason
/workflow abort --reason "Requirements changed, need to restart"

# Get phase info
/workflow phase review
```
