---
name: verification-loop
description: Autonomous execution engine with RARV-cycle task orchestration, self-healing, live task registry management, and iterative perfection loops. Includes 6-phase verification (build, types, lint, tests, security, diff) plus full Phases 5-9 autonomous development. Use after completing features, before PRs, or as the execution engine after /sparc:integration generates a validated master plan.
---

# Verification Loop

A comprehensive verification and autonomous execution system for Claude Code sessions. Combines quick 6-phase quality gates with full RARV-cycle task orchestration for autonomous development.

## When to Use

Invoke this skill:
- After completing a feature or significant code change (quick mode)
- Before creating a PR (quick mode)
- When you want to ensure quality gates pass (quick mode)
- After refactoring (quick mode)
- To execute all tasks from a MASTER_TODO_REGISTRY (autonomous mode)
- As the final stage of the `/repo-integrator` -> `/sparc:integration` -> `/verification-loop` pipeline

---

# QUICK VERIFICATION MODE

## Verification Phases

### Phase 1: Build Verification
```bash
# Check if project builds
npm run build 2>&1 | tail -20
# OR
pnpm build 2>&1 | tail -20
```

If build fails, STOP and fix before continuing.

### Phase 2: Type Check
```bash
# TypeScript projects
npx tsc --noEmit 2>&1 | head -30

# Python projects
pyright . 2>&1 | head -30
```

Report all type errors. Fix critical ones before continuing.

### Phase 3: Lint Check
```bash
# JavaScript/TypeScript
npm run lint 2>&1 | head -30

# Python
ruff check . 2>&1 | head -30
```

### Phase 4: Test Suite
```bash
# Run tests with coverage
npm run test -- --coverage 2>&1 | tail -50

# Check coverage threshold
# Target: 80% minimum
```

Report:
- Total tests: X
- Passed: X
- Failed: X
- Coverage: X%

### Phase 5: Security Scan
```bash
# Check for secrets
grep -rn "sk-" --include="*.ts" --include="*.js" . 2>/dev/null | head -10
grep -rn "api_key" --include="*.ts" --include="*.js" . 2>/dev/null | head -10

# Check for console.log
grep -rn "console.log" --include="*.ts" --include="*.tsx" src/ 2>/dev/null | head -10
```

### Phase 6: Diff Review
```bash
# Show what changed
git diff --stat
git diff HEAD~1 --name-only
```

Review each changed file for:
- Unintended changes
- Missing error handling
- Potential edge cases

## Quick Verification Output Format

```
VERIFICATION REPORT
==================

Build:     [PASS/FAIL]
Types:     [PASS/FAIL] (X errors)
Lint:      [PASS/FAIL] (X warnings)
Tests:     [PASS/FAIL] (X/Y passed, Z% coverage)
Security:  [PASS/FAIL] (X issues)
Diff:      [X files changed]

Overall:   [READY/NOT READY] for PR

Issues to Fix:
1. ...
2. ...
```

## Continuous Mode

For long sessions, run verification every 15 minutes or after major changes:

```markdown
Set a mental checkpoint:
- After completing each function
- After finishing a component
- Before moving to next task

Run: /verification-loop
```

## Integration with Hooks

This skill complements PostToolUse hooks but provides deeper verification.
Hooks catch issues immediately; this skill provides comprehensive review.

---

# AUTONOMOUS EXECUTION MODE (Phases 5-9)

## Overview

When a `.project-intelligence/MASTER_TODO_REGISTRY.json` exists (created by `/repo-integrator`) and a `MASTER_PLAN_V2_OPTIMIZED.md` has been generated (by `/sparc:integration`), this mode executes ALL tasks using RARV cycles, self-healing, and real-time task registry updates.

## Pipeline Position

```
/repo-integrator       Phase 1: Audit + Task Genesis
        |              (8 swarms, generates MASTER_TODO_REGISTRY.json)
        v
/sparc:integration     Phases 2-4: Plan + Research + Refine
        |              (generates MASTER_PLAN_V2_OPTIMIZED.md)
        v
/verification-loop     Phases 5-9: Execute + Validate + Document + Commit + Iterate
                       (RARV cycles until perfection or human stop)
```

## Prerequisites

- `.project-intelligence/MASTER_TODO_REGISTRY.json` populated
- `MASTER_PLAN_V2_OPTIMIZED.md` generated
- `.project-intelligence/state/orchestrator.json` at PHASE_5_EXECUTION
- `.project-intelligence/CONTINUITY.md` current

## Execution Framework

- Primary: Loki Mode RARV cycle + Auto-Orchestrator
- Task Tracking: Real-time registry updates (pending -> in-progress -> completed)
- Memory: CONTINUITY.md (read/write EVERY turn) + session-end hooks
- Healing: Self-healing with task retry + strategy adaptation
- Parallelization: Up to 10 agents simultaneously

## Initialization

```
/loki-mode initialize rarv-cycle=continuous
/automation self-healing enable
/automation session-memory enable
/hive-mind-init strategy=consensus memory=persistent
/hooks-setup enable-all
/monitoring real-time-view start
```

---

## PHASE 5: Autonomous Task Execution Engine

### Execution Loop (runs until registry is 100% complete)

#### Step 5.1: REASON (Load Context) — Every Turn

1. **Read CONTINUITY.md** — current status, last completed task, active agents, blocked tasks
2. **Read orchestrator.json** — phase, task counts, quality score
3. **Read queue/pending.json** — tasks ready for execution (sorted by priority)
4. **Read queue/in-progress.json** — currently executing tasks with RARV state

5. **DECIDE NEXT ACTION:**
   - If pending empty AND in-progress empty AND blocked not empty: check if blocking tasks completed, unblock and move to pending
   - If pending has tasks AND active_agents < 10: spawn new agents for next tasks
   - If in-progress tasks stuck (no updates in >10 minutes): investigate, trigger self-healing
   - If all tasks completed: advance to PHASE 6 (Validation)

#### Step 5.2: ACT (Execute Tasks with RARV)

For each task claimed from pending queue:

**Task Claim Protocol:**
```json
{
  "taskId": "TASK-EPIC2-007",
  "status": "in_progress",
  "assignedAgents": ["frontend-developer", "react-pro"],
  "claimedAt": "[timestamp]",
  "raRVCycle": {
    "iteration": 1,
    "currentStep": "REASON",
    "history": []
  }
}
```

**RARV Cycle Execution:**

**R - REASON:**
1. Read full task details from MASTER_TODO_REGISTRY.json
2. Load acceptance criteria
3. Review research findings and implementation notes
4. Check related files and dependencies
5. Formulate implementation strategy
6. Log reasoning to RARV history:
   ```json
   {
     "step": "REASON",
     "timestamp": "[timestamp]",
     "reasoning": "Task requires component with form validation. Using react-hook-form based on research.",
     "strategy": "TDD approach: write tests first, implement, verify"
   }
   ```

**A - ACT:**
1. Implement solution following strategy
2. Write code with coding-standards compliance
3. Create/update tests (TDD workflow)
4. Run linters and type checkers
5. Log actions to RARV history:
   ```json
   {
     "step": "ACT",
     "timestamp": "[timestamp]",
     "actions": ["Created component", "Added 12 unit tests", "Updated API integration"],
     "filesModified": ["src/components/Feature.tsx"],
     "linesAdded": 287,
     "linesDeleted": 34
   }
   ```

**R - REFLECT:**
1. Review implementation against acceptance criteria
2. Self-assess code quality (0.0-1.0)
3. Identify potential issues
4. Log self-assessment:
   ```json
   {
     "step": "REFLECT",
     "selfAssessment": {
       "acceptanceCriteriaMet": ["Criterion 1", "Criterion 2"],
       "acceptanceCriteriaPending": ["Criterion 3"],
       "codeQuality": 0.85,
       "potentialIssues": ["Edge case in validation"],
       "confidence": 0.80
     }
   }
   ```

**V - VERIFY:**
1. Run all tests (unit + integration)
2. Spawn verification swarm:
   - code-review-expert (code quality)
   - test-automator (test coverage)
   - security-auditor (security check)
3. Hive Mind consensus vote
4. Log verification:
   ```json
   {
     "step": "VERIFY",
     "verification": {
       "testResults": {
         "unit": {"passed": 12, "failed": 0},
         "integration": {"passed": 3, "failed": 0}
       },
       "codeReview": {"score": 0.88, "issues": ["Minor: Add JSDoc"]},
       "security": {"score": 1.0, "vulnerabilities": 0},
       "testCoverage": 94.2,
       "hiveMindConsensus": 0.87,
       "overallQuality": 0.89,
       "verdict": "PASS"
     }
   }
   ```

**Verification Outcomes:**

IF verdict = PASS:
- Update task status to "completed" in registry
- Move to queue/completed.json
- Increment orchestrator.json tasksCompleted
- Update CONTINUITY.md with success
- Store learned patterns in AgentDB procedural memory
- Check if blocked tasks can now unblock (dependency resolution)

IF verdict = FAIL:
- Enter self-healing loop
- Increment attempts counter (max 5)
- Adapt strategy based on failure reason
- Log to `.project-intelligence/signals/QUALITY_GATE_FAILURES.log`
- Retry with adapted strategy
- If still failing after 5 attempts: escalate to human via ESCALATIONS.log

#### Step 5.3: Continuous Task Queue Management

Background processes (run every 60 seconds):

1. **Dependency Resolution:**
   For each task in blocked queue, check if all dependencies are in completed queue. If so, move to pending and log unblock.

2. **Parallel Agent Spawning:**
   Count active agents. If < 10 and pending queue has tasks, spawn agents for next highest-priority tasks.

3. **Drift Detection:**
   If estimated completion time exceeds deadline, log DRIFT_DETECTED signal with mitigation (increase parallelization or reduce scope).

4. **CONTINUITY.md Updates (every 5 tasks completed):**
   ```markdown
   ## Progress Update [timestamp]
   Tasks completed: [N] / [total] ([percent]%)
   Tasks in progress: [N]
   Tasks blocked: [N]
   Average quality score: [score]
   Velocity: [N] tasks/hour
   ETA: [hours] remaining

   ## Recent Completions
   - TASK-ID: Title (quality: score)

   ## Active Focus
   Currently: [description]
   Next milestone: [description]
   ```

5. **Session Persistence (every 30 minutes + on session-end hook):**
   Save all in-progress task state, persist CONTINUITY.md, checkpoint orchestrator.json, store episodic memory to AgentDB, archive current state.

#### Step 5.4: Task Completion & Epic Milestones

When an epic completes:
1. Verify all tasks in epic are "completed"
2. Run epic-level integration tests
3. Quality gate: average task quality score > 0.85
4. Update task registry metadata with epic completion stats
5. Unblock dependent epics (move their tasks from blocked to pending)
6. Update CONTINUITY.md with epic milestone

---

## PHASE 6: Full Aggregation Validation

After all tasks complete:

1. **Load completed task registry** — verify 100% completion
2. **Run full system validation** (uses Quick Verification phases above):
   - Build: PASS required
   - Types: 0 errors required
   - Lint: 0 errors (warnings acceptable)
   - Tests: 100% passing, coverage >= 80%
   - Security: zero critical/high vulnerabilities
   - Performance benchmarks: all targets met
3. **Generate VALIDATION_REPORT_FINAL.md:**
   - Truth score calculation (target >= 0.95)
   - Issues found (with severity)
   - Remediation plan (if truth score < 0.95)
4. **Update task registry** with validation status

---

## PHASE 7: Documentation Update with Task Audit Trail

1. **Generate CHANGELOG.md from task registry:**
   ```markdown
   # CHANGELOG - Iteration [N]

   ## Added
   - Feature description (TASK-EPIC2-007, TASK-EPIC2-011)

   ## Changed
   - Refactoring description (TASK-EPIC6-003)

   ## Fixed
   - Bug fix description (TASK-EPIC3-015)
   ```

2. **Update README.md** with features extracted from completed task titles
3. **Archive task registry** to `.project-intelligence/archives/iteration-[N]/REGISTRY_FINAL.json`

---

## PHASE 8: Commit & Push with Task Traceability

Git commit format references task IDs:
```
feat(scope): Description

- Detail 1
- Detail 2

Tasks: TASK-EPIC1-008, TASK-EPIC1-012
Quality Score: 0.93
Closes #42
```

---

## PHASE 9: Iterative Perfection Loop

After Phase 8 completes:

1. **Archive current iteration** to `.project-intelligence/archives/iteration-[N]-COMPLETE`
2. **Reset orchestrator** for next iteration:
   ```json
   {
     "currentPhase": "PHASE_1_AUDIT",
     "iteration": "[N+1]",
     "previousIteration": {
       "number": "[N]",
       "tasksCompleted": "[count]",
       "averageQuality": "[score]"
     }
   }
   ```
3. **Loop back to Phase 1** (`/repo-integrator`) — re-audit project with completed tasks, generate NEW task registry
4. **Continue until ZERO improvements possible**

### Termination Criteria

```
IF Phase 1 audit generates 0 new tasks:
    PROJECT PERFECTION ACHIEVED
    orchestrator.status = "PERFECT"
    truthScore = 1.0
    STOP
```

---

## Training Data Extraction (After Each Iteration)

Extract and store:
1. **Task execution patterns** — which strategies worked best per task type
2. **RARV cycle learnings** — common failure modes and solutions
3. **Agent coordination patterns** — optimal swarm topologies per task type
4. **Effort estimation models** — improve future estimates from actuals vs. estimates

Store in:
- `.project-intelligence/memory/procedural/` (patterns)
- AgentDB semantic memory (embeddings for similarity search)
- ReasoningBank (reasoning patterns)

Use in future iterations:
- Phase 1 audit uses learned patterns to identify similar issues faster
- Phase 2 planning uses historical velocity for better estimates
- Phase 5 execution uses proven strategies first (skips previously failed approaches)

---

## Invocation

### Quick Mode (6-phase verification only)
```
/verification-loop
```

### Autonomous Mode (full Phases 5-9)
```
/loki-mode initialize rarv-cycle=continuous iterations=infinite
/auto-orchestrator tier=4 mode=autonomous
/hive-mind-spawn full-army
/monitoring real-time-view enable
/hooks session-end enable
/memory-persistence enable
```
