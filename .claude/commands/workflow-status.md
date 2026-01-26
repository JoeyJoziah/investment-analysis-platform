---
name: workflow-status
description: Display current workflow status and progress
version: 1.0.0
invocable: true
aliases: [/ws, /wstatus]
---

# Workflow Status Command

Display the current status of active workflows, phase progress, and quality metrics.

## Command Syntax

```
/workflow status [options]
```

## Options

| Option | Description |
|--------|-------------|
| `--verbose` | Show detailed phase information |
| `--metrics` | Include performance metrics |
| `--history` | Show recent workflow history |
| `--json` | Output as JSON |

## Status Display

### Active Workflow Status

```
╔══════════════════════════════════════════════════════════════╗
║                    WORKFLOW STATUS                            ║
╠══════════════════════════════════════════════════════════════╣
║ Workflow: feature                                             ║
║ Task: "Implement user authentication"                         ║
║ Started: 2026-01-26 10:30:00                                 ║
║ Current Phase: BUILD (3/8)                                    ║
║ Status: IN_PROGRESS                                           ║
╠══════════════════════════════════════════════════════════════╣
║ PHASES                                                        ║
║ ┌─────────────┬──────────┬──────────┬─────────────────────┐  ║
║ │ Phase       │ Status   │ Duration │ Checkpoint          │  ║
║ ├─────────────┼──────────┼──────────┼─────────────────────┤  ║
║ │ 1. INTAKE   │ ✅ Done  │ 5m 23s   │ Approved            │  ║
║ │ 2. DESIGN   │ ✅ Done  │ 8m 12s   │ Approved            │  ║
║ │ 3. BUILD    │ 🔄 Active│ 12m 45s  │ Auto                │  ║
║ │ 4. REVIEW   │ ⏳ Pending│ -        │ Required            │  ║
║ │ 5. INTEGRATE│ ⏳ Pending│ -        │ Required            │  ║
║ │ 6. DEPLOY   │ ⏳ Pending│ -        │ Required            │  ║
║ │ 7. LEARN    │ ⏳ Pending│ -        │ Auto                │  ║
║ │ 8. SYNC     │ ⏳ Pending│ -        │ Auto                │  ║
║ └─────────────┴──────────┴──────────┴─────────────────────┘  ║
╠══════════════════════════════════════════════════════════════╣
║ ACTIVE AGENTS                                                 ║
║ • coder (mesh) - Implementing auth service                    ║
║ • tdd-guide (mesh) - Running test suite                       ║
║ • build-error-resolver (standby)                              ║
╠══════════════════════════════════════════════════════════════╣
║ QUALITY METRICS                                               ║
║ • Test Coverage: 82% (target: 80%)                           ║
║ • Build Status: Passing                                       ║
║ • Issues Found: 0 Critical, 0 High, 3 Medium                 ║
╚══════════════════════════════════════════════════════════════╝
```

## Execution Instructions

1. **Load Workflow State**
   ```bash
   cat .claude/memory/workflow-state.json
   ```

2. **Display Phase Progress**
   - Show all phases with status indicators
   - Highlight current active phase
   - Show pending checkpoints

3. **Show Active Agents**
   - List currently executing agents
   - Show their topology configuration
   - Display current task

4. **Display Quality Metrics**
   - Load from .claude/memory/quality-report.md
   - Show test coverage
   - Show issue counts by severity

5. **Show Checkpoint Status**
   - Indicate which phases require approval
   - Show approval timestamps for completed checkpoints

## Phase Status Indicators

| Indicator | Meaning |
|-----------|---------|
| ✅ Done | Phase completed successfully |
| 🔄 Active | Phase currently executing |
| ⏳ Pending | Phase waiting to start |
| ⏸️ Paused | Phase paused at checkpoint |
| ❌ Failed | Phase failed with errors |
| ⏭️ Skipped | Phase was skipped |

## Sub-commands

### /workflow status --verbose

Shows detailed information for each phase:
- Agent invocations
- Commands executed
- Outputs generated
- Time breakdown

### /workflow status --metrics

Shows performance metrics:
- Total tokens used
- Phase durations
- Agent efficiency
- Parallel execution stats

### /workflow status --history

Shows recent workflow history:
```
╔════════════════════════════════════════════════════════════════╗
║ WORKFLOW HISTORY (Last 5)                                      ║
╠════════════════════════════════════════════════════════════════╣
║ 1. bugfix - "Fix API timeout" - ✅ Completed - 23m - Jan 26   ║
║ 2. feature - "Add charts" - ✅ Completed - 1h 12m - Jan 25    ║
║ 3. refactor - "Extract utils" - ✅ Completed - 45m - Jan 25   ║
║ 4. hotfix - "Security patch" - ✅ Completed - 15m - Jan 24    ║
║ 5. feature - "User profiles" - ✅ Completed - 2h 5m - Jan 24  ║
╚════════════════════════════════════════════════════════════════╝
```

## Integration

The status command integrates with:
- `.claude/memory/workflow-state.json` - Active state
- `.claude/memory/workflow-metrics.json` - Metrics
- `.claude/memory/quality-report.md` - Quality data
- `.claude/memory/workflow-history.json` - History

## Examples

```bash
# Basic status
/workflow status

# Verbose with metrics
/workflow status --verbose --metrics

# Show history
/workflow status --history

# JSON output for scripting
/workflow status --json
```
