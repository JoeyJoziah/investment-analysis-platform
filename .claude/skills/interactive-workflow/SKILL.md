---
name: interactive-workflow
version: 1.0.0
description: Ultimate Interactive Development Workflow System with 8-phase orchestration, intelligent agent coordination, and quality gates
author: Investment Analysis Platform
category: development-workflow
tags: [workflow, orchestration, agents, quality-gates, automation]
invocable: true
trigger_patterns:
  - "workflow"
  - "feature workflow"
  - "development workflow"
  - "start workflow"
  - "run workflow"
---

# Interactive Workflow Skill

Comprehensive event-driven development workflow system unifying 98 agents, 87 skills, and 108 commands.

## Overview

This skill provides:
- **8-Phase Workflow Execution**: INTAKE → DESIGN → BUILD → REVIEW → INTEGRATE → DEPLOY → LEARN → SYNC
- **Intelligent Agent Orchestration**: Auto-routing to 11 specialized swarms
- **Dynamic Topology Selection**: Star, Mesh, Hierarchical, Parallel, Hive Mind
- **Quality Gates**: Pre-commit, Pre-push, Pre-merge, Pre-deploy
- **Continuous Learning**: Pattern extraction and memory persistence
- **Cross-Session Memory**: Workflow state preservation

## Workflow Types

### Feature Workflow (All 8 Phases)
```bash
/workflow feature "Implement new dashboard"
```

Full development cycle for new features:
1. **INTAKE**: Requirements capture, planning (checkpoint)
2. **DESIGN**: Architecture decisions (conditional checkpoint)
3. **BUILD**: TDD implementation (autonomous)
4. **REVIEW**: Multi-agent code review (checkpoint on issues)
5. **INTEGRATE**: PR creation, CI (checkpoint)
6. **DEPLOY**: Release management (checkpoint)
7. **LEARN**: Pattern extraction (automatic)
8. **SYNC**: Documentation sync (automatic)

### Bugfix Workflow
```bash
/workflow bugfix "Fix null pointer exception"
```

Streamlined: INTAKE → BUILD → REVIEW → INTEGRATE

### Refactor Workflow
```bash
/workflow refactor "Extract common utilities"
```

Code improvement: INTAKE → DESIGN → BUILD → REVIEW

### Hotfix Workflow
```bash
/workflow hotfix "Critical security patch"
```

Emergency: BUILD → REVIEW → DEPLOY (expedited)

### Release Workflow
```bash
/workflow release "v2.1.0"
```

Release: INTEGRATE → DEPLOY → SYNC

## Phase Execution

### Phase 1: INTAKE
**Purpose**: Capture requirements and create implementation plan

**Agents**:
- `planner` - Implementation planning
- `team-coordinator` - Route to specialists
- `specification` - Requirements analysis
- `researcher` - Context gathering

**Topology**: Star (team-coordinator as hub)

**Commands**: `/plan`, `/repo-analyze`, `/analyze-codebase`

**Checkpoint**: MANDATORY - Plan must be approved before proceeding

**Example Workflow**:
```
User Request → planner creates plan
            → team-coordinator identifies specialists
            → /repo-analyze gathers context
            → Generate implementation plan
            → [USER APPROVAL REQUIRED]
```

### Phase 2: DESIGN
**Purpose**: Architecture decisions and topology selection

**Agents**:
- `architect` - System design
- `architecture-reviewer` - Validate decisions
- `system-architect` - Pattern guidance
- `topology-optimizer` - Optimize topology

**Topology**: Hierarchical (architecture-reviewer leads)

**Commands**: `/architect`, `/sparc-architect`, `/auto-topology`

**Checkpoint**: Conditional - on architectural changes

**Outputs**: Architecture Decision Record (ADR)

### Phase 3: BUILD
**Purpose**: Implement code with TDD

**Agents**:
- `tdd-guide` - Test-first methodology
- `coder` - Implementation
- `build-error-resolver` - Fix build errors
- Domain swarms based on file patterns

**Topology**: Mesh (parallel execution)

**Commands**: `/tdd`, `/code`, `/build-fix`, `/parallel-execute`

**Auto-Actions**:
- Build errors → `build-error-resolver` (3 retries)
- File routing to specialized swarms

**File Routing Rules**:
| Pattern | Swarm |
|---------|-------|
| `backend/**/*.py` | backend-api-swarm |
| `frontend/**/*.tsx` | ui-visualization-swarm |
| `backend/ml/**/*.py` | data-ml-pipeline-swarm |
| `**/financial/**` | financial-analysis-swarm |
| `infrastructure/**` | infrastructure-devops-swarm |

### Phase 4: REVIEW
**Purpose**: Multi-agent code review and quality gates

**Agents** (Parallel):
- `security-reviewer` - OWASP, CVE, secrets
- `code-reviewer` - Quality, maintainability
- `code-analyzer` - Complexity, patterns
- `performance-optimizer` - Benchmarks, big-O

**Topology**: Parallel (all reviewers simultaneously)

**Commands**: `/code-review-swarm`, `/verify`, `/security-review`

**Issue Thresholds**:
| Level | Action |
|-------|--------|
| CRITICAL | Block + immediate notification |
| HIGH | Block until addressed |
| MEDIUM | Warn, allow proceed |
| LOW | Suggest, auto-approve |

**Checkpoint**: MANDATORY on CRITICAL/HIGH issues

### Phase 5: INTEGRATE
**Purpose**: PR creation and CI coordination

**Agents**:
- `github-swarm-coordinator` - Lead coordination
- `pr-manager` - PR lifecycle
- `issue-triager` - Issue management
- `sync-coordinator` - Multi-repo sync

**Topology**: Hierarchical

**Commands**: `/github-swarm`, `/pr-manager`, `/pr-enhance`, `/project-board-sync`

**Checkpoint**: MANDATORY before PR creation

### Phase 6: DEPLOY
**Purpose**: Release management and deployment

**Agents**:
- `release-swarm` - Coordinate release
- `release-manager` - Lead deployment
- `infrastructure-devops-swarm` - Infrastructure
- `workflow-automation` - CI/CD automation

**Topology**: Hierarchical

**Commands**: `/release-swarm`, `/release-manager`, `/sparc-devops`

**Checkpoint**: MANDATORY before deploy

### Phase 7: LEARN
**Purpose**: Pattern extraction and knowledge building

**Agents**:
- `sona-learning-optimizer` - Self-optimizing learning
- `v3-memory-specialist` - Memory unification
- `memory-coordinator` - Memory management

**Topology**: Star (memory-coordinator hub)

**Commands**: `/learn`, `/continuous-learning`, `/memory-persist`

**Pattern Types**:
- error_resolution
- user_corrections
- workarounds
- debugging_techniques
- project_specific
- architectural_decisions

**Promotion Criteria**:
- Quality threshold: 0.6
- Minimum usage count: 3

**Trigger**: Automatic on phase completion

### Phase 8: SYNC
**Purpose**: Documentation and external sync

**Agents**:
- `doc-updater` - Documentation updates
- `documentation-agent` - Comprehensive docs

**Topology**: Sequential

**Commands**: `/update-docs`, `/notion`

**Mandatory Actions**:
1. Update TODO.md
2. Update IMPLEMENTATION_STATUS.md
3. Run `./notion-sync.sh push` (CRITICAL)
4. Commit status updates

**Trigger**: Automatic on session end

## Configuration

Configuration files:
- `.claude/config/workflow-engine.json` - Engine configuration
- `.claude/config/topology-rules.json` - Topology selection rules
- `.claude/config/quality-gates.json` - Quality gate thresholds

## State Persistence

Workflow state is persisted to:
- `.claude/memory/workflow-state.json` - Active workflow state
- `.claude/memory/workflow-metrics.json` - Execution metrics
- `.claude/memory/quality-report.md` - Quality gate reports
- `.claude/memory/workflow-history.json` - Workflow history

## Integration with Session Protocol

This skill integrates with the mandatory session completion protocol:
1. LEARN phase extracts patterns automatically
2. SYNC phase ensures Notion sync
3. All state changes are committed to git

## Performance Targets

- Token reduction: 32.3%
- Speedup: 2.8x - 4.4x with parallel execution
- Quality gate compliance: 100%
- Test coverage: ≥80%

## Usage Examples

```bash
# Start feature workflow
/workflow feature "Add user authentication"

# Check status
/workflow status

# Approve checkpoint
/workflow approve

# Skip a phase
/workflow skip design

# Pause/Resume
/workflow pause
/workflow resume

# View phase details
/workflow phase build

# Abort workflow
/workflow abort --reason "Requirements changed"
```

## Related Skills

- `continuous-learning` - Pattern extraction
- `agentdb-memory-patterns` - Memory persistence
- `github-workflow-automation` - CI/CD integration
- `verification-quality` - Quality validation
