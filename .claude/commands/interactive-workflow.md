---
name: workflow
description: Execute the Ultimate Interactive Development Workflow System
version: 1.0.0
invocable: true
aliases: [/iw, /flow]
---

# Interactive Development Workflow System

Execute comprehensive, event-driven development workflows with intelligent agent orchestration.

## Command Syntax

```
/workflow <type> [options]
```

## Workflow Types

| Type | Phases | Description |
|------|--------|-------------|
| `feature` | All 8 phases | Full workflow for new feature implementation |
| `bugfix` | intake → build → review → integrate | Streamlined workflow for bug fixes |
| `refactor` | intake → design → build → review | Workflow for code refactoring |
| `hotfix` | build → review → deploy | Emergency hotfix (expedited) |
| `release` | integrate → deploy → sync | Release management workflow |

## Options

| Option | Description |
|--------|-------------|
| `--skip <phase>` | Skip a specific phase |
| `--parallel` | Enable maximum parallelism |
| `--auto-approve` | Auto-approve non-critical checkpoints |
| `--topology <type>` | Force topology (star/mesh/hierarchical/parallel/hive_mind) |
| `--dry-run` | Preview workflow without execution |

## Workflow Phases

### Phase 1: INTAKE (User Approval Required)
- **Purpose**: Capture requirements, analyze context, create implementation plan
- **Agents**: planner, team-coordinator, specification, researcher
- **Topology**: Star (team-coordinator as hub)
- **Checkpoint**: MANDATORY - Plan must be approved

### Phase 2: DESIGN (Conditional Checkpoint)
- **Purpose**: Architecture decisions, topology selection
- **Agents**: architect, architecture-reviewer, system-architect
- **Topology**: Hierarchical (architecture-reviewer leads)
- **Checkpoint**: Required on architectural changes

### Phase 3: BUILD (Autonomous with TDD)
- **Purpose**: Implement code with test-driven development
- **Agents**: tdd-guide, coder, build-error-resolver, specialized swarms
- **Topology**: Mesh (parallel execution)
- **Auto-Actions**: Build errors trigger build-error-resolver

### Phase 4: REVIEW (Auto + User Review)
- **Purpose**: Multi-agent code review, security scan
- **Agents**: code-review-swarm, security-reviewer, code-analyzer
- **Topology**: Parallel (all reviewers simultaneously)
- **Checkpoint**: MANDATORY on CRITICAL/HIGH issues

### Phase 5: INTEGRATE (User Approval Required)
- **Purpose**: Create PR, run CI, coordinate merge
- **Agents**: github-swarm-coordinator, pr-manager
- **Topology**: Hierarchical
- **Checkpoint**: MANDATORY before PR creation

### Phase 6: DEPLOY (User Approval Required)
- **Purpose**: Release management, deployment
- **Agents**: release-swarm, infrastructure-devops-swarm
- **Topology**: Hierarchical
- **Checkpoint**: MANDATORY before deploy

### Phase 7: LEARN (Automatic)
- **Purpose**: Extract patterns, build knowledge
- **Agents**: sona-learning-optimizer, memory-coordinator
- **Topology**: Star
- **Trigger**: Automatic on phase completion

### Phase 8: SYNC (Automatic)
- **Purpose**: Documentation, external sync
- **Agents**: doc-updater, documentation-agent
- **Mandatory**: ./notion-sync.sh push

## Execution Instructions

When this command is invoked:

1. **Initialize Workflow**
   ```javascript
   // Load workflow configuration
   const config = require('.claude/config/workflow-engine.json');
   const topologyRules = require('.claude/config/topology-rules.json');
   const qualityGates = require('.claude/config/quality-gates.json');
   ```

2. **Determine Workflow Type**
   - Parse the workflow type argument
   - Load corresponding template from .claude/workflows/

3. **Execute Phases Sequentially**
   For each phase in the workflow:

   a. **Pre-Phase**
      - Check phase prerequisites
      - Select optimal topology using topology-rules.json
      - Spawn required agents

   b. **Phase Execution**
      - Route to specialized swarms based on file patterns
      - Execute phase commands
      - Collect outputs

   c. **Post-Phase**
      - Run quality gates if applicable
      - Check checkpoint requirements
      - Persist state to memory
      - WAIT for user approval if checkpoint required

4. **Handle Checkpoints**
   - Phases with `checkpoint_required: true` MUST pause for user approval
   - Display checkpoint status and required actions
   - Resume only after explicit user approval

5. **Error Handling**
   - On error, activate relevant resolver agent
   - Max 3 retries with exponential backoff
   - Fallback to team-coordinator for routing

6. **Completion**
   - Execute LEARN phase automatically
   - Execute SYNC phase automatically
   - Display workflow summary

## Agent Routing Rules

| File Pattern | Swarm |
|--------------|-------|
| `backend/**/*.py` | backend-api-swarm |
| `frontend/**/*.tsx` | ui-visualization-swarm |
| `backend/ml/**/*.py` | data-ml-pipeline-swarm |
| `**/financial/**` | financial-analysis-swarm |
| `infrastructure/**` | infrastructure-devops-swarm |
| `data_pipelines/**` | data-ml-pipeline-swarm |

## Quality Gate Integration

Each phase integrates with quality gates:

- **pre_commit**: Lint, format, type check, secrets check
- **pre_push**: Unit tests, coverage (80%+), security scan
- **pre_merge**: Code review, security review, integration tests
- **pre_deploy**: E2E tests, performance checks, staging validation

## Examples

```bash
# Full feature workflow
/workflow feature "Implement user authentication"

# Bug fix workflow
/workflow bugfix "Fix null pointer in payment processing"

# Refactoring workflow
/workflow refactor "Extract common utilities to shared module"

# Emergency hotfix
/workflow hotfix "Critical security patch for SQL injection"

# Release workflow
/workflow release "v2.1.0"

# With options
/workflow feature "Add dashboard charts" --parallel --topology mesh

# Dry run preview
/workflow feature "New API endpoint" --dry-run
```

## State Management

Workflow state is persisted to:
- `.claude/memory/workflow-state.json` - Active workflow state
- `.claude/memory/workflow-metrics.json` - Execution metrics
- `.claude/memory/quality-report.md` - Quality gate reports

## Integration with Session Completion Protocol

At workflow completion, the SYNC phase automatically:
1. Updates TODO.md with completed items
2. Updates IMPLEMENTATION_STATUS.md
3. Runs `./notion-sync.sh push` (MANDATORY)
4. Commits status updates to git
