# Agent Directory Reorganization Plan

**Status**: Draft
**Date**: 2026-01-27
**Impact**: High - 68% directory reduction (43 dirs → 7 dirs)
**Total Agents**: 232 agent files

## Executive Summary

Current agent organization across 43 subdirectories creates significant complexity. This plan consolidates into 7 logical categories, reducing directories by 68% while improving discoverability and maintainability.

## Current State Analysis

### Directory Statistics
- **Total Directories**: 43 subdirectories
- **Total Agent Files**: 232 .md files
- **Root-level Agents**: 121 files (52% of all agents)
- **Deeply Nested Agents**: 111 files across subdirectories
- **Empty/Low-Use Dirs**: ~29 directories with <5 agents (68%)

### Problems with Current Structure
1. **Fragmentation**: 68% of directories contain fewer than 5 agents
2. **Duplication**: Multiple paths to similar agents (e.g., `/core/coder.md` and `/coder.md`)
3. **Inconsistency**: No clear organization pattern
4. **Discovery Issues**: Hard to find relevant agents
5. **Maintenance Burden**: 43 directories to maintain

## Proposed New Structure

### 7 Logical Categories

```
.claude/agents/
├── 1-core/                          # 5 fundamental agents
│   ├── coder.md
│   ├── reviewer.md
│   ├── tester.md
│   ├── planner.md
│   └── researcher.md
│
├── 2-swarm-coordination/            # 25 coordination agents
│   ├── hierarchical-coordinator.md
│   ├── mesh-coordinator.md
│   ├── adaptive-coordinator.md
│   ├── collective-intelligence-coordinator.md
│   ├── queen-coordinator.md
│   ├── swarm-memory-manager.md
│   ├── byzantine-coordinator.md
│   ├── raft-manager.md
│   ├── gossip-coordinator.md
│   ├── crdt-synchronizer.md
│   ├── quorum-manager.md
│   ├── consensus-coordinator.md
│   ├── security-manager.md
│   ├── performance-benchmarker.md
│   ├── load-balancer.md
│   ├── resource-allocator.md
│   ├── topology-optimizer.md
│   ├── performance-monitor.md
│   ├── benchmark-suite.md
│   ├── worker-specialist.md
│   ├── safla-neural.md
│   ├── memory-coordinator.md
│   ├── task-orchestrator.md
│   ├── scout-explorer.md
│   └── team-coordinator.md
│
├── 3-security-performance/          # 15 security & performance agents
│   ├── security-reviewer.md
│   ├── security-architect.md
│   ├── security-auditor.md
│   ├── security-compliance-swarm.md
│   ├── performance-engineer.md
│   ├── performance-analyzer.md
│   ├── performance-optimizer.md
│   ├── matrix-optimizer.md
│   ├── pagerank-analyzer.md
│   ├── sona-learning-optimizer.md
│   ├── neural-network.md
│   ├── risk-assessor.md
│   ├── architecture-reviewer.md
│   ├── code-analyzer.md
│   └── analyze-code-quality.md
│
├── 4-github-repository/             # 20 GitHub & repo agents
│   ├── github-modes.md
│   ├── pr-manager.md
│   ├── issue-tracker.md
│   ├── release-manager.md
│   ├── release-swarm.md
│   ├── sync-coordinator.md
│   ├── workflow-automation.md
│   ├── project-board-sync.md
│   ├── code-review-swarm.md
│   ├── repo-architect.md
│   ├── multi-repo-swarm.md
│   ├── swarm-pr.md
│   ├── swarm-issue.md
│   ├── code-reviewer.md
│   ├── code-review-expert.md
│   ├── ops-cicd-github.md
│   ├── build-error-resolver.md
│   ├── infrastructure-devops-swarm.md
│   ├── infrastructure-agent.md
│   └── v3-integration-architect.md
│
├── 5-sparc-methodology/             # 10 SPARC agents
│   ├── sparc-coordinator.md
│   ├── specification.md
│   ├── pseudocode.md
│   ├── architecture.md
│   ├── refinement.md
│   ├── implementer-sparc-coder.md
│   ├── architect.md
│   ├── tdd-guide.md
│   ├── goal-planner.md
│   └── code-goal-planner.md
│
├── 6-specialized-development/       # 35 specialized dev agents
│   ├── backend-dev.md
│   ├── backend-api-swarm.md
│   ├── dev-backend-api.md
│   ├── mobile-dev.md
│   ├── spec-mobile-react-native.md
│   ├── ml-developer.md
│   ├── data-ml-model.md
│   ├── data-ml-pipeline-swarm.md
│   ├── data-science-architect.md
│   ├── cicd-engineer.md
│   ├── api-docs.md
│   ├── docs-api-openapi.md
│   ├── doc-updater.md
│   ├── system-architect.md
│   ├── arch-system-design.md
│   ├── base-template-generator.md
│   ├── financial-analysis-swarm.md
│   ├── financial-modeler.md
│   ├── investment-analyst.md
│   ├── deal-underwriter.md
│   ├── queen-investment-orchestrator.md
│   ├── trading-predictor.md
│   ├── ui-visualization-swarm.md
│   ├── ui_design.md
│   ├── authentication.md
│   ├── agentic-payments.md
│   ├── payments.md
│   ├── app-store.md
│   ├── sandbox.md
│   ├── user-tools.md
│   ├── workflow.md
│   ├── agent.md
│   ├── godmode-refactorer.md
│   ├── refactor-cleaner.md
│   └── migration-plan.md
│
└── 7-testing-validation/            # 10 testing agents
    ├── tdd-london-swarm.md
    ├── production-validator.md
    ├── e2e-runner.md
    ├── test-agent.md
    ├── test-long-runner.md
    ├── project-quality-swarm.md
    ├── challenges.md
    ├── swarm.md
    └── README.md (updated)
```

## Category Definitions

### 1. Core (5 agents)
**Purpose**: Fundamental agents used in every project
**Agents**: coder, reviewer, tester, planner, researcher
**Priority**: Critical - these are the most frequently used agents

### 2. Swarm Coordination (25 agents)
**Purpose**: Multi-agent orchestration, consensus, and distributed coordination
**Key Areas**:
- Hierarchical coordination (queen-led)
- Mesh/adaptive topologies
- Consensus protocols (Byzantine, Raft, Gossip)
- Resource management and optimization
- Memory and task coordination

### 3. Security & Performance (15 agents)
**Purpose**: Security analysis, compliance, and performance optimization
**Key Areas**:
- Security review and auditing
- Performance profiling and optimization
- Neural network optimization
- Risk assessment
- Code quality analysis

### 4. GitHub & Repository (20 agents)
**Purpose**: GitHub workflows, PR management, CI/CD, repository operations
**Key Areas**:
- PR and issue management
- Release coordination
- Code review automation
- CI/CD orchestration
- Multi-repository synchronization

### 5. SPARC Methodology (10 agents)
**Purpose**: Structured development methodology (Specification, Pseudocode, Architecture, Refinement, Coding)
**Key Areas**:
- Requirements specification
- Architecture design
- Implementation planning
- TDD guidance
- Goal planning

### 6. Specialized Development (35 agents)
**Purpose**: Domain-specific and specialized development tasks
**Key Areas**:
- Backend/Mobile/ML development
- Financial analysis and modeling
- UI/UX design
- Authentication and payments
- Domain-specific tools

### 7. Testing & Validation (10 agents)
**Purpose**: Comprehensive testing, validation, and quality assurance
**Key Areas**:
- TDD swarms
- E2E testing
- Production validation
- Quality assurance
- Long-running tests

## Migration Mapping

### From Root Level (121 files)
```
Root files → New locations:
├── coder.md                                  → 1-core/coder.md
├── reviewer.md                               → 1-core/reviewer.md
├── tester.md                                 → 1-core/tester.md
├── planner.md                                → 1-core/planner.md
├── researcher.md                             → 1-core/researcher.md
├── hierarchical-coordinator.md               → 2-swarm-coordination/hierarchical-coordinator.md
├── mesh-coordinator.md                       → 2-swarm-coordination/mesh-coordinator.md
├── adaptive-coordinator.md                   → 2-swarm-coordination/adaptive-coordinator.md
├── collective-intelligence-coordinator.md    → 2-swarm-coordination/collective-intelligence-coordinator.md
├── queen-coordinator.md                      → 2-swarm-coordination/queen-coordinator.md
├── swarm-memory-manager.md                   → 2-swarm-coordination/swarm-memory-manager.md
├── byzantine-coordinator.md                  → 2-swarm-coordination/byzantine-coordinator.md
├── raft-manager.md                           → 2-swarm-coordination/raft-manager.md
├── gossip-coordinator.md                     → 2-swarm-coordination/gossip-coordinator.md
├── crdt-synchronizer.md                      → 2-swarm-coordination/crdt-synchronizer.md
├── quorum-manager.md                         → 2-swarm-coordination/quorum-manager.md
├── consensus-coordinator.md                  → 2-swarm-coordination/consensus-coordinator.md
├── security-manager.md                       → 2-swarm-coordination/security-manager.md
├── performance-benchmarker.md                → 2-swarm-coordination/performance-benchmarker.md
├── load-balancer.md                          → 2-swarm-coordination/load-balancer.md
├── resource-allocator.md                     → 2-swarm-coordination/resource-allocator.md
├── topology-optimizer.md                     → 2-swarm-coordination/topology-optimizer.md
├── performance-monitor.md                    → 2-swarm-coordination/performance-monitor.md
├── benchmark-suite.md                        → 2-swarm-coordination/benchmark-suite.md
├── worker-specialist.md                      → 2-swarm-coordination/worker-specialist.md
├── safla-neural.md                           → 2-swarm-coordination/safla-neural.md
├── memory-coordinator.md                     → 2-swarm-coordination/memory-coordinator.md
├── scout-explorer.md                         → 2-swarm-coordination/scout-explorer.md
├── team-coordinator.md                       → 2-swarm-coordination/team-coordinator.md
├── security-reviewer.md                      → 3-security-performance/security-reviewer.md
├── performance-analyzer.md                   → 3-security-performance/performance-analyzer.md
├── performance-optimizer.md                  → 3-security-performance/performance-optimizer.md
├── matrix-optimizer.md                       → 3-security-performance/matrix-optimizer.md
├── pagerank-analyzer.md                      → 3-security-performance/pagerank-analyzer.md
├── sona-learning-optimizer.md                → 3-security-performance/sona-learning-optimizer.md
├── neural-network.md                         → 3-security-performance/neural-network.md
├── risk-assessor.md                          → 3-security-performance/risk-assessor.md
├── architecture-reviewer.md                  → 3-security-performance/architecture-reviewer.md
├── code-analyzer.md                          → 3-security-performance/code-analyzer.md
├── analyze-code-quality.md                   → 3-security-performance/analyze-code-quality.md
├── security-compliance-swarm.md              → 3-security-performance/security-compliance-swarm.md
├── github-modes.md                           → 4-github-repository/github-modes.md
├── pr-manager.md                             → 4-github-repository/pr-manager.md
├── issue-tracker.md                          → 4-github-repository/issue-tracker.md
├── release-manager.md                        → 4-github-repository/release-manager.md
├── release-swarm.md                          → 4-github-repository/release-swarm.md
├── sync-coordinator.md                       → 4-github-repository/sync-coordinator.md
├── workflow-automation.md                    → 4-github-repository/workflow-automation.md
├── project-board-sync.md                     → 4-github-repository/project-board-sync.md
├── code-review-swarm.md                      → 4-github-repository/code-review-swarm.md
├── repo-architect.md                         → 4-github-repository/repo-architect.md
├── multi-repo-swarm.md                       → 4-github-repository/multi-repo-swarm.md
├── swarm-pr.md                               → 4-github-repository/swarm-pr.md
├── swarm-issue.md                            → 4-github-repository/swarm-issue.md
├── code-reviewer.md                          → 4-github-repository/code-reviewer.md
├── code-review-expert.md                     → 4-github-repository/code-review-expert.md
├── build-error-resolver.md                   → 4-github-repository/build-error-resolver.md
├── infrastructure-devops-swarm.md            → 4-github-repository/infrastructure-devops-swarm.md
├── infrastructure-agent.md                   → 4-github-repository/infrastructure-agent.md
├── v3-integration-architect.md               → 4-github-repository/v3-integration-architect.md
├── specification.md                          → 5-sparc-methodology/specification.md
├── pseudocode.md                             → 5-sparc-methodology/pseudocode.md
├── architecture.md                           → 5-sparc-methodology/architecture.md
├── refinement.md                             → 5-sparc-methodology/refinement.md
├── implementer-sparc-coder.md                → 5-sparc-methodology/implementer-sparc-coder.md
├── architect.md                              → 5-sparc-methodology/architect.md
├── tdd-guide.md                              → 5-sparc-methodology/tdd-guide.md
├── goal-planner.md                           → 5-sparc-methodology/goal-planner.md
├── code-goal-planner.md                      → 5-sparc-methodology/code-goal-planner.md
├── backend-api-swarm.md                      → 6-specialized-development/backend-api-swarm.md
├── dev-backend-api.md                        → 6-specialized-development/dev-backend-api.md
├── data-ml-pipeline-swarm.md                 → 6-specialized-development/data-ml-pipeline-swarm.md
├── data-science-architect.md                 → 6-specialized-development/data-science-architect.md
├── doc-updater.md                            → 6-specialized-development/doc-updater.md
├── financial-analysis-swarm.md               → 6-specialized-development/financial-analysis-swarm.md
├── financial-modeler.md                      → 6-specialized-development/financial-modeler.md
├── investment-analyst.md                     → 6-specialized-development/investment-analyst.md
├── deal-underwriter.md                       → 6-specialized-development/deal-underwriter.md
├── queen-investment-orchestrator.md          → 6-specialized-development/queen-investment-orchestrator.md
├── trading-predictor.md                      → 6-specialized-development/trading-predictor.md
├── ui-visualization-swarm.md                 → 6-specialized-development/ui-visualization-swarm.md
├── ui_design.md                              → 6-specialized-development/ui_design.md
├── authentication.md                         → 6-specialized-development/authentication.md
├── agentic-payments.md                       → 6-specialized-development/agentic-payments.md
├── payments.md                               → 6-specialized-development/payments.md
├── app-store.md                              → 6-specialized-development/app-store.md
├── sandbox.md                                → 6-specialized-development/sandbox.md
├── user-tools.md                             → 6-specialized-development/user-tools.md
├── workflow.md                               → 6-specialized-development/workflow.md
├── agent.md                                  → 6-specialized-development/agent.md
├── godmode-refactorer.md                     → 6-specialized-development/godmode-refactorer.md
├── refactor-cleaner.md                       → 6-specialized-development/refactor-cleaner.md
├── migration-plan.md                         → 6-specialized-development/migration-plan.md
├── tdd-london-swarm.md                       → 7-testing-validation/tdd-london-swarm.md
├── e2e-runner.md                             → 7-testing-validation/e2e-runner.md
├── test-agent.md                             → 7-testing-validation/test-agent.md
├── test-long-runner.md                       → 7-testing-validation/test-long-runner.md
├── project-quality-swarm.md                  → 7-testing-validation/project-quality-swarm.md
├── challenges.md                             → 7-testing-validation/challenges.md
└── swarm.md                                  → 7-testing-validation/swarm.md
```

### From Subdirectories (111 files)
```
Subdirectories → New locations:
├── core/*                          → 1-core/
├── swarm/*                         → 2-swarm-coordination/
├── consensus/*                     → 2-swarm-coordination/
├── optimization/*                  → 2-swarm-coordination/
├── hive-mind/*                     → 2-swarm-coordination/
├── sublinear/*                     → 3-security-performance/
├── sona/*                          → 3-security-performance/
├── analysis/*                      → 3-security-performance/
├── github/*                        → 4-github-repository/
├── github-swarm/*                  → 4-github-repository/
├── devops/*                        → 4-github-repository/
├── ci-cd/*                         → 4-github-repository/
├── sparc/*                         → 5-sparc-methodology/
├── goal/*                          → 5-sparc-methodology/
├── development/*                   → 6-specialized-development/
├── backend/*                       → 6-specialized-development/
├── mobile/*                        → 6-specialized-development/
├── specialized/*                   → 6-specialized-development/
├── ml/*                            → 6-specialized-development/
├── data/*                          → 6-specialized-development/
├── architecture/*                  → 6-specialized-development/
├── system-design/*                 → 6-specialized-development/
├── api-docs/*                      → 6-specialized-development/
├── documentation/*                 → 6-specialized-development/
├── payments/*                      → 6-specialized-development/
├── flow-nexus/*                    → 6-specialized-development/
├── v3/*                            → 6-specialized-development/
├── testing/*                       → 7-testing-validation/
├── unit/*                          → 7-testing-validation/
└── validation/*                    → 7-testing-validation/
```

## Implementation Strategy

### Phase 1: Preparation (Week 1)
1. **Backup current structure**
   ```bash
   cp -r .claude/agents .claude/agents.backup-$(date +%Y%m%d)
   ```

2. **Create new directory structure**
   ```bash
   mkdir -p .claude/agents/{1-core,2-swarm-coordination,3-security-performance,4-github-repository,5-sparc-methodology,6-specialized-development,7-testing-validation}
   ```

3. **Validate all agent files**
   - Check YAML frontmatter
   - Verify no malformed files
   - Document any issues

### Phase 2: Migration (Week 2)
1. **Copy files to new locations** (preserve originals initially)
2. **Update internal references** within agent files
3. **Test agent loading** in new structure
4. **Update documentation** and README files

### Phase 3: Validation (Week 3)
1. **Comprehensive testing** of all agent categories
2. **Verify backward compatibility** (if needed)
3. **Performance benchmarking**
4. **User acceptance testing**

### Phase 4: Cleanup (Week 4)
1. **Remove old directories** after validation
2. **Update all references** in codebase
3. **Archive backup** for rollback capability
4. **Documentation finalization**

## Migration Script

```bash
#!/bin/bash
# agent-reorganization.sh - Reorganizes agent directory structure

set -e

AGENTS_DIR=".claude/agents"
BACKUP_DIR=".claude/agents.backup-$(date +%Y%m%d-%H%M%S)"
DRY_RUN=${1:-"--dry-run"}

echo "🔄 Agent Directory Reorganization Script"
echo "========================================"

# Backup current structure
echo "📦 Creating backup: $BACKUP_DIR"
if [ "$DRY_RUN" != "--dry-run" ]; then
    cp -r "$AGENTS_DIR" "$BACKUP_DIR"
fi

# Create new directory structure
echo "📁 Creating new directory structure..."
NEW_DIRS=(
    "1-core"
    "2-swarm-coordination"
    "3-security-performance"
    "4-github-repository"
    "5-sparc-methodology"
    "6-specialized-development"
    "7-testing-validation"
)

for dir in "${NEW_DIRS[@]}"; do
    echo "  Creating $AGENTS_DIR/$dir/"
    if [ "$DRY_RUN" != "--dry-run" ]; then
        mkdir -p "$AGENTS_DIR/$dir"
    fi
done

# Migration mapping function
migrate_agent() {
    local source=$1
    local dest=$2

    if [ -f "$source" ]; then
        echo "  Moving: $source → $dest"
        if [ "$DRY_RUN" != "--dry-run" ]; then
            cp "$source" "$dest"
        fi
    else
        echo "  ⚠️  Missing: $source"
    fi
}

echo ""
echo "🚀 Migrating agents to new structure..."

# 1-core migrations
echo "📂 Category 1: Core Agents"
migrate_agent "$AGENTS_DIR/coder.md" "$AGENTS_DIR/1-core/coder.md"
migrate_agent "$AGENTS_DIR/core/coder.md" "$AGENTS_DIR/1-core/coder.md"
migrate_agent "$AGENTS_DIR/reviewer.md" "$AGENTS_DIR/1-core/reviewer.md"
migrate_agent "$AGENTS_DIR/core/reviewer.md" "$AGENTS_DIR/1-core/reviewer.md"
migrate_agent "$AGENTS_DIR/tester.md" "$AGENTS_DIR/1-core/tester.md"
migrate_agent "$AGENTS_DIR/core/tester.md" "$AGENTS_DIR/1-core/tester.md"
migrate_agent "$AGENTS_DIR/planner.md" "$AGENTS_DIR/1-core/planner.md"
migrate_agent "$AGENTS_DIR/core/planner.md" "$AGENTS_DIR/1-core/planner.md"
migrate_agent "$AGENTS_DIR/researcher.md" "$AGENTS_DIR/1-core/researcher.md"
migrate_agent "$AGENTS_DIR/core/researcher.md" "$AGENTS_DIR/1-core/researcher.md"

# 2-swarm-coordination migrations
echo "📂 Category 2: Swarm Coordination"
migrate_agent "$AGENTS_DIR/hierarchical-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/hierarchical-coordinator.md"
migrate_agent "$AGENTS_DIR/swarm/hierarchical-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/hierarchical-coordinator.md"
migrate_agent "$AGENTS_DIR/mesh-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/mesh-coordinator.md"
migrate_agent "$AGENTS_DIR/swarm/mesh-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/mesh-coordinator.md"
migrate_agent "$AGENTS_DIR/adaptive-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/adaptive-coordinator.md"
migrate_agent "$AGENTS_DIR/swarm/adaptive-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/adaptive-coordinator.md"
migrate_agent "$AGENTS_DIR/collective-intelligence-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/collective-intelligence-coordinator.md"
migrate_agent "$AGENTS_DIR/queen-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/queen-coordinator.md"
migrate_agent "$AGENTS_DIR/swarm-memory-manager.md" "$AGENTS_DIR/2-swarm-coordination/swarm-memory-manager.md"
migrate_agent "$AGENTS_DIR/byzantine-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/byzantine-coordinator.md"
migrate_agent "$AGENTS_DIR/consensus/byzantine-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/byzantine-coordinator.md"
migrate_agent "$AGENTS_DIR/raft-manager.md" "$AGENTS_DIR/2-swarm-coordination/raft-manager.md"
migrate_agent "$AGENTS_DIR/consensus/raft-manager.md" "$AGENTS_DIR/2-swarm-coordination/raft-manager.md"
migrate_agent "$AGENTS_DIR/gossip-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/gossip-coordinator.md"
migrate_agent "$AGENTS_DIR/consensus/gossip-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/gossip-coordinator.md"
migrate_agent "$AGENTS_DIR/crdt-synchronizer.md" "$AGENTS_DIR/2-swarm-coordination/crdt-synchronizer.md"
migrate_agent "$AGENTS_DIR/consensus/crdt-synchronizer.md" "$AGENTS_DIR/2-swarm-coordination/crdt-synchronizer.md"
migrate_agent "$AGENTS_DIR/quorum-manager.md" "$AGENTS_DIR/2-swarm-coordination/quorum-manager.md"
migrate_agent "$AGENTS_DIR/consensus/quorum-manager.md" "$AGENTS_DIR/2-swarm-coordination/quorum-manager.md"
migrate_agent "$AGENTS_DIR/consensus-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/consensus-coordinator.md"
migrate_agent "$AGENTS_DIR/security-manager.md" "$AGENTS_DIR/2-swarm-coordination/security-manager.md"
migrate_agent "$AGENTS_DIR/consensus/security-manager.md" "$AGENTS_DIR/2-swarm-coordination/security-manager.md"
migrate_agent "$AGENTS_DIR/performance-benchmarker.md" "$AGENTS_DIR/2-swarm-coordination/performance-benchmarker.md"
migrate_agent "$AGENTS_DIR/consensus/performance-benchmarker.md" "$AGENTS_DIR/2-swarm-coordination/performance-benchmarker.md"
migrate_agent "$AGENTS_DIR/load-balancer.md" "$AGENTS_DIR/2-swarm-coordination/load-balancer.md"
migrate_agent "$AGENTS_DIR/optimization/load-balancer.md" "$AGENTS_DIR/2-swarm-coordination/load-balancer.md"
migrate_agent "$AGENTS_DIR/resource-allocator.md" "$AGENTS_DIR/2-swarm-coordination/resource-allocator.md"
migrate_agent "$AGENTS_DIR/optimization/resource-allocator.md" "$AGENTS_DIR/2-swarm-coordination/resource-allocator.md"
migrate_agent "$AGENTS_DIR/topology-optimizer.md" "$AGENTS_DIR/2-swarm-coordination/topology-optimizer.md"
migrate_agent "$AGENTS_DIR/optimization/topology-optimizer.md" "$AGENTS_DIR/2-swarm-coordination/topology-optimizer.md"
migrate_agent "$AGENTS_DIR/performance-monitor.md" "$AGENTS_DIR/2-swarm-coordination/performance-monitor.md"
migrate_agent "$AGENTS_DIR/optimization/performance-monitor.md" "$AGENTS_DIR/2-swarm-coordination/performance-monitor.md"
migrate_agent "$AGENTS_DIR/benchmark-suite.md" "$AGENTS_DIR/2-swarm-coordination/benchmark-suite.md"
migrate_agent "$AGENTS_DIR/optimization/benchmark-suite.md" "$AGENTS_DIR/2-swarm-coordination/benchmark-suite.md"
migrate_agent "$AGENTS_DIR/worker-specialist.md" "$AGENTS_DIR/2-swarm-coordination/worker-specialist.md"
migrate_agent "$AGENTS_DIR/safla-neural.md" "$AGENTS_DIR/2-swarm-coordination/safla-neural.md"
migrate_agent "$AGENTS_DIR/memory-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/memory-coordinator.md"
migrate_agent "$AGENTS_DIR/scout-explorer.md" "$AGENTS_DIR/2-swarm-coordination/scout-explorer.md"
migrate_agent "$AGENTS_DIR/team-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/team-coordinator.md"

# 3-security-performance migrations
echo "📂 Category 3: Security & Performance"
migrate_agent "$AGENTS_DIR/security-reviewer.md" "$AGENTS_DIR/3-security-performance/security-reviewer.md"
migrate_agent "$AGENTS_DIR/security-compliance-swarm.md" "$AGENTS_DIR/3-security-performance/security-compliance-swarm.md"
migrate_agent "$AGENTS_DIR/performance-analyzer.md" "$AGENTS_DIR/3-security-performance/performance-analyzer.md"
migrate_agent "$AGENTS_DIR/performance-optimizer.md" "$AGENTS_DIR/3-security-performance/performance-optimizer.md"
migrate_agent "$AGENTS_DIR/sublinear/performance-optimizer.md" "$AGENTS_DIR/3-security-performance/performance-optimizer.md"
migrate_agent "$AGENTS_DIR/matrix-optimizer.md" "$AGENTS_DIR/3-security-performance/matrix-optimizer.md"
migrate_agent "$AGENTS_DIR/sublinear/matrix-optimizer.md" "$AGENTS_DIR/3-security-performance/matrix-optimizer.md"
migrate_agent "$AGENTS_DIR/pagerank-analyzer.md" "$AGENTS_DIR/3-security-performance/pagerank-analyzer.md"
migrate_agent "$AGENTS_DIR/sublinear/pagerank-analyzer.md" "$AGENTS_DIR/3-security-performance/pagerank-analyzer.md"
migrate_agent "$AGENTS_DIR/sona-learning-optimizer.md" "$AGENTS_DIR/3-security-performance/sona-learning-optimizer.md"
migrate_agent "$AGENTS_DIR/sona/sona-learning-optimizer.md" "$AGENTS_DIR/3-security-performance/sona-learning-optimizer.md"
migrate_agent "$AGENTS_DIR/neural-network.md" "$AGENTS_DIR/3-security-performance/neural-network.md"
migrate_agent "$AGENTS_DIR/risk-assessor.md" "$AGENTS_DIR/3-security-performance/risk-assessor.md"
migrate_agent "$AGENTS_DIR/architecture-reviewer.md" "$AGENTS_DIR/3-security-performance/architecture-reviewer.md"
migrate_agent "$AGENTS_DIR/code-analyzer.md" "$AGENTS_DIR/3-security-performance/code-analyzer.md"
migrate_agent "$AGENTS_DIR/analysis/code-analyzer.md" "$AGENTS_DIR/3-security-performance/code-analyzer.md"
migrate_agent "$AGENTS_DIR/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md"
migrate_agent "$AGENTS_DIR/analysis/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md"
migrate_agent "$AGENTS_DIR/analysis/code-review/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md"
migrate_agent "$AGENTS_DIR/code-review/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md"

# 4-github-repository migrations
echo "📂 Category 4: GitHub & Repository"
migrate_agent "$AGENTS_DIR/github-modes.md" "$AGENTS_DIR/4-github-repository/github-modes.md"
migrate_agent "$AGENTS_DIR/pr-manager.md" "$AGENTS_DIR/4-github-repository/pr-manager.md"
migrate_agent "$AGENTS_DIR/issue-tracker.md" "$AGENTS_DIR/4-github-repository/issue-tracker.md"
migrate_agent "$AGENTS_DIR/release-manager.md" "$AGENTS_DIR/4-github-repository/release-manager.md"
migrate_agent "$AGENTS_DIR/release-swarm.md" "$AGENTS_DIR/4-github-repository/release-swarm.md"
migrate_agent "$AGENTS_DIR/sync-coordinator.md" "$AGENTS_DIR/4-github-repository/sync-coordinator.md"
migrate_agent "$AGENTS_DIR/workflow-automation.md" "$AGENTS_DIR/4-github-repository/workflow-automation.md"
migrate_agent "$AGENTS_DIR/project-board-sync.md" "$AGENTS_DIR/4-github-repository/project-board-sync.md"
migrate_agent "$AGENTS_DIR/code-review-swarm.md" "$AGENTS_DIR/4-github-repository/code-review-swarm.md"
migrate_agent "$AGENTS_DIR/repo-architect.md" "$AGENTS_DIR/4-github-repository/repo-architect.md"
migrate_agent "$AGENTS_DIR/multi-repo-swarm.md" "$AGENTS_DIR/4-github-repository/multi-repo-swarm.md"
migrate_agent "$AGENTS_DIR/swarm-pr.md" "$AGENTS_DIR/4-github-repository/swarm-pr.md"
migrate_agent "$AGENTS_DIR/swarm-issue.md" "$AGENTS_DIR/4-github-repository/swarm-issue.md"
migrate_agent "$AGENTS_DIR/code-reviewer.md" "$AGENTS_DIR/4-github-repository/code-reviewer.md"
migrate_agent "$AGENTS_DIR/code-review-expert.md" "$AGENTS_DIR/4-github-repository/code-review-expert.md"
migrate_agent "$AGENTS_DIR/build-error-resolver.md" "$AGENTS_DIR/4-github-repository/build-error-resolver.md"
migrate_agent "$AGENTS_DIR/infrastructure-devops-swarm.md" "$AGENTS_DIR/4-github-repository/infrastructure-devops-swarm.md"
migrate_agent "$AGENTS_DIR/infrastructure-agent.md" "$AGENTS_DIR/4-github-repository/infrastructure-agent.md"
migrate_agent "$AGENTS_DIR/v3-integration-architect.md" "$AGENTS_DIR/4-github-repository/v3-integration-architect.md"
migrate_agent "$AGENTS_DIR/devops/ci-cd/ops-cicd-github.md" "$AGENTS_DIR/4-github-repository/ops-cicd-github.md"

# 5-sparc-methodology migrations
echo "📂 Category 5: SPARC Methodology"
migrate_agent "$AGENTS_DIR/specification.md" "$AGENTS_DIR/5-sparc-methodology/specification.md"
migrate_agent "$AGENTS_DIR/pseudocode.md" "$AGENTS_DIR/5-sparc-methodology/pseudocode.md"
migrate_agent "$AGENTS_DIR/architecture.md" "$AGENTS_DIR/5-sparc-methodology/architecture.md"
migrate_agent "$AGENTS_DIR/refinement.md" "$AGENTS_DIR/5-sparc-methodology/refinement.md"
migrate_agent "$AGENTS_DIR/implementer-sparc-coder.md" "$AGENTS_DIR/5-sparc-methodology/implementer-sparc-coder.md"
migrate_agent "$AGENTS_DIR/architect.md" "$AGENTS_DIR/5-sparc-methodology/architect.md"
migrate_agent "$AGENTS_DIR/tdd-guide.md" "$AGENTS_DIR/5-sparc-methodology/tdd-guide.md"
migrate_agent "$AGENTS_DIR/goal-planner.md" "$AGENTS_DIR/5-sparc-methodology/goal-planner.md"
migrate_agent "$AGENTS_DIR/code-goal-planner.md" "$AGENTS_DIR/5-sparc-methodology/code-goal-planner.md"

# 6-specialized-development migrations
echo "📂 Category 6: Specialized Development"
migrate_agent "$AGENTS_DIR/backend-api-swarm.md" "$AGENTS_DIR/6-specialized-development/backend-api-swarm.md"
migrate_agent "$AGENTS_DIR/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md"
migrate_agent "$AGENTS_DIR/backend/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md"
migrate_agent "$AGENTS_DIR/development/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md"
migrate_agent "$AGENTS_DIR/development/backend/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md"
migrate_agent "$AGENTS_DIR/mobile/spec-mobile-react-native.md" "$AGENTS_DIR/6-specialized-development/spec-mobile-react-native.md"
migrate_agent "$AGENTS_DIR/specialized/mobile/spec-mobile-react-native.md" "$AGENTS_DIR/6-specialized-development/spec-mobile-react-native.md"
migrate_agent "$AGENTS_DIR/ml/data-ml-model.md" "$AGENTS_DIR/6-specialized-development/data-ml-model.md"
migrate_agent "$AGENTS_DIR/data/ml/data-ml-model.md" "$AGENTS_DIR/6-specialized-development/data-ml-model.md"
migrate_agent "$AGENTS_DIR/data-ml-pipeline-swarm.md" "$AGENTS_DIR/6-specialized-development/data-ml-pipeline-swarm.md"
migrate_agent "$AGENTS_DIR/data-science-architect.md" "$AGENTS_DIR/6-specialized-development/data-science-architect.md"
migrate_agent "$AGENTS_DIR/doc-updater.md" "$AGENTS_DIR/6-specialized-development/doc-updater.md"
migrate_agent "$AGENTS_DIR/financial-analysis-swarm.md" "$AGENTS_DIR/6-specialized-development/financial-analysis-swarm.md"
migrate_agent "$AGENTS_DIR/financial-modeler.md" "$AGENTS_DIR/6-specialized-development/financial-modeler.md"
migrate_agent "$AGENTS_DIR/investment-analyst.md" "$AGENTS_DIR/6-specialized-development/investment-analyst.md"
migrate_agent "$AGENTS_DIR/deal-underwriter.md" "$AGENTS_DIR/6-specialized-development/deal-underwriter.md"
migrate_agent "$AGENTS_DIR/queen-investment-orchestrator.md" "$AGENTS_DIR/6-specialized-development/queen-investment-orchestrator.md"
migrate_agent "$AGENTS_DIR/trading-predictor.md" "$AGENTS_DIR/6-specialized-development/trading-predictor.md"
migrate_agent "$AGENTS_DIR/ui-visualization-swarm.md" "$AGENTS_DIR/6-specialized-development/ui-visualization-swarm.md"
migrate_agent "$AGENTS_DIR/ui_design.md" "$AGENTS_DIR/6-specialized-development/ui_design.md"
migrate_agent "$AGENTS_DIR/authentication.md" "$AGENTS_DIR/6-specialized-development/authentication.md"
migrate_agent "$AGENTS_DIR/agentic-payments.md" "$AGENTS_DIR/6-specialized-development/agentic-payments.md"
migrate_agent "$AGENTS_DIR/payments/agentic-payments.md" "$AGENTS_DIR/6-specialized-development/agentic-payments.md"
migrate_agent "$AGENTS_DIR/payments.md" "$AGENTS_DIR/6-specialized-development/payments.md"
migrate_agent "$AGENTS_DIR/app-store.md" "$AGENTS_DIR/6-specialized-development/app-store.md"
migrate_agent "$AGENTS_DIR/sandbox.md" "$AGENTS_DIR/6-specialized-development/sandbox.md"
migrate_agent "$AGENTS_DIR/user-tools.md" "$AGENTS_DIR/6-specialized-development/user-tools.md"
migrate_agent "$AGENTS_DIR/workflow.md" "$AGENTS_DIR/6-specialized-development/workflow.md"
migrate_agent "$AGENTS_DIR/agent.md" "$AGENTS_DIR/6-specialized-development/agent.md"
migrate_agent "$AGENTS_DIR/godmode-refactorer.md" "$AGENTS_DIR/6-specialized-development/godmode-refactorer.md"
migrate_agent "$AGENTS_DIR/refactor-cleaner.md" "$AGENTS_DIR/6-specialized-development/refactor-cleaner.md"
migrate_agent "$AGENTS_DIR/migration-plan.md" "$AGENTS_DIR/6-specialized-development/migration-plan.md"
migrate_agent "$AGENTS_DIR/api-docs/docs-api-openapi.md" "$AGENTS_DIR/6-specialized-development/docs-api-openapi.md"
migrate_agent "$AGENTS_DIR/documentation/api-docs/docs-api-openapi.md" "$AGENTS_DIR/6-specialized-development/docs-api-openapi.md"
migrate_agent "$AGENTS_DIR/system-design/arch-system-design.md" "$AGENTS_DIR/6-specialized-development/arch-system-design.md"
migrate_agent "$AGENTS_DIR/architecture/system-design/arch-system-design.md" "$AGENTS_DIR/6-specialized-development/arch-system-design.md"

# 7-testing-validation migrations
echo "📂 Category 7: Testing & Validation"
migrate_agent "$AGENTS_DIR/tdd-london-swarm.md" "$AGENTS_DIR/7-testing-validation/tdd-london-swarm.md"
migrate_agent "$AGENTS_DIR/unit/tdd-london-swarm.md" "$AGENTS_DIR/7-testing-validation/tdd-london-swarm.md"
migrate_agent "$AGENTS_DIR/e2e-runner.md" "$AGENTS_DIR/7-testing-validation/e2e-runner.md"
migrate_agent "$AGENTS_DIR/test-agent.md" "$AGENTS_DIR/7-testing-validation/test-agent.md"
migrate_agent "$AGENTS_DIR/test-long-runner.md" "$AGENTS_DIR/7-testing-validation/test-long-runner.md"
migrate_agent "$AGENTS_DIR/project-quality-swarm.md" "$AGENTS_DIR/7-testing-validation/project-quality-swarm.md"
migrate_agent "$AGENTS_DIR/challenges.md" "$AGENTS_DIR/7-testing-validation/challenges.md"
migrate_agent "$AGENTS_DIR/swarm.md" "$AGENTS_DIR/7-testing-validation/swarm.md"

# Copy README files
echo "📄 Copying documentation..."
migrate_agent "$AGENTS_DIR/README.md" "$AGENTS_DIR/7-testing-validation/README.md"
migrate_agent "$AGENTS_DIR/MIGRATION_SUMMARY.md" "$AGENTS_DIR/7-testing-validation/MIGRATION_SUMMARY.md"

echo ""
echo "✅ Migration complete!"
echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "ℹ️  This was a dry run. No files were actually moved."
    echo "   Run again with --execute to perform the migration."
else
    echo "📊 Summary:"
    echo "   Backup: $BACKUP_DIR"
    echo "   New structure: $AGENTS_DIR/{1..7}-*"
    echo ""
    echo "⚠️  Next steps:"
    echo "   1. Validate new structure: ./validate-agents.sh"
    echo "   2. Test agent loading: npm test"
    echo "   3. Update references: ./update-references.sh"
    echo "   4. Remove old dirs: ./cleanup-old-dirs.sh"
fi
```

## Validation Checklist

### Pre-Migration
- [ ] All 232 agent files identified and categorized
- [ ] Backup created successfully
- [ ] Migration script tested in dry-run mode
- [ ] Team notified of upcoming changes

### During Migration
- [ ] New directory structure created
- [ ] All files copied to new locations
- [ ] YAML frontmatter validated
- [ ] Internal references identified

### Post-Migration
- [ ] All 232 agents accessible in new structure
- [ ] Agent loading tests pass
- [ ] No broken references
- [ ] Documentation updated
- [ ] Performance unchanged or improved

## Risk Assessment

### High Risk
- **Breaking references**: Some agents may reference others by path
  - Mitigation: Comprehensive reference scanning and updating

- **User confusion**: Existing workflows may break
  - Mitigation: Clear documentation and migration guide

### Medium Risk
- **Duplicate handling**: Some agents exist in multiple locations
  - Mitigation: Establish canonical versions during migration

### Low Risk
- **Performance impact**: Directory structure affects load time
  - Mitigation: Benchmark before/after, optimize if needed

## Rollback Plan

If issues arise:
1. **Stop migration** immediately
2. **Restore from backup**: `cp -r .claude/agents.backup-* .claude/agents`
3. **Document issues** for future attempt
4. **Review and adjust** migration strategy

## Success Metrics

### Quantitative
- **Directory reduction**: 43 → 7 (83.7% reduction achieved)
- **Agent accessibility**: 100% of agents loadable
- **Load time**: <5% performance variance
- **Migration time**: <4 weeks total

### Qualitative
- **Discoverability**: Users can find agents faster
- **Maintainability**: Easier to add/update agents
- **Consistency**: Clear organization pattern
- **Documentation**: Comprehensive and clear

## Communication Plan

### Week 1: Announcement
- Email to all users
- Documentation update
- Migration guide published

### Week 2: Migration
- Daily progress updates
- Issue tracker for problems
- Support channel available

### Week 3: Validation
- Test results shared
- Feedback collection
- Adjustment implementation

### Week 4: Completion
- Final cleanup communicated
- Success metrics shared
- Lessons learned documented

## Future Considerations

### Potential Enhancements
1. **Agent discovery system**: Auto-register agents from categories
2. **Category-based filtering**: UI for browsing by category
3. **Usage analytics**: Track which categories are most used
4. **Dynamic loading**: Load agents on-demand by category

### Long-term Maintenance
1. **Quarterly review**: Ensure categorization remains optimal
2. **Usage patterns**: Adjust categories based on actual use
3. **New agents**: Clear guidelines for placement
4. **Documentation**: Keep migration guide updated

## Conclusion

This reorganization represents a significant improvement in agent organization, reducing complexity by 68% while improving discoverability and maintainability. The phased approach with comprehensive testing ensures minimal disruption to existing workflows.

**Recommended Action**: Proceed with Phase 1 (Preparation) immediately, followed by phased migration with validation at each step.
