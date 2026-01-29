#!/bin/bash
# agent-reorganization.sh - Reorganizes agent directory structure
# Usage: ./agent-reorganization.sh [--dry-run|--execute]

set -e

AGENTS_DIR=".claude/agents"
BACKUP_DIR=".claude/agents.backup-$(date +%Y%m%d-%H%M%S)"
DRY_RUN=${1:-"--dry-run"}

echo "🔄 Agent Directory Reorganization Script"
echo "========================================"
echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "ℹ️  Running in DRY-RUN mode (no changes will be made)"
else
    echo "⚠️  Running in EXECUTE mode (files will be moved)"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi
echo ""

# Backup current structure
echo "📦 Creating backup: $BACKUP_DIR"
if [ "$DRY_RUN" != "--dry-run" ]; then
    cp -r "$AGENTS_DIR" "$BACKUP_DIR"
    echo "   ✅ Backup created successfully"
else
    echo "   [DRY-RUN] Would create backup"
fi
echo ""

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
    if [ "$DRY_RUN" != "--dry-run" ]; then
        mkdir -p "$AGENTS_DIR/$dir"
        echo "   ✅ Created $AGENTS_DIR/$dir/"
    else
        echo "   [DRY-RUN] Would create $AGENTS_DIR/$dir/"
    fi
done
echo ""

# Migration counters
MIGRATED=0
MISSING=0
SKIPPED=0

# Migration mapping function
migrate_agent() {
    local source=$1
    local dest=$2
    local force_overwrite=${3:-"no"}

    if [ -f "$source" ]; then
        if [ -f "$dest" ] && [ "$force_overwrite" = "no" ]; then
            echo "   ⚠️  Skipped (exists): $dest"
            ((SKIPPED++))
        else
            if [ "$DRY_RUN" != "--dry-run" ]; then
                cp "$source" "$dest"
                echo "   ✅ Migrated: $(basename "$source") → $dest"
            else
                echo "   [DRY-RUN] Would migrate: $source → $dest"
            fi
            ((MIGRATED++))
        fi
    else
        echo "   ⚠️  Missing: $source"
        ((MISSING++))
    fi
}

echo "🚀 Migrating agents to new structure..."
echo ""

# 1-core migrations
echo "📂 Category 1: Core Agents (5 agents)"
migrate_agent "$AGENTS_DIR/core/coder.md" "$AGENTS_DIR/1-core/coder.md" "yes"
migrate_agent "$AGENTS_DIR/coder.md" "$AGENTS_DIR/1-core/coder.md" "no"
migrate_agent "$AGENTS_DIR/core/reviewer.md" "$AGENTS_DIR/1-core/reviewer.md" "yes"
migrate_agent "$AGENTS_DIR/reviewer.md" "$AGENTS_DIR/1-core/reviewer.md" "no"
migrate_agent "$AGENTS_DIR/core/tester.md" "$AGENTS_DIR/1-core/tester.md" "yes"
migrate_agent "$AGENTS_DIR/tester.md" "$AGENTS_DIR/1-core/tester.md" "no"
migrate_agent "$AGENTS_DIR/core/planner.md" "$AGENTS_DIR/1-core/planner.md" "yes"
migrate_agent "$AGENTS_DIR/planner.md" "$AGENTS_DIR/1-core/planner.md" "no"
migrate_agent "$AGENTS_DIR/core/researcher.md" "$AGENTS_DIR/1-core/researcher.md" "yes"
migrate_agent "$AGENTS_DIR/researcher.md" "$AGENTS_DIR/1-core/researcher.md" "no"
echo ""

# 2-swarm-coordination migrations
echo "📂 Category 2: Swarm Coordination (25 agents)"
migrate_agent "$AGENTS_DIR/swarm/hierarchical-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/hierarchical-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/hierarchical-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/hierarchical-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/swarm/mesh-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/mesh-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/mesh-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/mesh-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/swarm/adaptive-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/adaptive-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/adaptive-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/adaptive-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/collective-intelligence-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/collective-intelligence-coordinator.md"
migrate_agent "$AGENTS_DIR/queen-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/queen-coordinator.md"
migrate_agent "$AGENTS_DIR/swarm-memory-manager.md" "$AGENTS_DIR/2-swarm-coordination/swarm-memory-manager.md"
migrate_agent "$AGENTS_DIR/consensus/byzantine-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/byzantine-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/byzantine-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/byzantine-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/consensus/raft-manager.md" "$AGENTS_DIR/2-swarm-coordination/raft-manager.md" "yes"
migrate_agent "$AGENTS_DIR/raft-manager.md" "$AGENTS_DIR/2-swarm-coordination/raft-manager.md" "no"
migrate_agent "$AGENTS_DIR/consensus/gossip-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/gossip-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/gossip-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/gossip-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/consensus/crdt-synchronizer.md" "$AGENTS_DIR/2-swarm-coordination/crdt-synchronizer.md" "yes"
migrate_agent "$AGENTS_DIR/crdt-synchronizer.md" "$AGENTS_DIR/2-swarm-coordination/crdt-synchronizer.md" "no"
migrate_agent "$AGENTS_DIR/consensus/quorum-manager.md" "$AGENTS_DIR/2-swarm-coordination/quorum-manager.md" "yes"
migrate_agent "$AGENTS_DIR/quorum-manager.md" "$AGENTS_DIR/2-swarm-coordination/quorum-manager.md" "no"
migrate_agent "$AGENTS_DIR/sublinear/consensus-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/consensus-coordinator.md" "yes"
migrate_agent "$AGENTS_DIR/consensus-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/consensus-coordinator.md" "no"
migrate_agent "$AGENTS_DIR/consensus/security-manager.md" "$AGENTS_DIR/2-swarm-coordination/security-manager.md" "yes"
migrate_agent "$AGENTS_DIR/security-manager.md" "$AGENTS_DIR/2-swarm-coordination/security-manager.md" "no"
migrate_agent "$AGENTS_DIR/consensus/performance-benchmarker.md" "$AGENTS_DIR/2-swarm-coordination/performance-benchmarker.md" "yes"
migrate_agent "$AGENTS_DIR/performance-benchmarker.md" "$AGENTS_DIR/2-swarm-coordination/performance-benchmarker.md" "no"
migrate_agent "$AGENTS_DIR/optimization/load-balancer.md" "$AGENTS_DIR/2-swarm-coordination/load-balancer.md" "yes"
migrate_agent "$AGENTS_DIR/load-balancer.md" "$AGENTS_DIR/2-swarm-coordination/load-balancer.md" "no"
migrate_agent "$AGENTS_DIR/optimization/resource-allocator.md" "$AGENTS_DIR/2-swarm-coordination/resource-allocator.md" "yes"
migrate_agent "$AGENTS_DIR/resource-allocator.md" "$AGENTS_DIR/2-swarm-coordination/resource-allocator.md" "no"
migrate_agent "$AGENTS_DIR/optimization/topology-optimizer.md" "$AGENTS_DIR/2-swarm-coordination/topology-optimizer.md" "yes"
migrate_agent "$AGENTS_DIR/topology-optimizer.md" "$AGENTS_DIR/2-swarm-coordination/topology-optimizer.md" "no"
migrate_agent "$AGENTS_DIR/optimization/performance-monitor.md" "$AGENTS_DIR/2-swarm-coordination/performance-monitor.md" "yes"
migrate_agent "$AGENTS_DIR/performance-monitor.md" "$AGENTS_DIR/2-swarm-coordination/performance-monitor.md" "no"
migrate_agent "$AGENTS_DIR/optimization/benchmark-suite.md" "$AGENTS_DIR/2-swarm-coordination/benchmark-suite.md" "yes"
migrate_agent "$AGENTS_DIR/benchmark-suite.md" "$AGENTS_DIR/2-swarm-coordination/benchmark-suite.md" "no"
migrate_agent "$AGENTS_DIR/worker-specialist.md" "$AGENTS_DIR/2-swarm-coordination/worker-specialist.md"
migrate_agent "$AGENTS_DIR/safla-neural.md" "$AGENTS_DIR/2-swarm-coordination/safla-neural.md"
migrate_agent "$AGENTS_DIR/memory-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/memory-coordinator.md"
migrate_agent "$AGENTS_DIR/orchestrator-task.md" "$AGENTS_DIR/2-swarm-coordination/orchestrator-task.md"
migrate_agent "$AGENTS_DIR/scout-explorer.md" "$AGENTS_DIR/2-swarm-coordination/scout-explorer.md"
migrate_agent "$AGENTS_DIR/team-coordinator.md" "$AGENTS_DIR/2-swarm-coordination/team-coordinator.md"
echo ""

# 3-security-performance migrations
echo "📂 Category 3: Security & Performance (15 agents)"
migrate_agent "$AGENTS_DIR/security-reviewer.md" "$AGENTS_DIR/3-security-performance/security-reviewer.md"
migrate_agent "$AGENTS_DIR/security-compliance-swarm.md" "$AGENTS_DIR/3-security-performance/security-compliance-swarm.md"
migrate_agent "$AGENTS_DIR/performance-analyzer.md" "$AGENTS_DIR/3-security-performance/performance-analyzer.md"
migrate_agent "$AGENTS_DIR/sublinear/performance-optimizer.md" "$AGENTS_DIR/3-security-performance/performance-optimizer.md" "yes"
migrate_agent "$AGENTS_DIR/performance-optimizer.md" "$AGENTS_DIR/3-security-performance/performance-optimizer.md" "no"
migrate_agent "$AGENTS_DIR/sublinear/matrix-optimizer.md" "$AGENTS_DIR/3-security-performance/matrix-optimizer.md" "yes"
migrate_agent "$AGENTS_DIR/matrix-optimizer.md" "$AGENTS_DIR/3-security-performance/matrix-optimizer.md" "no"
migrate_agent "$AGENTS_DIR/sublinear/pagerank-analyzer.md" "$AGENTS_DIR/3-security-performance/pagerank-analyzer.md" "yes"
migrate_agent "$AGENTS_DIR/pagerank-analyzer.md" "$AGENTS_DIR/3-security-performance/pagerank-analyzer.md" "no"
migrate_agent "$AGENTS_DIR/sublinear/trading-predictor.md" "$AGENTS_DIR/3-security-performance/trading-predictor.md" "yes"
migrate_agent "$AGENTS_DIR/trading-predictor.md" "$AGENTS_DIR/3-security-performance/trading-predictor.md" "no"
migrate_agent "$AGENTS_DIR/sona/sona-learning-optimizer.md" "$AGENTS_DIR/3-security-performance/sona-learning-optimizer.md" "yes"
migrate_agent "$AGENTS_DIR/sona-learning-optimizer.md" "$AGENTS_DIR/3-security-performance/sona-learning-optimizer.md" "no"
migrate_agent "$AGENTS_DIR/neural-network.md" "$AGENTS_DIR/3-security-performance/neural-network.md"
migrate_agent "$AGENTS_DIR/risk-assessor.md" "$AGENTS_DIR/3-security-performance/risk-assessor.md"
migrate_agent "$AGENTS_DIR/architecture-reviewer.md" "$AGENTS_DIR/3-security-performance/architecture-reviewer.md"
migrate_agent "$AGENTS_DIR/analysis/code-analyzer.md" "$AGENTS_DIR/3-security-performance/code-analyzer.md" "yes"
migrate_agent "$AGENTS_DIR/code-analyzer.md" "$AGENTS_DIR/3-security-performance/code-analyzer.md" "no"
migrate_agent "$AGENTS_DIR/analysis/code-review/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md" "yes"
migrate_agent "$AGENTS_DIR/analysis/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md" "no"
migrate_agent "$AGENTS_DIR/code-review/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md" "no"
migrate_agent "$AGENTS_DIR/analyze-code-quality.md" "$AGENTS_DIR/3-security-performance/analyze-code-quality.md" "no"
echo ""

# 4-github-repository migrations
echo "📂 Category 4: GitHub & Repository (20 agents)"
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
echo ""

# 5-sparc-methodology migrations
echo "📂 Category 5: SPARC Methodology (10 agents)"
migrate_agent "$AGENTS_DIR/specification.md" "$AGENTS_DIR/5-sparc-methodology/specification.md"
migrate_agent "$AGENTS_DIR/pseudocode.md" "$AGENTS_DIR/5-sparc-methodology/pseudocode.md"
migrate_agent "$AGENTS_DIR/architecture.md" "$AGENTS_DIR/5-sparc-methodology/architecture.md"
migrate_agent "$AGENTS_DIR/refinement.md" "$AGENTS_DIR/5-sparc-methodology/refinement.md"
migrate_agent "$AGENTS_DIR/implementer-sparc-coder.md" "$AGENTS_DIR/5-sparc-methodology/implementer-sparc-coder.md"
migrate_agent "$AGENTS_DIR/architect.md" "$AGENTS_DIR/5-sparc-methodology/architect.md"
migrate_agent "$AGENTS_DIR/tdd-guide.md" "$AGENTS_DIR/5-sparc-methodology/tdd-guide.md"
migrate_agent "$AGENTS_DIR/goal-planner.md" "$AGENTS_DIR/5-sparc-methodology/goal-planner.md"
migrate_agent "$AGENTS_DIR/code-goal-planner.md" "$AGENTS_DIR/5-sparc-methodology/code-goal-planner.md"
echo ""

# 6-specialized-development migrations
echo "📂 Category 6: Specialized Development (35 agents)"
migrate_agent "$AGENTS_DIR/backend-api-swarm.md" "$AGENTS_DIR/6-specialized-development/backend-api-swarm.md"
migrate_agent "$AGENTS_DIR/development/backend/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md" "yes"
migrate_agent "$AGENTS_DIR/development/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md" "no"
migrate_agent "$AGENTS_DIR/backend/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md" "no"
migrate_agent "$AGENTS_DIR/dev-backend-api.md" "$AGENTS_DIR/6-specialized-development/dev-backend-api.md" "no"
migrate_agent "$AGENTS_DIR/specialized/mobile/spec-mobile-react-native.md" "$AGENTS_DIR/6-specialized-development/spec-mobile-react-native.md" "yes"
migrate_agent "$AGENTS_DIR/mobile/spec-mobile-react-native.md" "$AGENTS_DIR/6-specialized-development/spec-mobile-react-native.md" "no"
migrate_agent "$AGENTS_DIR/data/ml/data-ml-model.md" "$AGENTS_DIR/6-specialized-development/data-ml-model.md" "yes"
migrate_agent "$AGENTS_DIR/ml/data-ml-model.md" "$AGENTS_DIR/6-specialized-development/data-ml-model.md" "no"
migrate_agent "$AGENTS_DIR/data-ml-pipeline-swarm.md" "$AGENTS_DIR/6-specialized-development/data-ml-pipeline-swarm.md"
migrate_agent "$AGENTS_DIR/data-science-architect.md" "$AGENTS_DIR/6-specialized-development/data-science-architect.md"
migrate_agent "$AGENTS_DIR/doc-updater.md" "$AGENTS_DIR/6-specialized-development/doc-updater.md"
migrate_agent "$AGENTS_DIR/financial-analysis-swarm.md" "$AGENTS_DIR/6-specialized-development/financial-analysis-swarm.md"
migrate_agent "$AGENTS_DIR/financial-modeler.md" "$AGENTS_DIR/6-specialized-development/financial-modeler.md"
migrate_agent "$AGENTS_DIR/investment-analyst.md" "$AGENTS_DIR/6-specialized-development/investment-analyst.md"
migrate_agent "$AGENTS_DIR/deal-underwriter.md" "$AGENTS_DIR/6-specialized-development/deal-underwriter.md"
migrate_agent "$AGENTS_DIR/queen-investment-orchestrator.md" "$AGENTS_DIR/6-specialized-development/queen-investment-orchestrator.md"
migrate_agent "$AGENTS_DIR/ui-visualization-swarm.md" "$AGENTS_DIR/6-specialized-development/ui-visualization-swarm.md"
migrate_agent "$AGENTS_DIR/ui_design.md" "$AGENTS_DIR/6-specialized-development/ui_design.md"
migrate_agent "$AGENTS_DIR/authentication.md" "$AGENTS_DIR/6-specialized-development/authentication.md"
migrate_agent "$AGENTS_DIR/payments/agentic-payments.md" "$AGENTS_DIR/6-specialized-development/agentic-payments.md" "yes"
migrate_agent "$AGENTS_DIR/agentic-payments.md" "$AGENTS_DIR/6-specialized-development/agentic-payments.md" "no"
migrate_agent "$AGENTS_DIR/payments.md" "$AGENTS_DIR/6-specialized-development/payments.md"
migrate_agent "$AGENTS_DIR/app-store.md" "$AGENTS_DIR/6-specialized-development/app-store.md"
migrate_agent "$AGENTS_DIR/sandbox.md" "$AGENTS_DIR/6-specialized-development/sandbox.md"
migrate_agent "$AGENTS_DIR/user-tools.md" "$AGENTS_DIR/6-specialized-development/user-tools.md"
migrate_agent "$AGENTS_DIR/workflow.md" "$AGENTS_DIR/6-specialized-development/workflow.md"
migrate_agent "$AGENTS_DIR/agent.md" "$AGENTS_DIR/6-specialized-development/agent.md"
migrate_agent "$AGENTS_DIR/godmode-refactorer.md" "$AGENTS_DIR/6-specialized-development/godmode-refactorer.md"
migrate_agent "$AGENTS_DIR/refactor-cleaner.md" "$AGENTS_DIR/6-specialized-development/refactor-cleaner.md"
migrate_agent "$AGENTS_DIR/migration-plan.md" "$AGENTS_DIR/6-specialized-development/migration-plan.md"
migrate_agent "$AGENTS_DIR/documentation/api-docs/docs-api-openapi.md" "$AGENTS_DIR/6-specialized-development/docs-api-openapi.md" "yes"
migrate_agent "$AGENTS_DIR/api-docs/docs-api-openapi.md" "$AGENTS_DIR/6-specialized-development/docs-api-openapi.md" "no"
migrate_agent "$AGENTS_DIR/architecture/system-design/arch-system-design.md" "$AGENTS_DIR/6-specialized-development/arch-system-design.md" "yes"
migrate_agent "$AGENTS_DIR/system-design/arch-system-design.md" "$AGENTS_DIR/6-specialized-development/arch-system-design.md" "no"
echo ""

# 7-testing-validation migrations
echo "📂 Category 7: Testing & Validation (10 agents)"
migrate_agent "$AGENTS_DIR/unit/tdd-london-swarm.md" "$AGENTS_DIR/7-testing-validation/tdd-london-swarm.md" "yes"
migrate_agent "$AGENTS_DIR/tdd-london-swarm.md" "$AGENTS_DIR/7-testing-validation/tdd-london-swarm.md" "no"
migrate_agent "$AGENTS_DIR/e2e-runner.md" "$AGENTS_DIR/7-testing-validation/e2e-runner.md"
migrate_agent "$AGENTS_DIR/test-agent.md" "$AGENTS_DIR/7-testing-validation/test-agent.md"
migrate_agent "$AGENTS_DIR/test-long-runner.md" "$AGENTS_DIR/7-testing-validation/test-long-runner.md"
migrate_agent "$AGENTS_DIR/project-quality-swarm.md" "$AGENTS_DIR/7-testing-validation/project-quality-swarm.md"
migrate_agent "$AGENTS_DIR/challenges.md" "$AGENTS_DIR/7-testing-validation/challenges.md"
migrate_agent "$AGENTS_DIR/swarm.md" "$AGENTS_DIR/7-testing-validation/swarm.md"
echo ""

# Copy README and documentation
echo "📄 Copying documentation..."
if [ -f "$AGENTS_DIR/README.md" ]; then
    migrate_agent "$AGENTS_DIR/README.md" "$AGENTS_DIR/README.old.md"
fi
echo ""

echo "✅ Migration process complete!"
echo ""
echo "📊 Summary:"
echo "   Migrated: $MIGRATED files"
echo "   Missing:  $MISSING files"
echo "   Skipped:  $SKIPPED files (already exist)"
echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "ℹ️  This was a DRY-RUN. No files were actually moved."
    echo "   Run again with --execute to perform the migration:"
    echo "   ./agent-reorganization.sh --execute"
else
    echo "📦 Backup location: $BACKUP_DIR"
    echo ""
    echo "⚠️  Next steps:"
    echo "   1. Verify migration: ls -la $AGENTS_DIR/{1..7}-*/"
    echo "   2. Test agent loading"
    echo "   3. Update documentation"
    echo "   4. Remove old directories (after validation)"
    echo ""
    echo "   To rollback: rm -rf $AGENTS_DIR && mv $BACKUP_DIR $AGENTS_DIR"
fi
