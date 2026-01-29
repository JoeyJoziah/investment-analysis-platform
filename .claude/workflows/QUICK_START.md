# Intelligent Debug Workflow - Quick Start Guide

## 🚀 Getting Started in 60 Seconds

### Option 1: Automatic Execution (Simplest)

Just run the Python script:

```bash
# Execute workflow for current repository issues
python .claude/workflows/execute_debug_workflow.py

# With specific bug context
python .claude/workflows/execute_debug_workflow.py \
  --bug-id "WAVE-4.5" \
  --context "7 test errors in integration suite"

# With auto-deploy enabled
python .claude/workflows/execute_debug_workflow.py --auto-deploy

# Dry run (no actual changes)
python .claude/workflows/execute_debug_workflow.py --dry-run
```

### Option 2: Using Claude Code Task Tool (Recommended)

In your Claude Code conversation:

```
Execute the intelligent debug workflow to fix the remaining 7 test errors.

Use the workflow defined in .claude/workflows/intelligent-debug-workflow.json with:
- Hierarchical-mesh topology
- 8 max agents
- Auto-deploy: false (require approval)
- Focus on integration test failures
```

Claude Code will automatically spawn the swarm coordinator and execute all 7 phases.

### Option 3: Using CLI (Advanced)

```bash
# Via Claude Flow CLI
npx @claude-flow/cli@latest workflow execute intelligent-debug-workflow

# With custom configuration
npx @claude-flow/cli@latest workflow execute intelligent-debug-workflow \
  --config .claude/workflows/config.json \
  --auto-deploy false
```

---

## 📋 What It Does

The workflow automatically:

1. ✅ **Detects bugs** using security scanner, test analyzer, and code reviewer
2. ✅ **Analyzes deeply** with swarm-coordinated debug agents (75% consensus)
3. ✅ **Researches solutions** using web search, docs, and internal patterns
4. ✅ **Reviews the plan** with 3 validators (85% consensus required)
5. ✅ **Executes fixes** in parallel with load balancing (Haiku agents)
6. ✅ **Validates everything** with tests, security scans, and code review
7. ✅ **Deploys** via PR creation and 24h monitoring

**No human intervention needed** (unless plan rejected or validation fails)

---

## 🎯 Example Usage

### Example 1: Fix Test Failures

```bash
# Wave 4.5 scenario: 7 test errors remaining
python .claude/workflows/execute_debug_workflow.py \
  --bug-id "WAVE-4.5" \
  --context "Transaction.trade_date and Stock.industry_id schema mismatches"
```

**Expected Output:**
```
🚀 Starting Intelligent Debug Workflow
⏰ Started at: 2026-01-28T19:30:00

=================================================================
📡 PHASE 1: Bug Detection & Initial Analysis
=================================================================
🤖 Spawning detection agents:
  ├─ security-agent: Scan for vulnerabilities
  ├─ test-agent: Analyze test failures
  └─ code-reviewer: Review code quality

📊 Aggregating findings...
✅ Phase 1 completed in 2.35s
📌 Total findings: 5

=================================================================
🔍 PHASE 2: Deep Debug with Swarm Coordination
=================================================================
🐝 Initializing debug swarm (hierarchical-mesh)
🤖 Spawning 3 debug agents...
🎯 Consensus achieved: 87% (threshold: 75%)
✅ Phase 2 completed in 3.42s

... (continues through all 7 phases)

✅ Workflow completed successfully!
📊 Duration: 45.2s
💾 Report saved to: .claude/workflows/reports/workflow_report_20260128_193000.json
```

---

## 🔧 Configuration

### Customize Behavior

Edit `.claude/workflows/config.json`:

```json
{
  "intelligent-debug-workflow": {
    "detection": {
      "auto_trigger": true,
      "threshold": {
        "test_failures": 3,      // Trigger after 3 failures
        "security_critical": 1   // Trigger on any critical security issue
      }
    },
    "debug": {
      "consensus_threshold": 0.75,  // Lower = easier to reach consensus
      "max_agents": 10              // Increase for complex issues
    },
    "orchestration": {
      "max_concurrent": 8,          // More parallel execution
      "model_preference": "haiku"   // Cost-optimized
    },
    "deployment": {
      "auto_deploy": false          // Require manual PR approval
    }
  }
}
```

---

## 📊 Monitoring & Reports

### View Execution Reports

```bash
# Latest report
cat .claude/workflows/reports/workflow_report_*.json | tail -1 | jq '.'

# Workflow metrics
cat .claude/workflows/reports/workflow_report_*.json | jq '.metrics'

# Success rate
ls .claude/workflows/reports/*.json | wc -l
grep -r "\"status\": \"success\"" .claude/workflows/reports/ | wc -l
```

### Live Monitoring

While workflow runs, monitor progress:

```bash
# Watch workflow state (if using background execution)
watch -n 2 'tail -20 .claude/workflows/workflow.log'

# Monitor agent activity
npx @claude-flow/cli@latest swarm status
```

---

## 🐛 Troubleshooting

### Workflow Stuck?

```bash
# Check current state
cat .claude/workflows/state_*.json | tail -1 | jq '.current_phase'

# Resume from saved state
python .claude/workflows/execute_debug_workflow.py \
  --resume .claude/workflows/state_20260128_193000.json
```

### Plan Keep Getting Rejected?

Lower the consensus threshold:

```json
{
  "debug": {
    "consensus_threshold": 0.70  // Default: 0.75
  }
}
```

### Too Slow?

Use more concurrent agents:

```json
{
  "orchestration": {
    "max_concurrent": 10,        // Default: 5
    "model_preference": "haiku"  // Faster than sonnet
  }
}
```

---

## 🎓 Advanced Features

### Custom Swarm Strategies

Create `.claude/workflows/strategies/custom-strategy.js`:

```javascript
module.exports = {
  name: "aggressive-parallel",
  max_concurrent: 15,
  retry_failed: true,
  auto_escalate_to_sonnet: true
};
```

Use it:

```bash
python .claude/workflows/execute_debug_workflow.py \
  --strategy .claude/workflows/strategies/custom-strategy.js
```

### Integration with CI/CD

Add to `.github/workflows/intelligent-debug.yml`:

```yaml
name: Intelligent Debug
on: [push, pull_request]

jobs:
  debug:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Intelligent Debug Workflow
        run: |
          python .claude/workflows/execute_debug_workflow.py \
            --auto-deploy false
```

---

## 📚 Full Documentation

- **Detailed Guide**: [workflow-executor.md](./workflow-executor.md)
- **Configuration**: [intelligent-debug-workflow.json](./intelligent-debug-workflow.json)
- **Architecture**: See flowchart in workflow-executor.md

---

## ✅ Quick Checklist

Before running the workflow:

- [ ] Repository is clean (no uncommitted changes)
- [ ] Tests are currently failing (to have bugs to fix)
- [ ] Configuration reviewed (auto-deploy setting)
- [ ] Claude Flow CLI installed (if using CLI option)
- [ ] Python 3.8+ available (if using Python script)

After workflow completes:

- [ ] Review the generated PR
- [ ] Check validation results
- [ ] Review workflow report
- [ ] Merge PR if approved
- [ ] Monitor for 24h

---

## 🎯 Success Metrics

Typical workflow performance:

- **Detection**: 2-3 seconds
- **Debug**: 3-5 seconds
- **Research**: 4-6 seconds
- **Review**: 2-3 seconds
- **Orchestration**: 10-15 seconds
- **Validation**: 5-10 seconds
- **Deployment**: 2-3 seconds

**Total**: 30-50 seconds for complete bug detection → fix → deploy cycle

---

## 🆘 Need Help?

1. Check the [full documentation](./workflow-executor.md)
2. Review [example workflows](./ examples/)
3. Check [troubleshooting guide](./workflow-executor.md#troubleshooting)
4. Open an issue on GitHub

---

**Version**: 1.0.0
**Last Updated**: 2026-01-28
**Status**: Production Ready ✅
