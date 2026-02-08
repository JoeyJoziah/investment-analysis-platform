# Quick Fix Guide - Get Your Claude Flow V3 Working

**Status:** ⚠️ CLI Layer Broken - 15 Minutes to Fix

---

## TL;DR - 3 Commands to Fix Everything

```bash
# 1. Run the automated fix script
./.claude/scripts/fix-cli-infrastructure.sh

# 2. Validate everything works
./.claude/scripts/validate-v3-setup.sh

# 3. View your working statusline
npx @claude-flow/cli@latest hooks statusline
```

---

## What's Broken

Your `.claude` folder is **EXCELLENT** (sophisticated configuration, 134 agents, 71 skills, advanced workflows).

**BUT** the CLI layer is broken:
- npm cache corruption prevents `@claude-flow/cli` from installing
- Daemon not running (no background workers)
- Memory operations fail (can't store/retrieve patterns)
- Hooks call CLI but get no response

**Impact:** ALL V3 features are non-functional despite being perfectly configured.

---

## Manual Fix (If Automated Script Fails)

### Step 1: Clear npm corruption (30 seconds)
```bash
rm -rf ~/.npm/_npx
npm cache clean --force
```

### Step 2: Install CLI locally (1 minute)
```bash
cd /Users/devinmcgrath/Documents/GitHub/investment-analysis-platform
npm install --save-dev @claude-flow/cli@3.0.0-alpha.178
```

### Step 3: Initialize systems (2 minutes)
```bash
# Initialize memory
npx @claude-flow/cli@latest memory init --force --verbose

# Start daemon with workers
npx @claude-flow/cli@latest daemon start

# Verify daemon is running
npx @claude-flow/cli@latest daemon status
```

### Step 4: Test it works (1 minute)
```bash
# Test memory
npx @claude-flow/cli@latest memory store --key "test" --value "working" --namespace patterns
npx @claude-flow/cli@latest memory search --query "test"

# View statusline
npx @claude-flow/cli@latest hooks statusline

# List hooks
npx @claude-flow/cli@latest hooks list
```

---

## After It's Fixed

### Enable Auto-Learning
```bash
# Before each task
npx @claude-flow/cli@latest hooks pre-task --description "Your task here"

# After task success
npx @claude-flow/cli@latest hooks post-task --success true --store-results true

# Train neural patterns
npx @claude-flow/cli@latest neural train --pattern-type coordination
```

### Initialize Swarm
```bash
# Start swarm with optimal topology
npx @claude-flow/cli@latest swarm init --topology hierarchical-mesh --max-agents 15

# Check swarm status
npx @claude-flow/cli@latest swarm status
```

### Enable Background Workers
```bash
# List available workers
npx @claude-flow/cli@latest hooks worker list

# Trigger specific workers
npx @claude-flow/cli@latest hooks worker dispatch --trigger audit
npx @claude-flow/cli@latest hooks worker dispatch --trigger optimize

# Check worker status
npx @claude-flow/cli@latest hooks worker status
```

### View Metrics
```bash
# Dashboard view
npx @claude-flow/cli@latest hooks metrics --v3-dashboard

# Performance metrics
npx @claude-flow/cli@latest performance benchmark --suite all

# Statusline (shows live metrics)
npx @claude-flow/cli@latest hooks statusline --json
```

---

## What You'll Unlock

Once CLI is fixed, you'll have access to:

### Immediate Benefits
- ✅ **Continuous Learning** - Every session stores patterns for reuse
- ✅ **Memory Across Sessions** - Never repeat solved problems
- ✅ **Background Optimization** - 12 workers constantly improving code
- ✅ **Intelligent Routing** - Tasks auto-routed to optimal agents
- ✅ **Neural Predictions** - AI predicts best approach based on past success

### Cost & Performance
- 💰 **75% cost reduction** via intelligent model routing (Haiku for simple tasks)
- ⚡ **2.8-4.4x speedup** from parallel agent execution
- 🔍 **150x-12,500x faster** pattern search with HNSW indexing
- 🧠 **Self-optimizing** system that learns from every interaction

### Advanced Features
- 🐝 **Swarm Orchestration** - Coordinate multiple agents automatically
- 📊 **Quality Gates** - Automated checks at each workflow phase
- 🔄 **Session Persistence** - Resume exactly where you left off
- 🎯 **Coverage-Aware Routing** - Auto-fill test coverage gaps
- 🌐 **Cross-Project Learning** - Import patterns from other projects

---

## Validation Checklist

After running the fix script, verify:

- [ ] CLI responds to `--version`
- [ ] Daemon shows as "running"
- [ ] Memory store/retrieve/search all work
- [ ] Hooks list shows 27+ hooks
- [ ] Workers list shows 12 workers
- [ ] Statusline displays metrics
- [ ] Swarm init succeeds
- [ ] Agent spawn works

Run: `./.claude/scripts/validate-v3-setup.sh` to automate this checklist.

---

## Troubleshooting

### "CLI not found"
```bash
# Verify Node.js version (need 20+)
node --version

# Verify npm cache is clean
npm cache clean --force

# Reinstall CLI
npm install --save-dev @claude-flow/cli@3.0.0-alpha.178
```

### "Daemon won't start"
```bash
# Check for port conflicts
lsof -i :3000

# Force stop any existing daemon
pkill -f "claude-flow.*daemon"

# Restart daemon
npx @claude-flow/cli@latest daemon start
```

### "Memory init fails"
```bash
# Check memory.db permissions
ls -la .claude/memory.db

# Remove and reinitialize
rm .claude/memory.db
npx @claude-flow/cli@latest memory init --force
```

### "Hooks return errors"
```bash
# Verify hooks are in settings.local.json
cat .claude/settings.local.json | grep -A 5 "hooks"

# Test a single hook
npx @claude-flow/cli@latest hooks route --task "test task"
```

---

## Full Documentation

See `docs/CLAUDE_FOLDER_COMPREHENSIVE_REVIEW.md` for:
- Complete analysis of your setup
- Detailed optimization opportunities
- 5-phase improvement plan
- Expected performance gains
- All 134 agents explained
- 71 skills catalog
- Workflow engine deep dive

---

## Support

If you encounter issues:

1. **Run diagnostics:**
   ```bash
   npx @claude-flow/cli@latest doctor --fix
   ```

2. **Check logs:**
   ```bash
   tail -f ~/.claude-flow/daemon.log
   ```

3. **View GitHub issues:**
   https://github.com/ruvnet/claude-flow/issues

---

## Quick Reference Commands

```bash
# System Health
npx @claude-flow/cli@latest doctor --fix
npx @claude-flow/cli@latest daemon status

# Memory
npx @claude-flow/cli@latest memory init
npx @claude-flow/cli@latest memory store --key "..." --value "..." --namespace patterns
npx @claude-flow/cli@latest memory search --query "..."
npx @claude-flow/cli@latest memory list --namespace patterns

# Swarm
npx @claude-flow/cli@latest swarm init --topology hierarchical-mesh
npx @claude-flow/cli@latest swarm status
npx @claude-flow/cli@latest agent spawn -t coder --name my-coder
npx @claude-flow/cli@latest agent list

# Learning
npx @claude-flow/cli@latest hooks pretrain --model-type moe
npx @claude-flow/cli@latest neural train --pattern-type coordination
npx @claude-flow/cli@latest neural patterns --list

# Workers
npx @claude-flow/cli@latest hooks worker list
npx @claude-flow/cli@latest hooks worker dispatch --trigger audit
npx @claude-flow/cli@latest hooks worker status

# Metrics
npx @claude-flow/cli@latest hooks metrics --v3-dashboard
npx @claude-flow/cli@latest hooks statusline
npx @claude-flow/cli@latest performance benchmark --suite all
```

---

**Time to Fix:** ~15 minutes
**Time to Full Setup:** ~2-3 hours
**Payoff:** Unlimited (continuous learning, self-optimization, massive speedups)

🚀 **Get started now:** `./.claude/scripts/fix-cli-infrastructure.sh`
