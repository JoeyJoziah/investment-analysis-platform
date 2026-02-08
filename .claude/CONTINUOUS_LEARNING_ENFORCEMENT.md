# Continuous Learning Enforcement Guide

**Status:** ✅ ACTIVE - Mandatory for All Tasks
**Compliance:** REQUIRED - No Exceptions

---

## 🚨 MANDATORY REQUIREMENTS

### Every Task MUST Include:

1. **Pre-Task Hook** (BEFORE starting work)
2. **Post-Task Hook** (AFTER completing work)
3. **Memory Search** (Check for existing solutions)
4. **Pattern Storage** (Store successful approaches)

**Failure to comply prevents:**
- ❌ 75% cost savings via intelligent routing
- ❌ 150x-12,500x faster pattern search
- ❌ Cross-session knowledge persistence
- ❌ Self-optimization and continuous improvement

---

## ✅ Mandatory Workflow Template

**Copy and use this for EVERY task:**

```bash
# ====================================
# MANDATORY CONTINUOUS LEARNING WORKFLOW
# ====================================

# 1. Generate unique task ID
TASK_ID="task-$(date +%s)"

# 2. MANDATORY: Pre-task hook (MUST run before work starts)
npx @claude-flow/cli@latest hooks pre-task \
  --task-id "$TASK_ID" \
  --description "Detailed description of what you're doing"

# 3. Search memory for existing solutions
npx @claude-flow/cli@latest memory search \
  --query "keywords related to your task" \
  --namespace patterns \
  --limit 5

# 4. Do the actual work (use agent/model from pre-task recommendation)
# ... your work here ...

# 5. MANDATORY: Post-task hook (MUST run after work completes)
npx @claude-flow/cli@latest hooks post-task \
  --task-id "$TASK_ID" \
  --success true \
  --store-results true

# 6. Store specific learnings
npx @claude-flow/cli@latest memory store \
  --namespace patterns \
  --key "descriptive-key-$(date +%s)" \
  --value "What you learned or what worked"

# 7. Trigger relevant background worker (if applicable)
# npx @claude-flow/cli@latest hooks worker dispatch --trigger optimize  # For performance work
# npx @claude-flow/cli@latest hooks worker dispatch --trigger audit     # For security work
# npx @claude-flow/cli@latest hooks worker dispatch --trigger testgaps  # After adding features
```

---

## 📊 Pre-Task Hook Benefits

**What you get from running pre-task:**

### 1. Intelligent Agent Routing
```
Suggested Agents
+------------+------------+-------------------------------------+
| Agent Type | Confidence | Reason                              |
+------------+------------+-------------------------------------+
| tester     |      95.0% | Primary agent for testing tasks     |
| reviewer   |      90.0% | Alternative agent for review        |
+------------+------------+-------------------------------------+
```

### 2. Cost-Optimized Model Recommendations
```
Intelligent Model Routing
  Tier 2: SONNET
  Complexity: 35%
  Est. Latency: 2000ms | Cost: $0.0030

[TASK_MODEL_RECOMMENDATION] Use model="sonnet" for this task
```

**Cost Savings:**
- Tier 1 (Agent Booster): $0 - Simple transforms
- Tier 2 (Haiku): $0.0002 - Simple tasks
- Tier 3 (Sonnet/Opus): $0.003-$0.015 - Complex work

**Result:** 75% average cost reduction

### 3. Complexity Estimation
```
Task Registered
| Task ID: test-1769660000                     |
| Description: Fix authentication bug          |
| Complexity: LOW                              |
| Est. Duration: 10-30 min                     |
```

### 4. Pattern Search Results
- Automatically searches for similar past solutions
- Prevents re-solving identical problems
- 150x-12,500x faster with HNSW indexing

---

## 🎯 Post-Task Hook Benefits

**What you get from running post-task:**

### 1. Pattern Learning
```
Task outcome recorded: SUCCESS
Patterns Updated: 2
New Patterns: 1
```

### 2. Neural Trajectory Creation
```
Trajectory ID: traj-1769660000
```

**Trajectories track:**
- Sequence of steps that led to success
- Agent decisions and routing
- Model selections
- Outcome quality

### 3. Cross-Session Memory
- Patterns persist across conversations
- Knowledge accumulates over time
- Self-optimization improves continuously

### 4. Metrics Tracking
- Success rates
- Time to completion
- Cost per task type
- Pattern confidence scores

---

## 🔍 Validation Commands

**Check if continuous learning is working:**

### 1. Quick Health Check
```bash
# Should show 80+ patterns, 88+ trajectories
npx @claude-flow/cli@latest neural patterns --list

# Should show entries in patterns namespace
npx @claude-flow/cli@latest memory list --namespace patterns --limit 10

# Should show daemon running with workers
npx @claude-flow/cli@latest daemon status
```

### 2. Full Validation
```bash
# Run comprehensive validation script
./.claude/scripts/validate-continuous-learning.sh
```

### 3. Test Learning Pipeline
```bash
# Test pre-task hook
npx @claude-flow/cli@latest hooks pre-task \
  --task-id "test-$(date +%s)" \
  --description "Test continuous learning"

# Test post-task hook
npx @claude-flow/cli@latest hooks post-task \
  --task-id "test-123" \
  --success true \
  --store-results true

# Verify pattern was created
npx @claude-flow/cli@latest neural patterns --list
```

---

## 📈 Optimization Targets

### Current Status
- ✅ Neural Patterns: 80+ stored
- ✅ Trajectories: 88+ learning paths
- ✅ Memory Database: HNSW-indexed
- ✅ Daemon: Running with 5 workers

### Short-term Targets (1 month)
- 🎯 Neural Patterns: 200+ (growth from continuous use)
- 🎯 Trajectories: 300+ (more learning paths)
- 🎯 Cost Savings: 40-60% realized
- 🎯 Pattern Confidence: 80%+ average

### Long-term Targets (3 months)
- 🎯 Neural Patterns: 500+ (comprehensive knowledge base)
- 🎯 Trajectories: 1000+ (deep learning)
- 🎯 Cost Savings: 75% fully realized
- 🎯 Pattern Confidence: 90%+ average
- 🎯 Repeat Problems: <5% (near-zero)

---

## 🛠️ Troubleshooting

### Issue: Pre-task hook fails
**Symptoms:**
```
[ERROR] Required option missing: --task-id
```

**Solution:**
```bash
# Always provide both --task-id and --description
npx @claude-flow/cli@latest hooks pre-task \
  --task-id "task-$(date +%s)" \
  --description "Your task description"
```

### Issue: Post-task hook doesn't create patterns
**Symptoms:**
- Pattern count doesn't increase
- No new trajectories

**Solution:**
```bash
# 1. Verify daemon is running
npx @claude-flow/cli@latest daemon status

# 2. If not running, start it
npx @claude-flow/cli@latest daemon start

# 3. Re-run post-task hook
npx @claude-flow/cli@latest hooks post-task \
  --task-id "$TASK_ID" \
  --success true \
  --store-results true
```

### Issue: Memory search returns no results
**Symptoms:**
```
[WARN] No results found
```

**Solution:**
```bash
# 1. Check if memory database has entries
npx @claude-flow/cli@latest memory list

# 2. If empty, initialize
npx @claude-flow/cli@latest memory init --force --verbose

# 3. Train on existing codebase
npx @claude-flow/cli@latest hooks pretrain --model-type moe --epochs 10
```

### Issue: Daemon not running
**Symptoms:**
- Hooks fail silently
- No pattern learning
- Workers not processing

**Solution:**
```bash
# Start daemon
npx @claude-flow/cli@latest daemon start

# Verify it's running
npx @claude-flow/cli@latest daemon status

# Check workers are enabled
npx @claude-flow/cli@latest hooks worker list
```

---

## 🎓 Best Practices

### 1. Always Generate Unique Task IDs
```bash
# GOOD: Unique timestamp-based ID
TASK_ID="task-$(date +%s)"

# BAD: Reusing same ID
TASK_ID="my-task"  # ❌ Will conflict
```

### 2. Use Descriptive Task Descriptions
```bash
# GOOD: Specific and detailed
--description "Fix authentication bug in login endpoint by validating JWT expiration"

# BAD: Too vague
--description "Fix bug"  # ❌ Not useful for pattern learning
```

### 3. Store Specific Learnings
```bash
# GOOD: Actionable pattern
npx @claude-flow/cli@latest memory store \
  --key "csrf-double-submit-pattern" \
  --value "Use double-submit cookie pattern with SameSite=Strict and secure flag for CSRF protection" \
  --namespace patterns

# BAD: Too vague
npx @claude-flow/cli@latest memory store \
  --key "security" \
  --value "fixed security"  # ❌ Not useful
```

### 4. Use Recommended Models
```bash
# After pre-task hook recommends model, use it in Task tool
Task({
  prompt: "...",
  subagent_type: "coder",
  model: "haiku"  // ← FROM PRE-TASK RECOMMENDATION
})

# Don't ignore recommendations - that's where cost savings come from
```

### 5. Trigger Appropriate Workers
```bash
# After performance optimization
npx @claude-flow/cli@latest hooks worker dispatch --trigger optimize

# After adding features
npx @claude-flow/cli@latest hooks worker dispatch --trigger testgaps

# After security changes
npx @claude-flow/cli@latest hooks worker dispatch --trigger audit

# After API changes
npx @claude-flow/cli@latest hooks worker dispatch --trigger document
```

---

## 📋 Quick Reference Card

**Print this and keep nearby:**

```
┌─────────────────────────────────────────────────────────────┐
│ MANDATORY CONTINUOUS LEARNING CHECKLIST                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ BEFORE ANY TASK:                                            │
│ □ Generate task ID: TASK_ID="task-$(date +%s)"             │
│ □ Run pre-task hook with --task-id and --description       │
│ □ Search memory for similar solutions                      │
│ □ Use recommended agent and model                          │
│                                                              │
│ AFTER ANY TASK:                                             │
│ □ Run post-task hook with --success and --store-results    │
│ □ Store specific learnings in memory                       │
│ □ Trigger appropriate background worker                    │
│ □ Verify pattern count increased                           │
│                                                              │
│ VALIDATION:                                                 │
│ □ Weekly: Run ./.claude/scripts/validate-continuous-       │
│   learning.sh                                               │
│ □ Monthly: Review metrics dashboard                        │
│ □ Quarterly: Analyze cost savings                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

**If you haven't been using continuous learning:**

### Step 1: Validate Current State
```bash
# Check if daemon is running
npx @claude-flow/cli@latest daemon status

# Check if patterns exist
npx @claude-flow/cli@latest neural patterns --list

# Check memory database
npx @claude-flow/cli@latest memory list --namespace patterns
```

### Step 2: Fix Any Issues
```bash
# Start daemon if not running
npx @claude-flow/cli@latest daemon start

# Initialize memory if empty
npx @claude-flow/cli@latest memory init --force --verbose

# Train on existing codebase
npx @claude-flow/cli@latest hooks pretrain --model-type moe --epochs 10
```

### Step 3: Start Using on Next Task
```bash
# Use the mandatory workflow template above
# Make it a habit - EVERY task needs pre/post hooks
```

### Step 4: Monitor Progress
```bash
# Weekly validation
./.claude/scripts/validate-continuous-learning.sh

# View metrics
npx @claude-flow/cli@latest hooks metrics --v3-dashboard
```

---

## 📊 Success Metrics

**How to measure if continuous learning is working:**

### Week 1
- ✓ Pre-task hooks used on 80%+ of tasks
- ✓ Post-task hooks used on 80%+ of tasks
- ✓ Pattern count increases by 10-20

### Month 1
- ✓ Pre-task hooks used on 95%+ of tasks
- ✓ Post-task hooks used on 95%+ of tasks
- ✓ Pattern count: 150-200
- ✓ Cost savings: 30-40% realized
- ✓ Repeat problems: <20%

### Quarter 1
- ✓ Pre-task hooks used on 100% of tasks (automated habit)
- ✓ Post-task hooks used on 100% of tasks (automated habit)
- ✓ Pattern count: 500+
- ✓ Cost savings: 75% fully realized
- ✓ Repeat problems: <5%
- ✓ Pattern confidence: 90%+

---

**Remember: Continuous learning is not optional. It's the difference between a good development environment and a self-optimizing, cost-efficient, knowledge-accumulating powerhouse.**

**Every task without pre/post hooks is a missed opportunity for improvement.**
