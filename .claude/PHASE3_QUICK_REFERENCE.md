# Phase 3 Quick Reference

## Hook Commands

### workflow-swarm-sync.sh

```bash
# Initialize
bash .claude/hooks/workflow-swarm-sync.sh init

# Sync phase
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  <workflow_id> <phase_name> <status> [json_data]

# Record output
bash .claude/hooks/workflow-swarm-sync.sh record-output \
  <workflow_id> <phase_name> <output_type> [json_data]

# Get state
bash .claude/hooks/workflow-swarm-sync.sh get-state <workflow_id>

# List workflows
bash .claude/hooks/workflow-swarm-sync.sh list-workflows
```

### swarm-consensus-sync.sh

```bash
# Handle workflow update
bash .claude/hooks/swarm-consensus-sync.sh workflow-update \
  <workflow_id> <phase_name> <status>

# Record decision
bash .claude/hooks/swarm-consensus-sync.sh record-decision \
  <workflow_id> <phase_name> <decision> [json_data]

# Get consensus
bash .claude/hooks/swarm-consensus-sync.sh get-consensus \
  <workflow_id> <phase_name>

# List pending
bash .claude/hooks/swarm-consensus-sync.sh list-pending
```

## Memory Operations

```bash
# List orchestration entries
npx @claude-flow/cli@latest memory list --namespace orchestration

# Search for specific workflow
npx @claude-flow/cli@latest memory search \
  --namespace orchestration \
  --query "workflow_<id>"

# Retrieve specific entry
npx @claude-flow/cli@latest memory retrieve \
  --namespace orchestration \
  --key "<key>"
```

## Verification

```bash
# Run full verification
bash .claude/scripts/verify-phase3-integration.sh

# Quick check
npx @claude-flow/cli@latest memory list --namespace orchestration
```

## Common Workflows

### Complete Feature Workflow

```bash
# 1. Start intake phase
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  "wf-feature-123" "intake" "started"

# 2. Complete intake
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  "wf-feature-123" "intake" "completed" \
  '{"plan_approved":true}'

# 3. Start design (requires consensus)
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  "wf-feature-123" "design" "pending_consensus"

# 4. Record consensus approval
bash .claude/hooks/swarm-consensus-sync.sh record-decision \
  "wf-feature-123" "design" "approved" \
  '{"approvers":["architect","security"]}'

# 5. Continue with build phase
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  "wf-feature-123" "build" "started"
```

## Configuration

Key settings in `.claude/config/workflow-engine.json`:

- `coordination.bidirectional_sync.enabled`: Enable bidirectional sync
- `coordination.consensus_protocol.required_phases`: Phases requiring consensus
- `coordination.audit_trail.enabled`: Enable audit trail
- `hooks.workflow_swarm_sync.enabled`: Enable workflow-swarm sync hook
- `hooks.swarm_consensus_sync.enabled`: Enable consensus sync hook

## Troubleshooting

```bash
# Make hooks executable
chmod +x .claude/hooks/*.sh

# Reinitialize namespace
bash .claude/hooks/workflow-swarm-sync.sh init

# Check daemon
npx @claude-flow/cli@latest daemon status

# View recent entries
npx @claude-flow/cli@latest memory list \
  --namespace orchestration \
  --limit 10
```
