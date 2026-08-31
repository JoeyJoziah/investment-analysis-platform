# Phase 3: Swarm-Workflow Coordination

> **SUPERSEDED (2026-06): historical snapshot, not current truth. See [docs/STATUS.md](../../STATUS.md).**


**Status**: ✅ COMPLETE
**Implementation Date**: 2026-01-28
**Version**: 1.0.0

---

## Overview

Phase 3 establishes unified task orchestration between the V3 workflow engine and swarm coordination system through bidirectional synchronization, consensus protocols, and comprehensive audit trails.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Workflow Engine                         │
│  (8 phases: intake → design → build → review → integrate)  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Phase Transitions
                     ↓
         ┌───────────────────────────┐
         │  workflow-swarm-sync.sh   │ ←── Hook
         │  • Sync phase transitions │
         │  • Record outputs         │
         │  • Update swarm state     │
         └───────────┬───────────────┘
                     │
                     ↓
         ┌───────────────────────────┐
         │  Orchestration Namespace  │ ←── V3 Memory
         │  • Transitions            │
         │  • Outputs                │
         │  • Consensus requests     │
         │  • Audit trail            │
         └───────────┬───────────────┘
                     │
                     ↑ Consensus Decisions
         ┌───────────────────────────┐
         │  swarm-consensus-sync.sh  │ ←── Hook
         │  • Handle consensus       │
         │  • Update workflow        │
         │  • Record decisions       │
         └───────────┬───────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Swarm Coordination                        │
│          (Byzantine consensus, agent coordination)          │
└─────────────────────────────────────────────────────────────┘
```

## Components Implemented

### 1. Hook Scripts

#### workflow-swarm-sync.sh
**Location**: `.claude/hooks/workflow-swarm-sync.sh`

**Purpose**: Synchronizes workflow phase transitions to swarm coordination state

**Commands**:
```bash
# Initialize orchestration namespace
./workflow-swarm-sync.sh init

# Sync phase transition
./workflow-swarm-sync.sh sync-phase <workflow_id> <phase_name> <status> [data]

# Record phase output
./workflow-swarm-sync.sh record-output <workflow_id> <phase_name> <output_type> [data]

# Get workflow state
./workflow-swarm-sync.sh get-state <workflow_id>

# List active workflows
./workflow-swarm-sync.sh list-workflows
```

**Features**:
- Phase transition tracking
- Output recording
- Swarm notification
- Audit trail creation
- State persistence

#### swarm-consensus-sync.sh
**Location**: `.claude/hooks/swarm-consensus-sync.sh`

**Purpose**: Syncs swarm consensus decisions back to workflow state

**Commands**:
```bash
# Handle workflow update
./swarm-consensus-sync.sh workflow-update <workflow_id> <phase_name> <status>

# Record consensus decision
./swarm-consensus-sync.sh record-decision <workflow_id> <phase_name> <decision> [data]

# Get consensus status
./swarm-consensus-sync.sh get-consensus <workflow_id> <phase_name>

# List pending consensus
./swarm-consensus-sync.sh list-pending
```

**Features**:
- Workflow update handling
- Consensus protocol triggering
- Decision recording
- Bidirectional sync
- Coordination events

### 2. Configuration Updates

#### workflow-engine.json
**Location**: `.claude/config/workflow-engine.json`

**New Sections Added**:

```json
{
  "memory_integration": {
    "namespaces": {
      "orchestration": "Swarm-workflow coordination and consensus state"
    }
  },

  "hooks": {
    "workflow_swarm_sync": {
      "path": ".claude/hooks/workflow-swarm-sync.sh",
      "enabled": true,
      "triggers": {
        "phase_start": true,
        "phase_complete": true,
        "phase_output": true
      }
    },
    "swarm_consensus_sync": {
      "path": ".claude/hooks/swarm-consensus-sync.sh",
      "enabled": true,
      "triggers": {
        "consensus_required": true,
        "consensus_decision": true,
        "workflow_update": true
      }
    }
  },

  "coordination": {
    "bidirectional_sync": {
      "enabled": true,
      "sync_interval_ms": 5000,
      "conflict_resolution": "workflow_precedence"
    },
    "consensus_protocol": {
      "enabled": true,
      "required_phases": ["design", "review", "integrate", "deploy"],
      "approval_threshold": 1,
      "timeout_ms": 300000
    },
    "audit_trail": {
      "enabled": true,
      "track_events": [
        "phase_transition",
        "consensus_decision",
        "workflow_update",
        "coordination_event"
      ],
      "retention_days": 90
    }
  }
}
```

### 3. Verification Script

#### verify-phase3-integration.sh
**Location**: `.claude/scripts/verify-phase3-integration.sh`

**Tests**:
1. Hook file existence and executability
2. Orchestration namespace initialization
3. Workflow phase synchronization
4. Phase output recording
5. Workflow state retrieval
6. Consensus decision recording
7. Memory entry verification
8. Hook help command functionality
9. Configuration validation

**Usage**:
```bash
bash .claude/scripts/verify-phase3-integration.sh
```

## Orchestration Namespace

### Memory Structure

The `orchestration` namespace stores:

#### Phase Transitions
```json
{
  "key": "transition_<workflow_id>_<phase>_<timestamp>",
  "value": {
    "workflow_id": "wf-123",
    "phase_name": "design",
    "phase_status": "completed",
    "timestamp": "2026-01-28T12:00:00Z",
    "data": {}
  }
}
```

#### Phase Outputs
```json
{
  "key": "output_<workflow_id>_<phase>_<type>",
  "value": {
    "decision": "Use microservices",
    "rationale": "Scalability requirements"
  }
}
```

#### Consensus Requests
```json
{
  "key": "consensus_request_<workflow_id>_<phase>_<timestamp>",
  "value": {
    "workflow_id": "wf-123",
    "phase_name": "design",
    "status": "initiated",
    "initiated_at": "2026-01-28T12:00:00Z",
    "votes": {},
    "required_approvals": 1
  }
}
```

#### Consensus Decisions
```json
{
  "key": "consensus_decision_<workflow_id>_<phase>_<timestamp>",
  "value": {
    "workflow_id": "wf-123",
    "phase_name": "design",
    "decision": "approved",
    "decided_at": "2026-01-28T12:05:00Z",
    "data": {
      "approvers": ["agent1", "agent2"],
      "votes": 2
    }
  }
}
```

#### Audit Trail
```json
{
  "key": "audit_<timestamp>_<event_type>",
  "value": {
    "event_type": "phase_transition",
    "timestamp": "2026-01-28T12:00:00Z",
    "data": {
      "workflow_id": "wf-123",
      "phase": "design",
      "status": "completed"
    }
  }
}
```

## Integration Flow

### Workflow → Swarm Flow

1. **Phase Transition Occurs** (workflow engine)
2. **workflow-swarm-sync.sh triggered**
   - Records transition in orchestration namespace
   - Updates current workflow state
   - Notifies swarm coordination layer
   - Creates audit trail entry
3. **swarm-consensus-sync.sh receives notification**
   - Stores swarm awareness of workflow state
   - Triggers consensus if required
   - Creates coordination event

### Swarm → Workflow Flow

1. **Consensus Decision Made** (swarm coordination)
2. **swarm-consensus-sync.sh triggered**
   - Records consensus decision
   - Determines new workflow status
   - Calls workflow-swarm-sync.sh
3. **workflow-swarm-sync.sh updates workflow**
   - Updates phase status with consensus decision
   - Records output if applicable
   - Creates audit trail entry

## Bidirectional Sync Features

### Conflict Resolution
- **Strategy**: `workflow_precedence`
- **Rule**: Workflow engine decisions take precedence over swarm suggestions
- **Mechanism**: Consensus used for validation, not override

### Sync Interval
- **Default**: 5000ms (5 seconds)
- **Configurable**: `coordination.bidirectional_sync.sync_interval_ms`

### Consensus Protocol
- **Required Phases**: design, review, integrate, deploy
- **Approval Threshold**: 1 (configurable)
- **Timeout**: 300000ms (5 minutes)

## Audit Trail

### Tracked Events
- `phase_transition` - Workflow phase changes
- `consensus_decision` - Consensus protocol outcomes
- `workflow_update` - Workflow state updates from consensus
- `coordination_event` - Inter-layer coordination activities

### Retention
- **Period**: 90 days
- **Storage**: Orchestration namespace
- **Format**: JSON with timestamps

## Usage Examples

### Example 1: Sync Design Phase Completion

```bash
# Workflow engine completes design phase
bash .claude/hooks/workflow-swarm-sync.sh sync-phase \
  "wf-feature-auth" "design" "completed" \
  '{"adr_created":true,"topology_selected":"hierarchical"}'

# This triggers:
# 1. Transition recorded in orchestration namespace
# 2. Swarm notified via swarm-consensus-sync.sh
# 3. Audit trail entry created
```

### Example 2: Consensus Approval

```bash
# Swarm reaches consensus on design
bash .claude/hooks/swarm-consensus-sync.sh record-decision \
  "wf-feature-auth" "design" "approved" \
  '{"approvers":["architect","security-reviewer"],"votes":2}'

# This triggers:
# 1. Decision recorded in orchestration namespace
# 2. Workflow updated to "approved" status
# 3. Coordination event created
```

### Example 3: Check Workflow State

```bash
# Get current workflow state
bash .claude/hooks/workflow-swarm-sync.sh get-state "wf-feature-auth"

# Returns:
# {
#   "current_phase": "design",
#   "status": "approved",
#   "updated_at": "2026-01-28T12:05:00Z"
# }
```

### Example 4: List Pending Consensus

```bash
# List all pending consensus requests
bash .claude/hooks/swarm-consensus-sync.sh list-pending

# Returns all consensus requests with status "initiated"
```

## Verification

### Run Verification Script

```bash
bash .claude/scripts/verify-phase3-integration.sh
```

### Expected Output

```
═══════════════════════════════════════════════════════
  Phase 3: Swarm-Workflow Coordination Verification
═══════════════════════════════════════════════════════

Checking hook files...
✓ PASS workflow-swarm-sync.sh exists
✓ PASS swarm-consensus-sync.sh exists
✓ PASS workflow-swarm-sync.sh is executable
✓ PASS swarm-consensus-sync.sh is executable

Initializing orchestration namespace...
✓ Orchestration namespace initialized

Testing workflow phase synchronization...
✓ Phase transition synced

Testing phase output recording...
✓ Phase output recorded

Testing consensus decision recording...
✓ Consensus decision recorded

Verifying memory entries...
✓ Memory entries found: X

═══════════════════════════════════════════════════════
  Test Summary
═══════════════════════════════════════════════════════
Total Tests: 13
Passed: 13
Failed: 0

✓ Phase 3 integration verification PASSED
```

### Manual Verification

```bash
# 1. List orchestration namespace entries
npx @claude-flow/cli@latest memory list --namespace orchestration

# 2. Check workflow configuration
cat .claude/config/workflow-engine.json | jq '.coordination'

# 3. Test dry-run workflow (if workflow CLI available)
npx @claude-flow/cli@latest workflow execute feature --task "Test" --dry-run
```

## Benefits

### 1. Unified Orchestration
- Single source of truth for workflow and swarm state
- Seamless coordination between layers
- Prevents state divergence

### 2. Consensus-Driven Quality Gates
- Design decisions validated by consensus
- Review outcomes confirmed by multiple reviewers
- Deployment approved through consensus protocol

### 3. Comprehensive Audit Trail
- Every coordination event tracked
- 90-day retention for compliance
- Detailed decision history

### 4. Bidirectional Communication
- Workflow informs swarm of progress
- Swarm validates workflow decisions
- Both layers stay synchronized

### 5. Extensibility
- Hook-based architecture allows custom extensions
- Pluggable consensus protocols
- Configurable sync strategies

## Next Steps

### Phase 4: Real-Time Coordination UI
- Visual workflow state dashboard
- Live consensus voting interface
- Agent activity monitoring

### Phase 5: Advanced Consensus
- Multi-strategy consensus (Byzantine, Raft, PBFT)
- Weighted voting based on agent expertise
- Automatic conflict resolution

### Phase 6: Predictive Orchestration
- ML-based phase duration prediction
- Intelligent agent assignment
- Proactive bottleneck detection

## Troubleshooting

### Hook Execution Fails

```bash
# Check hook permissions
ls -la .claude/hooks/*.sh

# Make executable
chmod +x .claude/hooks/*.sh
```

### Memory Operations Fail

```bash
# Initialize memory database
npx @claude-flow/cli@latest memory init --force

# Verify daemon is running
npx @claude-flow/cli@latest daemon status
```

### Consensus Not Triggering

```bash
# Check consensus configuration
cat .claude/config/workflow-engine.json | jq '.coordination.consensus_protocol'

# Verify required phases
cat .claude/config/workflow-engine.json | jq '.coordination.consensus_protocol.required_phases'
```

## References

- [Phase 2: V3 Memory Integration](./PHASE2_V3_MEMORY_INTEGRATION.md)
- [Workflow Engine Configuration](../.claude/config/workflow-engine.json)
- [Claude Flow V3 Documentation](../CLAUDE.md)

---

**Implementation Complete** ✅
Phase 3 establishes production-ready swarm-workflow coordination with bidirectional sync, consensus protocols, and comprehensive audit trails.
