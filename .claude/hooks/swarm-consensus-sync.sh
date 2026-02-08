#!/bin/bash
# Swarm-Consensus Synchronization Hook
# Phase 3: Swarm-Workflow Coordination
# Purpose: Sync swarm consensus decisions back to workflow state

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hook metadata
HOOK_NAME="swarm-consensus-sync"
HOOK_VERSION="1.0.0"
NAMESPACE="orchestration"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    case "$level" in
        INFO)
            echo -e "${BLUE}[${timestamp}] [${HOOK_NAME}] INFO:${NC} ${message}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] [${HOOK_NAME}] SUCCESS:${NC} ${message}"
            ;;
        WARNING)
            echo -e "${YELLOW}[${timestamp}] [${HOOK_NAME}] WARNING:${NC} ${message}"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] [${HOOK_NAME}] ERROR:${NC} ${message}"
            ;;
    esac
}

# Handle workflow update from workflow-swarm-sync
handle_workflow_update() {
    local workflow_id="$1"
    local phase_name="$2"
    local phase_status="$3"

    log INFO "Handling workflow update: ${workflow_id}/${phase_name} -> ${phase_status}"

    # Store swarm awareness of workflow state
    local swarm_state_key="swarm_workflow_${workflow_id}"
    local swarm_state=$(cat <<EOF
{
  "workflow_id": "${workflow_id}",
  "current_phase": "${phase_name}",
  "phase_status": "${phase_status}",
  "swarm_notified_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "consensus_required": $(if [[ "${phase_status}" == "pending_consensus" ]]; then echo "true"; else echo "false"; fi)
}
EOF
)

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${swarm_state_key}" \
        --value "${swarm_state}" \
        2>&1 || {
            log ERROR "Failed to store swarm workflow state"
            return 1
        }

    # If consensus required, trigger consensus protocol
    if [[ "${phase_status}" == "pending_consensus" ]]; then
        log INFO "Triggering consensus protocol for ${workflow_id}/${phase_name}..."
        trigger_consensus "${workflow_id}" "${phase_name}"
    fi

    log SUCCESS "Workflow update processed by swarm"
}

# Trigger consensus protocol
trigger_consensus() {
    local workflow_id="$1"
    local phase_name="$2"

    log INFO "Initiating consensus for ${workflow_id}/${phase_name}..."

    # Create consensus request
    local consensus_key="consensus_request_${workflow_id}_${phase_name}_$(date +%s)"
    local consensus_data=$(cat <<EOF
{
  "workflow_id": "${workflow_id}",
  "phase_name": "${phase_name}",
  "status": "initiated",
  "initiated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "votes": {},
  "required_approvals": 1
}
EOF
)

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${consensus_key}" \
        --value "${consensus_data}" \
        2>&1 || {
            log ERROR "Failed to create consensus request"
            return 1
        }

    log SUCCESS "Consensus initiated: ${consensus_key}"
}

# Record consensus decision
record_consensus_decision() {
    local workflow_id="$1"
    local phase_name="$2"
    local decision="$3"  # approved, rejected, needs_revision
    local decision_data="${4:-{}}"

    log INFO "Recording consensus decision: ${workflow_id}/${phase_name} -> ${decision}"

    # Store consensus decision
    local decision_key="consensus_decision_${workflow_id}_${phase_name}_$(date +%s)"
    local decision_record=$(cat <<EOF
{
  "workflow_id": "${workflow_id}",
  "phase_name": "${phase_name}",
  "decision": "${decision}",
  "decided_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "data": ${decision_data}
}
EOF
)

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${decision_key}" \
        --value "${decision_record}" \
        2>&1 || {
            log ERROR "Failed to record consensus decision"
            return 1
        }

    # Update workflow state with decision
    update_workflow_with_consensus "${workflow_id}" "${phase_name}" "${decision}"

    log SUCCESS "Consensus decision recorded: ${decision_key}"
}

# Update workflow state with consensus decision
update_workflow_with_consensus() {
    local workflow_id="$1"
    local phase_name="$2"
    local decision="$3"

    log INFO "Updating workflow with consensus decision: ${decision}"

    # Determine new phase status based on decision
    local new_status
    case "$decision" in
        approved)
            new_status="approved"
            ;;
        rejected)
            new_status="rejected"
            ;;
        needs_revision)
            new_status="needs_revision"
            ;;
        *)
            log ERROR "Unknown consensus decision: ${decision}"
            return 1
            ;;
    esac

    # Call workflow-swarm-sync to update workflow state
    if [[ -f "${SCRIPT_DIR}/workflow-swarm-sync.sh" ]]; then
        bash "${SCRIPT_DIR}/workflow-swarm-sync.sh" sync-phase \
            "${workflow_id}" "${phase_name}" "${new_status}" \
            "{\"consensus_decision\":\"${decision}\"}" || \
            log ERROR "Failed to update workflow state"
    else
        log WARNING "workflow-swarm-sync.sh not found, cannot update workflow"
    fi

    log SUCCESS "Workflow updated with consensus decision"
}

# Get consensus status
get_consensus_status() {
    local workflow_id="$1"
    local phase_name="$2"

    log INFO "Retrieving consensus status: ${workflow_id}/${phase_name}"

    npx @claude-flow/cli@latest memory search \
        --namespace "${NAMESPACE}" \
        --query "consensus_request_${workflow_id}_${phase_name}" \
        --limit 1 \
        2>&1 || {
            log WARNING "No consensus requests found"
            echo "{}"
            return 1
        }
}

# List pending consensus requests
list_pending_consensus() {
    log INFO "Listing pending consensus requests..."

    npx @claude-flow/cli@latest memory search \
        --namespace "${NAMESPACE}" \
        --query "consensus_request status:initiated" \
        --limit 50 \
        2>&1 || {
            log WARNING "Failed to list pending consensus"
            return 1
        }
}

# Create coordination event
create_coordination_event() {
    local event_type="$1"
    local event_data="$2"

    local event_key="coordination_event_$(date +%s)_${event_type}"

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${event_key}" \
        --value "{\"event_type\":\"${event_type}\",\"timestamp\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"data\":${event_data}}" \
        2>&1 || log WARNING "Failed to create coordination event"
}

# Main command dispatcher
main() {
    local command="${1:-help}"

    case "$command" in
        workflow-update)
            if [[ $# -lt 4 ]]; then
                log ERROR "Usage: $0 workflow-update <workflow_id> <phase_name> <phase_status>"
                exit 1
            fi
            handle_workflow_update "$2" "$3" "$4"
            create_coordination_event "workflow_update" "{\"workflow_id\":\"$2\",\"phase\":\"$3\",\"status\":\"$4\"}"
            ;;
        record-decision)
            if [[ $# -lt 4 ]]; then
                log ERROR "Usage: $0 record-decision <workflow_id> <phase_name> <decision> [data]"
                exit 1
            fi
            record_consensus_decision "$2" "$3" "$4" "${5:-{}}"
            create_coordination_event "consensus_decision" "{\"workflow_id\":\"$2\",\"phase\":\"$3\",\"decision\":\"$4\"}"
            ;;
        get-consensus)
            if [[ $# -lt 3 ]]; then
                log ERROR "Usage: $0 get-consensus <workflow_id> <phase_name>"
                exit 1
            fi
            get_consensus_status "$2" "$3"
            ;;
        list-pending)
            list_pending_consensus
            ;;
        help)
            cat <<EOF
${HOOK_NAME} v${HOOK_VERSION}
Swarm-Consensus Synchronization Hook

Usage: $0 <command> [arguments]

Commands:
  workflow-update <wf_id> <phase> <status>      Handle workflow update from workflow-swarm-sync
  record-decision <wf_id> <phase> <decision> [data]  Record consensus decision
  get-consensus <workflow_id> <phase_name>      Get consensus status
  list-pending                                  List pending consensus requests
  help                                          Show this help

Examples:
  $0 workflow-update wf-123 design pending_consensus
  $0 record-decision wf-123 design approved '{"approvers":["agent1","agent2"]}'
  $0 get-consensus wf-123 design
  $0 list-pending
EOF
            ;;
        *)
            log ERROR "Unknown command: $command"
            log INFO "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
