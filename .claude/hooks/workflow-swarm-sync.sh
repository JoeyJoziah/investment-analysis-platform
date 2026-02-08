#!/bin/bash
# Workflow-Swarm Synchronization Hook
# Phase 3: Swarm-Workflow Coordination
# Purpose: Sync workflow phase transitions to swarm coordination state

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Hook metadata
HOOK_NAME="workflow-swarm-sync"
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

# Initialize orchestration namespace
init_namespace() {
    log INFO "Initializing orchestration namespace..."

    # Store namespace metadata
    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "namespace_metadata" \
        --value "{\"created_at\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"version\":\"${HOOK_VERSION}\",\"purpose\":\"Swarm-Workflow Coordination\"}" \
        2>&1 || log WARNING "Failed to store namespace metadata"

    log SUCCESS "Orchestration namespace initialized"
}

# Sync workflow phase transition to swarm
sync_phase_transition() {
    local workflow_id="$1"
    local phase_name="$2"
    local phase_status="$3"
    local phase_data="${4:-{}}"

    log INFO "Syncing phase transition: ${workflow_id} -> ${phase_name} (${phase_status})"

    # Create transition event
    local transition_key="transition_${workflow_id}_${phase_name}_$(date +%s)"
    local transition_data=$(cat <<EOF
{
  "workflow_id": "${workflow_id}",
  "phase_name": "${phase_name}",
  "phase_status": "${phase_status}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "data": ${phase_data}
}
EOF
)

    # Store transition in orchestration namespace
    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${transition_key}" \
        --value "${transition_data}" \
        2>&1 || {
            log ERROR "Failed to store phase transition"
            return 1
        }

    # Update current workflow state
    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "current_workflow_${workflow_id}" \
        --value "{\"current_phase\":\"${phase_name}\",\"status\":\"${phase_status}\",\"updated_at\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}" \
        2>&1 || log WARNING "Failed to update current workflow state"

    # Notify swarm coordination layer
    if [[ -f "${SCRIPT_DIR}/swarm-consensus-sync.sh" ]]; then
        log INFO "Notifying swarm coordination layer..."
        bash "${SCRIPT_DIR}/swarm-consensus-sync.sh" workflow-update "${workflow_id}" "${phase_name}" "${phase_status}" || \
            log WARNING "Swarm notification failed (non-blocking)"
    fi

    log SUCCESS "Phase transition synced: ${transition_key}"
}

# Record phase output
record_phase_output() {
    local workflow_id="$1"
    local phase_name="$2"
    local output_type="$3"
    local output_data="${4:-{}}"

    log INFO "Recording phase output: ${workflow_id}/${phase_name}/${output_type}"

    local output_key="output_${workflow_id}_${phase_name}_${output_type}"

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${output_key}" \
        --value "${output_data}" \
        2>&1 || {
            log ERROR "Failed to record phase output"
            return 1
        }

    log SUCCESS "Phase output recorded: ${output_key}"
}

# Get workflow state
get_workflow_state() {
    local workflow_id="$1"

    log INFO "Retrieving workflow state: ${workflow_id}"

    npx @claude-flow/cli@latest memory retrieve \
        --namespace "${NAMESPACE}" \
        --key "current_workflow_${workflow_id}" \
        2>&1 || {
            log WARNING "Workflow state not found"
            echo "{}"
            return 1
        }
}

# List all active workflows
list_active_workflows() {
    log INFO "Listing active workflows..."

    npx @claude-flow/cli@latest memory search \
        --namespace "${NAMESPACE}" \
        --query "current_workflow" \
        --limit 50 \
        2>&1 || {
            log WARNING "Failed to list workflows"
            return 1
        }
}

# Create audit trail entry
create_audit_entry() {
    local event_type="$1"
    local event_data="$2"

    local audit_key="audit_$(date +%s)_${event_type}"

    npx @claude-flow/cli@latest memory store \
        --namespace "${NAMESPACE}" \
        --key "${audit_key}" \
        --value "{\"event_type\":\"${event_type}\",\"timestamp\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"data\":${event_data}}" \
        2>&1 || log WARNING "Failed to create audit entry"
}

# Main command dispatcher
main() {
    local command="${1:-help}"

    case "$command" in
        init)
            init_namespace
            ;;
        sync-phase)
            if [[ $# -lt 4 ]]; then
                log ERROR "Usage: $0 sync-phase <workflow_id> <phase_name> <phase_status> [phase_data]"
                exit 1
            fi
            sync_phase_transition "$2" "$3" "$4" "${5:-{}}"
            create_audit_entry "phase_transition" "{\"workflow_id\":\"$2\",\"phase\":\"$3\",\"status\":\"$4\"}"
            ;;
        record-output)
            if [[ $# -lt 4 ]]; then
                log ERROR "Usage: $0 record-output <workflow_id> <phase_name> <output_type> [output_data]"
                exit 1
            fi
            record_phase_output "$2" "$3" "$4" "${5:-{}}"
            create_audit_entry "phase_output" "{\"workflow_id\":\"$2\",\"phase\":\"$3\",\"output_type\":\"$4\"}"
            ;;
        get-state)
            if [[ $# -lt 2 ]]; then
                log ERROR "Usage: $0 get-state <workflow_id>"
                exit 1
            fi
            get_workflow_state "$2"
            ;;
        list-workflows)
            list_active_workflows
            ;;
        help)
            cat <<EOF
${HOOK_NAME} v${HOOK_VERSION}
Workflow-Swarm Synchronization Hook

Usage: $0 <command> [arguments]

Commands:
  init                                           Initialize orchestration namespace
  sync-phase <wf_id> <phase> <status> [data]   Sync phase transition to swarm
  record-output <wf_id> <phase> <type> [data]  Record phase output
  get-state <workflow_id>                       Get workflow state
  list-workflows                                List active workflows
  help                                          Show this help

Examples:
  $0 init
  $0 sync-phase wf-123 design completed '{"adr_created":true}'
  $0 record-output wf-123 design adr '{"decision":"Use microservices"}'
  $0 get-state wf-123
  $0 list-workflows
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
