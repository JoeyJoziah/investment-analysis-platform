#!/bin/bash
# GitHub Projects Board Sync Script
# Investment Analysis Platform
#
# Syncs tasks between TODO.md, GitHub Issues, and GitHub Projects
# Usage: ./scripts/github-board-sync.sh [command] [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_OWNER="${GITHUB_REPOSITORY_OWNER:-JoeyJoziah}"
REPO_NAME="${GITHUB_REPOSITORY_NAME:-investment-analysis-platform}"
PROJECT_TITLE="Investment Analysis Platform"
CONFIG_FILE=".github/board-sync.yml"
STATE_FILE=".github_board_sync_state.json"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed"
        echo "Install with: brew install gh"
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        log_error "Not authenticated with GitHub CLI"
        echo "Run: gh auth login"
        exit 1
    fi

    # Check for project scope
    local scopes
    scopes=$(gh auth status 2>&1 | grep "Token scopes" || true)
    if [[ ! "$scopes" =~ "project" ]]; then
        log_warn "Token may not have 'project' scope. You might need to re-authenticate."
        echo "Run: gh auth refresh -s project"
    fi

    log_success "Prerequisites check passed"
}

# Get or create project
get_or_create_project() {
    log_info "Looking for project: $PROJECT_TITLE"

    # Try to find existing project
    local project_id
    project_id=$(gh project list --owner "$REPO_OWNER" --format json 2>/dev/null | \
        jq -r ".projects[] | select(.title == \"$PROJECT_TITLE\") | .id" 2>/dev/null || echo "")

    if [[ -z "$project_id" || "$project_id" == "null" ]]; then
        log_info "Project not found, creating new project..."

        # Create project using gh CLI
        project_id=$(gh project create --owner "$REPO_OWNER" --title "$PROJECT_TITLE" --format json 2>/dev/null | \
            jq -r '.id' || echo "")

        if [[ -z "$project_id" || "$project_id" == "null" ]]; then
            log_error "Failed to create project. Ensure you have the 'project' scope."
            echo "Run: gh auth refresh -s project"
            exit 1
        fi

        log_success "Created project: $PROJECT_TITLE (ID: $project_id)"

        # Create default fields
        create_project_fields "$project_id"
    else
        log_success "Found project: $PROJECT_TITLE (ID: $project_id)"
    fi

    echo "$project_id"
}

# Create project fields
create_project_fields() {
    local project_id=$1
    log_info "Creating project fields..."

    # Priority field
    gh project field-create "$project_id" --owner "$REPO_OWNER" \
        --name "Priority" \
        --data-type "SINGLE_SELECT" \
        --single-select-options "🔴 Critical,🟡 High,🟢 Medium,⚪ Low" 2>/dev/null || true

    # Agent Type field
    gh project field-create "$project_id" --owner "$REPO_OWNER" \
        --name "Agent Type" \
        --data-type "SINGLE_SELECT" \
        --single-select-options "infrastructure-devops-swarm,data-ml-pipeline-swarm,financial-analysis-swarm,backend-api-swarm,ui-visualization-swarm,project-quality-swarm,security-compliance-swarm" 2>/dev/null || true

    # Complexity field
    gh project field-create "$project_id" --owner "$REPO_OWNER" \
        --name "Complexity" \
        --data-type "SINGLE_SELECT" \
        --single-select-options "XS (< 1 hour),S (1-4 hours),M (1-2 days),L (3-5 days),XL (1+ week)" 2>/dev/null || true

    # Cost field
    gh project field-create "$project_id" --owner "$REPO_OWNER" \
        --name "Estimated Cost" \
        --data-type "NUMBER" 2>/dev/null || true

    log_success "Project fields created"
}

# Parse TODO.md and create issues
sync_todo_to_issues() {
    log_info "Syncing TODO.md to GitHub Issues..."

    local todo_file="TODO.md"
    if [[ ! -f "$todo_file" ]]; then
        log_error "TODO.md not found"
        exit 1
    fi

    # Parse tasks from TODO.md
    local tasks=()
    local in_task=false
    local current_section=""
    local current_task=""
    local task_status=""

    while IFS= read -r line; do
        # Detect section headers
        if [[ "$line" =~ ^##[[:space:]]+(HIGH|MEDIUM|LOW)[[:space:]]+PRIORITY ]]; then
            current_section=$(echo "$line" | grep -oE "(HIGH|MEDIUM|LOW)" | head -1)
        fi

        # Detect task items (### numbered items)
        if [[ "$line" =~ ^###[[:space:]]+[0-9]+\.[[:space:]]+ ]]; then
            # Check if task is complete (strikethrough)
            if [[ "$line" =~ ~~.*~~ ]]; then
                task_status="done"
            else
                task_status="todo"
            fi

            # Extract task name
            current_task=$(echo "$line" | sed 's/^### [0-9]*\. //' | sed 's/~~//g' | sed 's/ ✅ COMPLETE//')

            # Map priority
            local priority="Medium"
            case "$current_section" in
                HIGH) priority="High" ;;
                MEDIUM) priority="Medium" ;;
                LOW) priority="Low" ;;
            esac

            if [[ "$task_status" != "done" && -n "$current_task" ]]; then
                tasks+=("$current_task|$priority|$current_section")
            fi
        fi
    done < "$todo_file"

    # Create issues for pending tasks
    local created_count=0
    for task_info in "${tasks[@]}"; do
        IFS='|' read -r task_name priority section <<< "$task_info"

        # Check if issue already exists
        local existing
        existing=$(gh issue list --repo "$REPO_OWNER/$REPO_NAME" --search "$task_name" --json title --jq '.[].title' 2>/dev/null | grep -F "$task_name" || true)

        if [[ -z "$existing" ]]; then
            log_info "Creating issue: $task_name"

            # Determine labels based on task name
            local labels="enhancement"
            if [[ "$task_name" =~ SSL|Certificate|Deploy ]]; then
                labels="$labels,infrastructure"
            elif [[ "$task_name" =~ Test|Testing ]]; then
                labels="$labels,testing"
            elif [[ "$task_name" =~ Performance ]]; then
                labels="$labels,performance"
            elif [[ "$task_name" =~ Frontend|UI ]]; then
                labels="$labels,frontend"
            elif [[ "$task_name" =~ Backend|API ]]; then
                labels="$labels,backend"
            fi

            # Map priority to label
            case "$priority" in
                High) labels="$labels,priority:high" ;;
                Medium) labels="$labels,priority:medium" ;;
                Low) labels="$labels,priority:low" ;;
            esac

            gh issue create \
                --repo "$REPO_OWNER/$REPO_NAME" \
                --title "$task_name" \
                --body "Auto-created from TODO.md

**Priority**: $priority
**Section**: $section Priority

---
_This issue was automatically created by github-board-sync.sh_" \
                --label "$labels" 2>/dev/null || log_warn "Failed to create issue: $task_name"

            ((created_count++)) || true
        else
            log_info "Issue already exists: $task_name"
        fi
    done

    log_success "Created $created_count new issues from TODO.md"
}

# Sync issues to project board
sync_issues_to_board() {
    local project_id=$1
    log_info "Syncing issues to project board..."

    # Get all open issues
    local issues
    issues=$(gh issue list --repo "$REPO_OWNER/$REPO_NAME" --state open --json number,title,url --limit 100 2>/dev/null)

    if [[ -z "$issues" || "$issues" == "[]" ]]; then
        log_warn "No open issues found"
        return
    fi

    local added_count=0
    echo "$issues" | jq -c '.[]' | while read -r issue; do
        local issue_url
        issue_url=$(echo "$issue" | jq -r '.url')
        local issue_title
        issue_title=$(echo "$issue" | jq -r '.title')

        # Add issue to project
        if gh project item-add "$project_id" --owner "$REPO_OWNER" --url "$issue_url" 2>/dev/null; then
            log_info "Added to board: $issue_title"
            ((added_count++)) || true
        fi
    done

    log_success "Synced issues to project board"
}

# Generate board status report
generate_report() {
    log_info "Generating board status report..."

    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  GitHub Projects Board Status Report${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""

    # Issue counts by state
    echo -e "${BLUE}Issue Summary:${NC}"
    local open_count
    open_count=$(gh issue list --repo "$REPO_OWNER/$REPO_NAME" --state open --json number --jq 'length' 2>/dev/null || echo "0")
    local closed_count
    closed_count=$(gh issue list --repo "$REPO_OWNER/$REPO_NAME" --state closed --json number --jq 'length' 2>/dev/null || echo "0")

    echo "  Open issues: $open_count"
    echo "  Closed issues: $closed_count"
    echo ""

    # Issues by label
    echo -e "${BLUE}Issues by Priority:${NC}"
    for priority in "priority:high" "priority:medium" "priority:low"; do
        local count
        count=$(gh issue list --repo "$REPO_OWNER/$REPO_NAME" --label "$priority" --json number --jq 'length' 2>/dev/null || echo "0")
        echo "  $priority: $count"
    done
    echo ""

    # Recent activity
    echo -e "${BLUE}Recent Issues (last 5):${NC}"
    gh issue list --repo "$REPO_OWNER/$REPO_NAME" --limit 5 --json number,title,state \
        --template '{{range .}}  #{{.number}}: {{.title}} ({{.state}}){{"\n"}}{{end}}' 2>/dev/null || true
    echo ""

    # PR status
    echo -e "${BLUE}Open Pull Requests:${NC}"
    local pr_count
    pr_count=$(gh pr list --repo "$REPO_OWNER/$REPO_NAME" --state open --json number --jq 'length' 2>/dev/null || echo "0")
    echo "  Open PRs: $pr_count"

    gh pr list --repo "$REPO_OWNER/$REPO_NAME" --limit 5 --json number,title,state \
        --template '{{range .}}  #{{.number}}: {{.title}}{{"\n"}}{{end}}' 2>/dev/null || true
    echo ""

    echo -e "${CYAN}========================================${NC}"
}

# List project items
list_board() {
    local project_id=$1
    log_info "Listing project board items..."

    gh project item-list "$project_id" --owner "$REPO_OWNER" --format json 2>/dev/null | \
        jq -r '.items[] | "\(.content.type // "Draft"): \(.content.title // .title) [\(.status // "No Status")]"' 2>/dev/null || \
        log_warn "Could not list project items"
}

# Show help
show_help() {
    echo "GitHub Projects Board Sync"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  init          Initialize project board and create fields"
    echo "  sync          Full sync: TODO.md -> Issues -> Board"
    echo "  sync-todo     Sync TODO.md to GitHub Issues"
    echo "  sync-board    Sync Issues to Project Board"
    echo "  list          List project board items"
    echo "  report        Generate status report"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 init                  # Create project board"
    echo "  $0 sync                  # Full synchronization"
    echo "  $0 report                # Show status report"
    echo ""
    echo "Configuration:"
    echo "  Config file: $CONFIG_FILE"
    echo "  State file: $STATE_FILE"
}

# Main function
main() {
    local command="${1:-help}"

    case "$command" in
        init)
            check_prerequisites
            get_or_create_project
            ;;
        sync)
            check_prerequisites
            local project_id
            project_id=$(get_or_create_project)
            sync_todo_to_issues
            sync_issues_to_board "$project_id"
            generate_report
            ;;
        sync-todo)
            check_prerequisites
            sync_todo_to_issues
            ;;
        sync-board)
            check_prerequisites
            local project_id
            project_id=$(get_or_create_project)
            sync_issues_to_board "$project_id"
            ;;
        list)
            check_prerequisites
            local project_id
            project_id=$(get_or_create_project)
            list_board "$project_id"
            ;;
        report)
            check_prerequisites
            generate_report
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
