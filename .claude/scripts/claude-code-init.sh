#!/bin/bash
# claude-code-init.sh
# Syncs everything-claude-code components while preserving custom investment-analysis configs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CLAUDE_DIR="$PROJECT_ROOT/.claude"
TEMP_DIR="/tmp/everything-claude-code"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Claude Code Integration Sync Script  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Custom files to preserve (will not be overwritten)
CUSTOM_AGENTS=(
    "investment-analyst.md"
    "deal-underwriter.md"
    "financial-modeler.md"
    "architecture-reviewer.md"
    "backend-api-swarm.md"
    "code-review-expert.md"
    "data-ml-pipeline-swarm.md"
    "data-science-architect.md"
    "financial-analysis-swarm.md"
    "godmode-refactorer.md"
    "infrastructure-devops-swarm.md"
    "project-quality-swarm.md"
    "security-compliance-swarm.md"
    "team-coordinator.md"
    "ui-visualization-swarm.md"
)

CUSTOM_SKILLS=(
    "financial-modeling"
    "deal-structuring"
    "underwriting-analysis"
    "api-rate-limiter"
    "cost-monitor"
    "sec-compliance"
    "stock-analysis"
)

CUSTOM_COMMANDS=(
    "underwrite.md"
    "model.md"
    "analyze-structure.md"
    "scenario.md"
    "analyze_codebase.md"
    "ui_design.md"
)

# Clone or update the repository
echo -e "${YELLOW}Fetching latest everything-claude-code...${NC}"
if [ -d "$TEMP_DIR" ]; then
    cd "$TEMP_DIR"
    git fetch origin
    git reset --hard origin/main
else
    git clone https://github.com/affaan-m/everything-claude-code.git "$TEMP_DIR"
fi

# Function to check if file is custom
is_custom_agent() {
    local file="$1"
    for custom in "${CUSTOM_AGENTS[@]}"; do
        if [ "$file" == "$custom" ]; then
            return 0
        fi
    done
    return 1
}

is_custom_skill() {
    local dir="$1"
    for custom in "${CUSTOM_SKILLS[@]}"; do
        if [ "$dir" == "$custom" ]; then
            return 0
        fi
    done
    return 1
}

is_custom_command() {
    local file="$1"
    for custom in "${CUSTOM_COMMANDS[@]}"; do
        if [ "$file" == "$custom" ]; then
            return 0
        fi
    done
    return 1
}

# Sync agents (skip custom)
echo -e "${YELLOW}Syncing agents...${NC}"
for agent in "$TEMP_DIR"/agents/*.md; do
    filename=$(basename "$agent")
    if is_custom_agent "$filename"; then
        echo -e "  ${BLUE}Skipping custom:${NC} $filename"
    else
        cp "$agent" "$CLAUDE_DIR/agents/"
        echo -e "  ${GREEN}Updated:${NC} $filename"
    fi
done

# Sync skills (skip custom)
echo -e "${YELLOW}Syncing skills...${NC}"
for skill_dir in "$TEMP_DIR"/skills/*/; do
    dirname=$(basename "$skill_dir")
    if is_custom_skill "$dirname"; then
        echo -e "  ${BLUE}Skipping custom:${NC} $dirname"
    else
        cp -r "$skill_dir" "$CLAUDE_DIR/skills/"
        echo -e "  ${GREEN}Updated:${NC} $dirname"
    fi
done

# Sync commands (skip custom)
echo -e "${YELLOW}Syncing commands...${NC}"
for command in "$TEMP_DIR"/commands/*.md; do
    filename=$(basename "$command")
    if is_custom_command "$filename"; then
        echo -e "  ${BLUE}Skipping custom:${NC} $filename"
    else
        cp "$command" "$CLAUDE_DIR/commands/"
        echo -e "  ${GREEN}Updated:${NC} $filename"
    fi
done

# Sync rules (always update)
echo -e "${YELLOW}Syncing rules...${NC}"
cp "$TEMP_DIR"/rules/*.md "$CLAUDE_DIR/rules/"
echo -e "  ${GREEN}Updated all rules${NC}"

# Sync hooks (always update)
echo -e "${YELLOW}Syncing hooks...${NC}"
cp -r "$TEMP_DIR"/hooks/* "$CLAUDE_DIR/hooks/"
echo -e "  ${GREEN}Updated all hooks${NC}"

# Sync contexts (always update)
echo -e "${YELLOW}Syncing contexts...${NC}"
cp "$TEMP_DIR"/contexts/*.md "$CLAUDE_DIR/contexts/"
echo -e "  ${GREEN}Updated all contexts${NC}"

# Sync MCP configs (always update)
echo -e "${YELLOW}Syncing MCP configs...${NC}"
cp "$TEMP_DIR"/mcp-configs/*.json "$CLAUDE_DIR/mcp/"
echo -e "  ${GREEN}Updated MCP configs${NC}"

# Update manifest timestamp
echo -e "${YELLOW}Updating integration manifest...${NC}"
if [ -f "$CLAUDE_DIR/integration-manifest.json" ]; then
    # Use sed to update the created_at field (cross-platform)
    today=$(date +%Y-%m-%d)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/\"created_at\": \"[^\"]*\"/\"created_at\": \"$today\"/" "$CLAUDE_DIR/integration-manifest.json"
    else
        sed -i "s/\"created_at\": \"[^\"]*\"/\"created_at\": \"$today\"/" "$CLAUDE_DIR/integration-manifest.json"
    fi
    echo -e "  ${GREEN}Updated manifest date to $today${NC}"
fi

# Cleanup
echo -e "${YELLOW}Cleaning up...${NC}"
rm -rf "$TEMP_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Integration sync complete!           ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Custom components preserved:"
echo -e "  - Investment analysis agents"
echo -e "  - Investment-specific skills"
echo -e "  - Investment-specific commands"
echo -e "  - Project-specific swarm agents"
echo ""
echo -e "Run ${BLUE}cat .claude/integration-manifest.json${NC} to see full component list."
