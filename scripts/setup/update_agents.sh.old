#!/bin/bash

# Script to update all 7 agent repositories
# This ensures Claude Code always has the latest version of all agents

declare -A REPOS=(
    ["claude-code-sub-agents"]="https://github.com/dl-ezo/claude-code-sub-agents.git"
    ["awesome-claude-code-agents"]="https://github.com/hesreallyhim/awesome-claude-code-agents.git"
    ["wshobson-agents"]="https://github.com/wshobson/agents.git"
    ["voltagent-subagents"]="https://github.com/VoltAgent/awesome-claude-code-subagents.git"
    ["furai-subagents"]="https://github.com/0xfurai/claude-code-subagents.git"
    ["lst97-subagents"]="https://github.com/lst97/claude-code-sub-agents.git"
    ["nuttall-agents"]="https://github.com/iannuttall/claude-agents.git"
)

echo "Updating all agent repositories..."

total_agents=0

for repo_name in "${!REPOS[@]}"; do
    repo_url="${REPOS[$repo_name]}"
    echo "Updating $repo_name..."
    
    if [ -d "$repo_name" ]; then
        cd "$repo_name"
        git fetch origin
        git pull origin main 2>/dev/null || git pull origin master 2>/dev/null
        echo "Successfully updated $repo_name"
        cd ..
    else
        echo "$repo_name directory not found. Cloning repository..."
        git clone "$repo_url" "$repo_name"
        echo "Successfully cloned $repo_name"
    fi
done

echo ""
echo "===== AGENT INVENTORY ====="

# claude-code-sub-agents (root level .md files)
if [ -d "claude-code-sub-agents" ]; then
    agents_count=$(ls -1 claude-code-sub-agents/*.md 2>/dev/null | grep -v README | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "claude-code-sub-agents: $agents_count agents"
fi

# awesome-claude-code-agents (agents/ subdirectory)
if [ -d "awesome-claude-code-agents/agents" ]; then
    agents_count=$(ls -1 awesome-claude-code-agents/agents/*.md 2>/dev/null | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "awesome-claude-code-agents: $agents_count agents"
fi

# wshobson-agents (root level .md files, excluding README)
if [ -d "wshobson-agents" ]; then
    agents_count=$(ls -1 wshobson-agents/*.md 2>/dev/null | grep -v README | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "wshobson-agents: $agents_count agents"
fi

# voltagent-subagents (categories/ subdirectory structure)
if [ -d "voltagent-subagents/categories" ]; then
    agents_count=$(find voltagent-subagents/categories -name "*.md" 2>/dev/null | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "voltagent-subagents: $agents_count agents"
fi

# furai-subagents (agents/ subdirectory)
if [ -d "furai-subagents/agents" ]; then
    agents_count=$(find furai-subagents/agents -name "*.md" 2>/dev/null | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "furai-subagents: $agents_count agents"
fi

# lst97-subagents (organized in multiple directories)
if [ -d "lst97-subagents" ]; then
    agents_count=$(find lst97-subagents -name "*.md" 2>/dev/null | grep -v README | grep -v CLAUDE.md | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "lst97-subagents: $agents_count agents"
fi

# nuttall-agents (agents/ subdirectory)
if [ -d "nuttall-agents/agents" ]; then
    agents_count=$(find nuttall-agents/agents -name "*.md" 2>/dev/null | wc -l)
    total_agents=$((total_agents + agents_count))
    echo "nuttall-agents: $agents_count agents"
fi

echo ""
echo "TOTAL AGENTS AVAILABLE: $total_agents"
echo ""
echo "All agent repositories updated successfully!"