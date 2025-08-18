#!/bin/bash

# Script to set up global access to Claude Code agents
# This creates symlinks and configuration for system-wide agent access

CLAUDE_CONFIG_DIR="$HOME/.config/claude-code"
CLAUDE_AGENTS_DIR="$CLAUDE_CONFIG_DIR/agents"
CURRENT_DIR="$(pwd)"

echo "Setting up global Claude Code agent access..."

# Create Claude Code config directory if it doesn't exist
mkdir -p "$CLAUDE_AGENTS_DIR"

# Function to create symlinks for agent repositories
create_agent_symlinks() {
    local repo_name="$1"
    local source_path="$CURRENT_DIR/$repo_name"
    local target_path="$CLAUDE_AGENTS_DIR/$repo_name"
    
    if [ -d "$source_path" ]; then
        if [ -L "$target_path" ] || [ -e "$target_path" ]; then
            rm -rf "$target_path"
        fi
        ln -sf "$source_path" "$target_path"
        echo "Created symlink for $repo_name"
    else
        echo "Warning: $repo_name not found at $source_path"
    fi
}

# Create symlinks for all agent repositories
create_agent_symlinks "claude-code-sub-agents"
create_agent_symlinks "awesome-claude-code-agents"
create_agent_symlinks "wshobson-agents"
create_agent_symlinks "voltagent-subagents"
create_agent_symlinks "furai-subagents"
create_agent_symlinks "lst97-subagents"
create_agent_symlinks "nuttall-agents"

# Create a global update script
cat > "$CLAUDE_AGENTS_DIR/update_global_agents.sh" << 'EOF'
#!/bin/bash

# Global agent update script
# Run this from any directory to update all Claude Code agents

SOURCE_DIR="/mnt/c/Users/Devin McGrathj/01.project_files/investment_analysis_app"

if [ -d "$SOURCE_DIR" ]; then
    cd "$SOURCE_DIR"
    ./update_agents.sh
else
    echo "Error: Source directory not found at $SOURCE_DIR"
    echo "Please update the SOURCE_DIR variable in this script to match your installation path"
    exit 1
fi
EOF

chmod +x "$CLAUDE_AGENTS_DIR/update_global_agents.sh"

# Create a CLAUDE.md file in the global agents directory
cat > "$CLAUDE_AGENTS_DIR/CLAUDE.md" << EOF
# Global Claude Code Agents

This directory contains symlinks to all Claude Code agent repositories.

## Available Agent Collections

- **claude-code-sub-agents/** - Core development agents
- **awesome-claude-code-agents/** - Specialized backend/frontend agents
- **wshobson-agents/** - Professional development agents (50+ agents)
- **voltagent-subagents/** - Enterprise development patterns (100+ agents)
- **furai-subagents/** - Advanced code analysis agents
- **lst97-subagents/** - Lifecycle coverage agents (organized by domain)
- **nuttall-agents/** - Modern development practice agents

## Usage

These agents are automatically available to Claude Code in any project.

## Updating Agents

To update all agents globally, run:
\`\`\`bash
~/.config/claude-code/agents/update_global_agents.sh
\`\`\`

Or from the original project directory:
\`\`\`bash
./update_agents.sh
\`\`\`

## Agent Locations

All agents are stored in: ~/.config/claude-code/agents/
These are symlinks to: $(pwd)
EOF

# Add to shell profile for easy access
SHELL_PROFILE=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
fi

if [ -n "$SHELL_PROFILE" ]; then
    if ! grep -q "CLAUDE_AGENTS_PATH" "$SHELL_PROFILE"; then
        echo "" >> "$SHELL_PROFILE"
        echo "# Claude Code Agents" >> "$SHELL_PROFILE"
        echo "export CLAUDE_AGENTS_PATH=\"$CLAUDE_AGENTS_DIR\"" >> "$SHELL_PROFILE"
        echo "alias update-claude-agents=\"\$CLAUDE_AGENTS_PATH/update_global_agents.sh\"" >> "$SHELL_PROFILE"
        echo "Added Claude Code agent environment variables to $SHELL_PROFILE"
    fi
fi

echo ""
echo "Global Claude Code agent setup complete!"
echo ""
echo "Agent collections are now available at: $CLAUDE_AGENTS_DIR"
echo "To update agents from anywhere, run: update-claude-agents"
echo "Or use: ~/.config/claude-code/agents/update_global_agents.sh"
echo ""
echo "Restart your shell or run 'source $SHELL_PROFILE' to use the new alias."