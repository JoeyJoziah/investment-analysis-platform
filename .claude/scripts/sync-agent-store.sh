#!/bin/bash
# Sync Agent Store Script
# Scans all agent markdown files and registers them in the agent store
#
# Usage: ./sync-agent-store.sh [--dry-run]
#
# This script:
# 1. Scans .claude/agents/ for all .md files
# 2. Extracts agent name, model, and description from frontmatter
# 3. Generates a synchronized store.json

set -e

# Configuration
AGENTS_DIR=".claude/agents"
STORE_FILE=".claude-flow/agents/store.json"
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Ensure we're in the project root
if [[ ! -d "$AGENTS_DIR" ]]; then
  echo "Error: $AGENTS_DIR directory not found. Run from project root."
  exit 1
fi

# Ensure store directory exists
mkdir -p "$(dirname "$STORE_FILE")"

# Initialize counters
total_agents=0
by_model_opus=0
by_model_sonnet=0
by_model_haiku=0
by_domain_investment=0
by_domain_development=0
by_domain_infrastructure=0
by_domain_other=0

# Temporary file for agents object
TEMP_AGENTS=$(mktemp)
echo "{" > "$TEMP_AGENTS"
first_agent=true

# Function to extract frontmatter value
extract_value() {
  local file="$1"
  local key="$2"
  grep -m1 "^${key}:" "$file" 2>/dev/null | sed "s/^${key}:[[:space:]]*//" | tr -d '"'
}

# Scan agent files
echo "Scanning agents in $AGENTS_DIR..."

# Process root-level agents
for agent_file in "$AGENTS_DIR"/*.md; do
  if [[ ! -f "$agent_file" ]]; then
    continue
  fi

  # Skip non-agent files
  filename=$(basename "$agent_file")
  if [[ "$filename" =~ ^(README|MIGRATION|HEALTH_ASSESSMENT) ]]; then
    continue
  fi

  # Extract frontmatter values
  name=$(extract_value "$agent_file" "name")
  model=$(extract_value "$agent_file" "model")
  description=$(extract_value "$agent_file" "description")

  # Skip if no name found
  if [[ -z "$name" ]]; then
    [[ "$VERBOSE" == true ]] && echo "  Skipping $filename (no name in frontmatter)"
    continue
  fi

  # Default model if not specified
  if [[ -z "$model" ]]; then
    model="sonnet"
  fi

  # Determine domain from description or filename
  domain="other"
  if [[ "$description" =~ (investment|deal|underwriter|financial|portfolio|risk) ]]; then
    domain="investment"
    ((by_domain_investment++))
  elif [[ "$description" =~ (docker|deploy|infrastructure|kubernetes|ci/cd|monitoring) ]]; then
    domain="infrastructure"
    ((by_domain_infrastructure++))
  elif [[ "$description" =~ (code|development|api|test|frontend|backend) ]]; then
    domain="development"
    ((by_domain_development++))
  else
    ((by_domain_other++))
  fi

  # Count by model
  case "$model" in
    opus) ((by_model_opus++)) ;;
    sonnet) ((by_model_sonnet++)) ;;
    haiku) ((by_model_haiku++)) ;;
  esac

  # Generate agent ID
  agent_id="agent-$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"

  # Add comma if not first
  if [[ "$first_agent" == true ]]; then
    first_agent=false
  else
    echo "," >> "$TEMP_AGENTS"
  fi

  # Escape description for JSON
  escaped_desc=$(echo "$description" | sed 's/"/\\"/g' | tr -d '\n\r' | cut -c1-200)

  # Write agent entry
  cat >> "$TEMP_AGENTS" << EOF
  "$agent_id": {
    "agentId": "$agent_id",
    "agentType": "$name",
    "status": "available",
    "health": 1,
    "taskCount": 0,
    "config": {
      "provider": "anthropic",
      "file": "$agent_file"
    },
    "model": "$model",
    "modelRoutedBy": "registry",
    "domain": "$domain",
    "description": "$escaped_desc"
  }
EOF

  ((total_agents++))
  [[ "$VERBOSE" == true ]] && echo "  + $name ($model, $domain)"
done

# Process subdirectory agents
for subdir in "$AGENTS_DIR"/*/; do
  if [[ ! -d "$subdir" ]]; then
    continue
  fi

  subdir_name=$(basename "$subdir")

  for agent_file in "$subdir"*.md; do
    if [[ ! -f "$agent_file" ]]; then
      continue
    fi

    filename=$(basename "$agent_file")
    if [[ "$filename" =~ ^(README|HEALTH_ASSESSMENT) ]]; then
      continue
    fi

    name=$(extract_value "$agent_file" "name")
    model=$(extract_value "$agent_file" "model")
    description=$(extract_value "$agent_file" "description")

    if [[ -z "$name" ]]; then
      [[ "$VERBOSE" == true ]] && echo "  Skipping $subdir_name/$filename (no name)"
      continue
    fi

    if [[ -z "$model" ]]; then
      model="sonnet"
    fi

    # Subdirectory agents are typically specialized
    domain="$subdir_name"

    case "$model" in
      opus) ((by_model_opus++)) ;;
      sonnet) ((by_model_sonnet++)) ;;
      haiku) ((by_model_haiku++)) ;;
    esac

    agent_id="agent-$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"

    echo "," >> "$TEMP_AGENTS"

    escaped_desc=$(echo "$description" | sed 's/"/\\"/g' | tr -d '\n\r' | cut -c1-200)

    cat >> "$TEMP_AGENTS" << EOF
  "$agent_id": {
    "agentId": "$agent_id",
    "agentType": "$name",
    "status": "available",
    "health": 1,
    "taskCount": 0,
    "config": {
      "provider": "anthropic",
      "file": "$agent_file",
      "swarm": "$subdir_name"
    },
    "model": "$model",
    "modelRoutedBy": "registry",
    "domain": "$domain",
    "description": "$escaped_desc"
  }
EOF

    ((total_agents++))
    [[ "$VERBOSE" == true ]] && echo "  + $subdir_name/$name ($model)"
  done
done

echo "}" >> "$TEMP_AGENTS"

# Generate final store.json
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [[ "$DRY_RUN" == true ]]; then
  echo ""
  echo "=== DRY RUN - Would generate: ==="
  echo ""
  cat << EOF
{
  "agents": $(cat "$TEMP_AGENTS"),
  "version": "3.0.0",
  "lastSync": "$TIMESTAMP",
  "statistics": {
    "total": $total_agents,
    "byModel": {
      "opus": $by_model_opus,
      "sonnet": $by_model_sonnet,
      "haiku": $by_model_haiku
    },
    "byDomain": {
      "investment": $by_domain_investment,
      "development": $by_domain_development,
      "infrastructure": $by_domain_infrastructure,
      "other": $by_domain_other
    }
  }
}
EOF
  echo ""
  echo "=== Statistics ==="
else
  cat > "$STORE_FILE" << EOF
{
  "agents": $(cat "$TEMP_AGENTS"),
  "version": "3.0.0",
  "lastSync": "$TIMESTAMP",
  "statistics": {
    "total": $total_agents,
    "byModel": {
      "opus": $by_model_opus,
      "sonnet": $by_model_sonnet,
      "haiku": $by_model_haiku
    },
    "byDomain": {
      "investment": $by_domain_investment,
      "development": $by_domain_development,
      "infrastructure": $by_domain_infrastructure,
      "other": $by_domain_other
    }
  }
}
EOF
  echo ""
  echo "=== Agent Store Synchronized ==="
  echo "Output: $STORE_FILE"
fi

# Cleanup
rm -f "$TEMP_AGENTS"

# Print summary
echo ""
echo "Total agents: $total_agents"
echo ""
echo "By Model:"
echo "  - opus:   $by_model_opus"
echo "  - sonnet: $by_model_sonnet"
echo "  - haiku:  $by_model_haiku"
echo ""
echo "By Domain:"
echo "  - investment:     $by_domain_investment"
echo "  - development:    $by_domain_development"
echo "  - infrastructure: $by_domain_infrastructure"
echo "  - other:          $by_domain_other"
echo ""

if [[ "$DRY_RUN" == false ]]; then
  echo "Store synchronized successfully!"
fi
