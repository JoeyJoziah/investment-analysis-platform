#!/bin/bash
# GitHub Swarm Label Initialization Script
# Creates required labels for the investment-analysis-platform repository

set -e

REPO="JoeyJoziah/investment-analysis-platform"

echo "Initializing GitHub labels for $REPO..."
echo ""

# Type labels
echo "Creating type labels..."
gh label create "bug" --color "d73a4a" --description "Something isn't working" --repo "$REPO" --force 2>/dev/null || echo "  bug: exists or error"
gh label create "feature" --color "0075ca" --description "New feature or request" --repo "$REPO" --force 2>/dev/null || echo "  feature: exists or error"
gh label create "enhancement" --color "a2eeef" --description "Improvement to existing feature" --repo "$REPO" --force 2>/dev/null || echo "  enhancement: exists or error"
gh label create "documentation" --color "0052cc" --description "Documentation updates" --repo "$REPO" --force 2>/dev/null || echo "  documentation: exists or error"
gh label create "infrastructure" --color "5319e7" --description "Docker, CI/CD, deployment" --repo "$REPO" --force 2>/dev/null || echo "  infrastructure: exists or error"
gh label create "security" --color "ff0000" --description "Security vulnerability or concern" --repo "$REPO" --force 2>/dev/null || echo "  security: exists or error"
gh label create "performance" --color "fbca04" --description "Performance optimization" --repo "$REPO" --force 2>/dev/null || echo "  performance: exists or error"
gh label create "testing" --color "1d76db" --description "Testing improvements" --repo "$REPO" --force 2>/dev/null || echo "  testing: exists or error"

# Component labels
echo ""
echo "Creating component labels..."
gh label create "backend" --color "c5def5" --description "FastAPI backend" --repo "$REPO" --force 2>/dev/null || echo "  backend: exists or error"
gh label create "frontend" --color "bfdadc" --description "React frontend" --repo "$REPO" --force 2>/dev/null || echo "  frontend: exists or error"
gh label create "ml-models" --color "d4c5f9" --description "ML/AI models (Prophet, XGBoost)" --repo "$REPO" --force 2>/dev/null || echo "  ml-models: exists or error"
gh label create "database" --color "f9d0c4" --description "PostgreSQL/TimescaleDB" --repo "$REPO" --force 2>/dev/null || echo "  database: exists or error"
gh label create "data-pipeline" --color "e99695" --description "ETL and data processing" --repo "$REPO" --force 2>/dev/null || echo "  data-pipeline: exists or error"
gh label create "api" --color "c2e0c6" --description "API endpoints" --repo "$REPO" --force 2>/dev/null || echo "  api: exists or error"

# Priority labels
echo ""
echo "Creating priority labels..."
gh label create "P0-critical" --color "b60205" --description "Critical - immediate action required" --repo "$REPO" --force 2>/dev/null || echo "  P0-critical: exists or error"
gh label create "P1-high" --color "d93f0b" --description "High priority" --repo "$REPO" --force 2>/dev/null || echo "  P1-high: exists or error"
gh label create "P2-medium" --color "fbca04" --description "Medium priority" --repo "$REPO" --force 2>/dev/null || echo "  P2-medium: exists or error"
gh label create "P3-low" --color "0e8a16" --description "Low priority" --repo "$REPO" --force 2>/dev/null || echo "  P3-low: exists or error"

# Status labels
echo ""
echo "Creating status labels..."
gh label create "needs-triage" --color "ededed" --description "Needs triage by maintainer" --repo "$REPO" --force 2>/dev/null || echo "  needs-triage: exists or error"
gh label create "in-progress" --color "fbca04" --description "Work in progress" --repo "$REPO" --force 2>/dev/null || echo "  in-progress: exists or error"
gh label create "ready-for-review" --color "0e8a16" --description "Ready for code review" --repo "$REPO" --force 2>/dev/null || echo "  ready-for-review: exists or error"
gh label create "blocked" --color "d93f0b" --description "Blocked by dependency" --repo "$REPO" --force 2>/dev/null || echo "  blocked: exists or error"

# Swarm-specific labels
echo ""
echo "Creating swarm-specific labels..."
gh label create "swarm-triaged" --color "bfe5bf" --description "Triaged by GitHub Swarm" --repo "$REPO" --force 2>/dev/null || echo "  swarm-triaged: exists or error"
gh label create "swarm-reviewed" --color "bfe5bf" --description "Reviewed by GitHub Swarm" --repo "$REPO" --force 2>/dev/null || echo "  swarm-reviewed: exists or error"

echo ""
echo "Label initialization complete!"
echo ""
echo "Labels created for: $REPO"
echo ""
echo "To verify labels, run:"
echo "  gh label list --repo $REPO"
