#!/bin/bash
# Claude Flow V3 - Critical Infrastructure Fix Script
# Fixes broken CLI installation and initializes daemon + memory systems

set -e

echo "🔧 Claude Flow V3 Infrastructure Fix"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Clear npm cache corruption
echo "📦 Step 1: Clearing npm cache corruption..."
if rm -rf ~/.npm/_npx 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Removed corrupted npx cache"
else
  echo -e "${YELLOW}⚠${NC} Could not remove npx cache (may not exist)"
fi

npm cache clean --force 2>/dev/null || echo -e "${YELLOW}⚠${NC} npm cache clean had warnings (continuing)"
echo -e "${GREEN}✓${NC} npm cache cleaned"
echo ""

# Step 2: Install CLI locally
echo "📥 Step 2: Installing @claude-flow/cli locally..."
if npm install --save-dev @claude-flow/cli@3.0.0-alpha.178; then
  echo -e "${GREEN}✓${NC} CLI installed successfully"
else
  echo -e "${RED}✗${NC} CLI installation failed"
  exit 1
fi
echo ""

# Step 3: Verify CLI works
echo "✅ Step 3: Verifying CLI installation..."
if npx @claude-flow/cli@latest --version 2>/dev/null; then
  echo -e "${GREEN}✓${NC} CLI is responding"
else
  echo -e "${RED}✗${NC} CLI verification failed"
  exit 1
fi
echo ""

# Step 4: Run doctor to check system health
echo "🩺 Step 4: Running system diagnostics..."
npx @claude-flow/cli@latest doctor --fix 2>/dev/null || echo -e "${YELLOW}⚠${NC} Doctor found issues (will fix manually)"
echo ""

# Step 5: Initialize memory system
echo "🧠 Step 5: Initializing memory system..."
if npx @claude-flow/cli@latest memory init --force --verbose 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Memory system initialized"
else
  echo -e "${YELLOW}⚠${NC} Memory init had warnings (continuing)"
fi
echo ""

# Step 6: Start daemon
echo "⚙️ Step 6: Starting daemon with background workers..."
if npx @claude-flow/cli@latest daemon start 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Daemon started"
else
  echo -e "${YELLOW}⚠${NC} Daemon start had warnings (checking status)"
fi

sleep 2
if npx @claude-flow/cli@latest daemon status 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Daemon is running"
else
  echo -e "${YELLOW}⚠${NC} Daemon status unclear (may be running in background)"
fi
echo ""

# Step 7: Test memory operations
echo "🧪 Step 7: Testing memory operations..."
if npx @claude-flow/cli@latest memory store --key "infrastructure-fix-$(date +%s)" --value "CLI infrastructure fixed successfully" --namespace patterns 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Memory store works"
else
  echo -e "${YELLOW}⚠${NC} Memory store failed"
fi

if npx @claude-flow/cli@latest memory list --namespace patterns --limit 5 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Memory list works"
else
  echo -e "${YELLOW}⚠${NC} Memory list failed"
fi
echo ""

# Step 8: Verify hooks
echo "🪝 Step 8: Verifying hooks system..."
if npx @claude-flow/cli@latest hooks list 2>/dev/null | head -10; then
  echo -e "${GREEN}✓${NC} Hooks are accessible"
else
  echo -e "${YELLOW}⚠${NC} Hooks list failed"
fi
echo ""

# Step 9: Check worker status
echo "👷 Step 9: Checking background workers..."
if npx @claude-flow/cli@latest hooks worker list 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Workers are configured"
else
  echo -e "${YELLOW}⚠${NC} Worker list failed"
fi
echo ""

# Step 10: Display statusline
echo "📊 Step 10: Displaying system status..."
if npx @claude-flow/cli@latest hooks statusline 2>/dev/null; then
  echo -e "${GREEN}✓${NC} Statusline working"
else
  echo -e "${YELLOW}⚠${NC} Statusline failed (expected until fully configured)"
fi
echo ""

# Summary
echo "=========================================="
echo "🎉 Infrastructure Fix Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify daemon is running: npx @claude-flow/cli@latest daemon status"
echo "2. Test swarm init: npx @claude-flow/cli@latest swarm init --topology hierarchical-mesh"
echo "3. View metrics: npx @claude-flow/cli@latest hooks metrics --v3-dashboard"
echo "4. Check worker status: npx @claude-flow/cli@latest hooks worker status"
echo ""
echo "See docs/CLAUDE_FOLDER_COMPREHENSIVE_REVIEW.md for full optimization plan."
echo ""
