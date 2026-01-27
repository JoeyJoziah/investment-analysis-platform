#!/bin/bash
# validate-agent-structure.sh - Validates the reorganized agent structure

set -e

AGENTS_DIR=".claude/agents"
ERROR_COUNT=0
WARNING_COUNT=0

echo "🔍 Agent Structure Validation"
echo "=============================="
echo ""

# Check if new structure exists
echo "📁 Checking new directory structure..."
NEW_DIRS=(
    "1-core"
    "2-swarm-coordination"
    "3-security-performance"
    "4-github-repository"
    "5-sparc-methodology"
    "6-specialized-development"
    "7-testing-validation"
)

for dir in "${NEW_DIRS[@]}"; do
    if [ -d "$AGENTS_DIR/$dir" ]; then
        agent_count=$(find "$AGENTS_DIR/$dir" -name "*.md" -type f | wc -l | tr -d ' ')
        echo "   ✅ $dir/ exists ($agent_count agents)"
    else
        echo "   ❌ $dir/ missing"
        ((ERROR_COUNT++))
    fi
done
echo ""

# Validate YAML frontmatter in all agent files
echo "📝 Validating agent file format..."
INVALID_FILES=0
VALID_FILES=0

for category in "${NEW_DIRS[@]}"; do
    if [ -d "$AGENTS_DIR/$category" ]; then
        for file in "$AGENTS_DIR/$category"/*.md; do
            if [ -f "$file" ]; then
                # Check if file has YAML frontmatter
                if head -n 1 "$file" | grep -q "^---$"; then
                    ((VALID_FILES++))
                else
                    echo "   ⚠️  Missing frontmatter: $file"
                    ((INVALID_FILES++))
                    ((WARNING_COUNT++))
                fi
            fi
        done
    fi
done

echo "   Valid files: $VALID_FILES"
echo "   Invalid files: $INVALID_FILES"
echo ""

# Check for expected minimum agent counts per category
echo "📊 Checking agent counts per category..."
declare -A EXPECTED_COUNTS=(
    ["1-core"]=5
    ["2-swarm-coordination"]=20
    ["3-security-performance"]=12
    ["4-github-repository"]=15
    ["5-sparc-methodology"]=8
    ["6-specialized-development"]=25
    ["7-testing-validation"]=8
)

for category in "${NEW_DIRS[@]}"; do
    if [ -d "$AGENTS_DIR/$category" ]; then
        actual_count=$(find "$AGENTS_DIR/$category" -name "*.md" -type f | wc -l | tr -d ' ')
        expected_count=${EXPECTED_COUNTS[$category]}

        if [ "$actual_count" -ge "$expected_count" ]; then
            echo "   ✅ $category: $actual_count agents (expected >= $expected_count)"
        else
            echo "   ⚠️  $category: $actual_count agents (expected >= $expected_count)"
            ((WARNING_COUNT++))
        fi
    fi
done
echo ""

# Check for duplicate files
echo "🔍 Checking for duplicate agent files..."
DUPLICATES=$(find "$AGENTS_DIR" -name "*.md" -type f -exec basename {} \; | sort | uniq -d)
if [ -z "$DUPLICATES" ]; then
    echo "   ✅ No duplicate filenames found"
else
    echo "   ⚠️  Duplicate filenames found:"
    echo "$DUPLICATES" | while read -r dup; do
        echo "      - $dup"
        find "$AGENTS_DIR" -name "$dup" -type f
    done
    ((WARNING_COUNT++))
fi
echo ""

# Check for broken symlinks
echo "🔗 Checking for broken symlinks..."
BROKEN_LINKS=$(find "$AGENTS_DIR" -type l ! -exec test -e {} \; -print 2>/dev/null)
if [ -z "$BROKEN_LINKS" ]; then
    echo "   ✅ No broken symlinks found"
else
    echo "   ❌ Broken symlinks found:"
    echo "$BROKEN_LINKS"
    ((ERROR_COUNT++))
fi
echo ""

# Generate statistics
echo "📈 Statistics:"
TOTAL_AGENTS=$(find "$AGENTS_DIR" -name "*.md" -type f | wc -l | tr -d ' ')
TOTAL_DIRS=$(find "$AGENTS_DIR" -type d -mindepth 1 | wc -l | tr -d ' ')
echo "   Total agents: $TOTAL_AGENTS"
echo "   Total directories: $TOTAL_DIRS"
echo "   Average agents per directory: $((TOTAL_AGENTS / TOTAL_DIRS))"
echo ""

# Check old structure (for cleanup)
echo "🗑️  Checking old structure..."
OLD_DIRS=(
    "core" "swarm" "consensus" "optimization" "hive-mind"
    "sublinear" "sona" "analysis" "github" "github-swarm"
    "devops" "ci-cd" "sparc" "goal" "development"
    "backend" "mobile" "specialized" "ml" "data"
    "architecture" "system-design" "api-docs" "documentation"
    "payments" "flow-nexus" "v3" "testing" "unit" "validation"
)

OLD_STRUCTURE_EXISTS=false
for dir in "${OLD_DIRS[@]}"; do
    if [ -d "$AGENTS_DIR/$dir" ]; then
        agent_count=$(find "$AGENTS_DIR/$dir" -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [ "$agent_count" -gt 0 ]; then
            echo "   ⚠️  Old directory exists with agents: $dir/ ($agent_count agents)"
            OLD_STRUCTURE_EXISTS=true
            ((WARNING_COUNT++))
        fi
    fi
done

if [ "$OLD_STRUCTURE_EXISTS" = false ]; then
    echo "   ✅ No old structure directories with agents found"
fi
echo ""

# Final summary
echo "════════════════════════════════"
if [ $ERROR_COUNT -eq 0 ] && [ $WARNING_COUNT -eq 0 ]; then
    echo "✅ Validation PASSED"
    echo "   No errors or warnings found"
    exit 0
elif [ $ERROR_COUNT -eq 0 ]; then
    echo "⚠️  Validation PASSED with WARNINGS"
    echo "   Errors: $ERROR_COUNT"
    echo "   Warnings: $WARNING_COUNT"
    exit 0
else
    echo "❌ Validation FAILED"
    echo "   Errors: $ERROR_COUNT"
    echo "   Warnings: $WARNING_COUNT"
    exit 1
fi
