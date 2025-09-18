#!/bin/bash

# Quick formatting script - runs the essential formatting tools
# This is a simplified version that works with current environment

set -e

echo "🎬 Quick Format - Essential Code Formatting"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📍 Working in: $PROJECT_ROOT"
echo ""

# Check what tools we have available
echo "🔍 Checking available tools..."
echo ""

HAVE_BLACK=false
HAVE_ISORT=false
HAVE_FLAKE8=false

# Try to find tools in venv first
if [ -d ".venv" ]; then
    if [ -f ".venv/bin/black" ]; then
        HAVE_BLACK=true
        BLACK_CMD=".venv/bin/black"
        echo -e "${GREEN}✓ Found black in .venv${NC}"
    fi
    if [ -f ".venv/bin/isort" ]; then
        HAVE_ISORT=true
        ISORT_CMD=".venv/bin/isort"
        echo -e "${GREEN}✓ Found isort in .venv${NC}"
    fi
    if [ -f ".venv/bin/flake8" ]; then
        HAVE_FLAKE8=true
        FLAKE8_CMD=".venv/bin/flake8"
        echo -e "${GREEN}✓ Found flake8 in .venv${NC}"
    fi
fi

# Try with uv if tools not found
if [ "$HAVE_BLACK" = false ] && command -v uv &> /dev/null; then
    if uv run python -c "import black" 2>/dev/null; then
        HAVE_BLACK=true
        BLACK_CMD="uv run black"
        echo -e "${GREEN}✓ Black available via uv${NC}"
    fi
fi

if [ "$HAVE_ISORT" = false ] && command -v uv &> /dev/null; then
    if uv run python -c "import isort" 2>/dev/null; then
        HAVE_ISORT=true
        ISORT_CMD="uv run isort"
        echo -e "${GREEN}✓ isort available via uv${NC}"
    fi
fi

if [ "$HAVE_FLAKE8" = false ] && command -v uv &> /dev/null; then
    if uv run python -c "import flake8" 2>/dev/null; then
        HAVE_FLAKE8=true
        FLAKE8_CMD="uv run flake8"
        echo -e "${GREEN}✓ flake8 available via uv${NC}"
    fi
fi

echo ""

# Run available tools
if [ "$HAVE_BLACK" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 Running Black formatter..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if $BLACK_CMD --check . 2>/dev/null; then
        echo -e "${GREEN}✓ Code is already formatted${NC}"
    else
        echo -e "${YELLOW}Applying Black formatting...${NC}"
        $BLACK_CMD . || echo -e "${YELLOW}⚠️ Black encountered some issues${NC}"
        echo -e "${GREEN}✓ Black formatting complete${NC}"
    fi
    echo ""
fi

if [ "$HAVE_ISORT" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Running import sorter..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if $ISORT_CMD --check-only . 2>/dev/null; then
        echo -e "${GREEN}✓ Imports are already sorted${NC}"
    else
        echo -e "${YELLOW}Sorting imports...${NC}"
        $ISORT_CMD . || echo -e "${YELLOW}⚠️ isort encountered some issues${NC}"
        echo -e "${GREEN}✓ Import sorting complete${NC}"
    fi
    echo ""
fi

if [ "$HAVE_FLAKE8" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 Running linter checks..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}Running flake8 (issues shown below if any)...${NC}"
    $FLAKE8_CMD core gui utils config cli.py main.py --max-line-length=120 --extend-ignore=E203,W503 2>/dev/null || {
        echo ""
        echo -e "${YELLOW}⚠️ Linting issues found - review above${NC}"
    }
    echo ""
fi

# Summary
echo "═════════════════════════════"
echo "📊 Summary"
echo "═════════════════════════════"

if [ "$HAVE_BLACK" = false ] && [ "$HAVE_ISORT" = false ] && [ "$HAVE_FLAKE8" = false ]; then
    echo -e "${RED}❌ No formatting tools found${NC}"
    echo ""
    echo "To install tools, run:"
    echo "  uv pip install black isort flake8"
    exit 1
fi

echo -e "${GREEN}✅ Formatting complete!${NC}"
echo ""
echo "Tools used:"
[ "$HAVE_BLACK" = true ] && echo "  • Black (code formatting)"
[ "$HAVE_ISORT" = true ] && echo "  • isort (import sorting)"
[ "$HAVE_FLAKE8" = true ] && echo "  • flake8 (linting)"

echo ""
echo "💡 Tip: For comprehensive checks, run:"
echo "   ./scripts/format.sh"
