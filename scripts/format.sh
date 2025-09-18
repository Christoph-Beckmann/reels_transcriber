#!/bin/bash

# format.sh - Comprehensive code formatting and quality checks
# This script runs all pre-commit checks and attempts to fix issues automatically

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}║                  Code Formatting & Quality Check                ║${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Change to project root
cd "$PROJECT_ROOT"

# Check if uv is available and prefer it
USE_UV=false
if command -v uv &> /dev/null; then
    USE_UV=true
    echo -e "${GREEN}Using uv for package management${NC}"

    # Ensure formatting tools are installed
    if ! uv run python -c "import black, isort" 2>/dev/null; then
        echo -e "${YELLOW}Installing formatting tools with uv...${NC}"
        uv pip install black isort flake8 pylint mypy --quiet || true
    fi
else
    # Fallback to traditional venv activation
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    fi
fi

# Track if any fixes were made
FIXES_MADE=false
ERRORS_FOUND=false

# Function to run a command and report status
run_check() {
    local name=$1
    local command=$2
    local fix_command=$3

    echo -e "\n${YELLOW}→ Running $name...${NC}"

    if [ -n "$fix_command" ]; then
        # Try to fix first
        if eval "$fix_command" 2>/dev/null; then
            echo -e "${GREEN}  ✓ Fixed issues with $name${NC}"
            FIXES_MADE=true
        fi
    fi

    # Run the check
    if eval "$command" 2>/dev/null; then
        echo -e "${GREEN}  ✓ $name passed${NC}"
        return 0
    else
        echo -e "${RED}  ✗ $name found issues${NC}"
        ERRORS_FOUND=true
        return 1
    fi
}

# 1. Ruff - Python linter and formatter
echo -e "\n${BLUE}[1/8] Ruff - Python Linting & Formatting${NC}"

# Fix import sorting and format issues
echo -e "${YELLOW}  → Auto-fixing with ruff...${NC}"
ruff check --fix --unsafe-fixes . 2>/dev/null || true
ruff format . 2>/dev/null || true

# Check if issues remain
if ruff check . && ruff format --check .; then
    echo -e "${GREEN}  ✓ Ruff checks passed${NC}"
else
    echo -e "${RED}  ✗ Ruff still has issues that need manual fixing${NC}"
    ERRORS_FOUND=true
fi

# 2. Black - Code formatting
echo -e "\n${BLUE}[2/8] Black - Code Formatting${NC}"
if [ "$USE_UV" = true ]; then
    BLACK_CHECK="uv run black --check --quiet ."
    BLACK_FIX="uv run black ."
else
    BLACK_CHECK="black --check --quiet ."
    BLACK_FIX="black ."
fi

if eval "$BLACK_CHECK"; then
    echo -e "${GREEN}  ✓ Black formatting is correct${NC}"
else
    echo -e "${YELLOW}  → Applying Black formatting...${NC}"
    eval "$BLACK_FIX"
    echo -e "${GREEN}  ✓ Black formatting applied${NC}"
    FIXES_MADE=true
fi

# 3. isort - Import sorting
echo -e "\n${BLUE}[3/8] isort - Import Sorting${NC}"
if [ "$USE_UV" = true ]; then
    ISORT_CHECK="uv run isort --check-only --quiet ."
    ISORT_FIX="uv run isort ."
else
    ISORT_CHECK="isort --check-only --quiet ."
    ISORT_FIX="isort ."
fi

if eval "$ISORT_CHECK"; then
    echo -e "${GREEN}  ✓ Import sorting is correct${NC}"
else
    echo -e "${YELLOW}  → Fixing import sorting...${NC}"
    eval "$ISORT_FIX"
    echo -e "${GREEN}  ✓ Import sorting fixed${NC}"
    FIXES_MADE=true
fi

# 4. Fix type hints (Dict -> dict, List -> list, Tuple -> tuple)
echo -e "\n${BLUE}[4/8] Type Hints - Modernizing for Python 3.9+${NC}"
echo -e "${YELLOW}  → Checking and fixing deprecated type hints...${NC}"

# Count deprecated type hints
DEPRECATED_COUNT=$(grep -r "from typing import.*\(Dict\|List\|Tuple\)" --include="*.py" . 2>/dev/null | wc -l || echo 0)

if [ "$DEPRECATED_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}  → Found $DEPRECATED_COUNT files with deprecated type hints, fixing...${NC}"

    # Fix deprecated type hints
    find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" ! -path "./.git/*" | while read -r file; do
        # Fix imports
        sed -i '' 's/from typing import \(.*\)Dict\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
        sed -i '' 's/from typing import \(.*\)List\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
        sed -i '' 's/from typing import \(.*\)Tuple\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true

        # Remove trailing commas in imports
        sed -i '' 's/from typing import \(.*\),$/from typing import \1/g' "$file" 2>/dev/null || true

        # Fix type annotations
        sed -i '' 's/Dict\[/dict[/g' "$file" 2>/dev/null || true
        sed -i '' 's/List\[/list[/g' "$file" 2>/dev/null || true
        sed -i '' 's/Tuple\[/tuple[/g' "$file" 2>/dev/null || true
        sed -i '' 's/: Dict\b/: dict/g' "$file" 2>/dev/null || true
        sed -i '' 's/: List\b/: list/g' "$file" 2>/dev/null || true
        sed -i '' 's/: Tuple\b/: tuple/g' "$file" 2>/dev/null || true
    done

    echo -e "${GREEN}  ✓ Type hints modernized${NC}"
    FIXES_MADE=true
else
    echo -e "${GREEN}  ✓ Type hints are already modern${NC}"
fi

# 5. Prettier - YAML/JSON/Markdown formatting
echo -e "\n${BLUE}[5/8] Prettier - YAML/JSON/Markdown Formatting${NC}"
if command -v prettier &> /dev/null; then
    if prettier --check "**/*.{json,yml,yaml,md}" 2>/dev/null; then
        echo -e "${GREEN}  ✓ File formatting is correct${NC}"
    else
        echo -e "${YELLOW}  → Formatting files with Prettier...${NC}"
        prettier --write "**/*.{json,yml,yaml,md}" 2>/dev/null || true
        echo -e "${GREEN}  ✓ Files formatted${NC}"
        FIXES_MADE=true
    fi
else
    echo -e "${YELLOW}  ⚠ Prettier not installed, skipping${NC}"
fi

# 6. Bandit - Security checks
echo -e "\n${BLUE}[6/8] Bandit - Security Analysis${NC}"
if bandit -r . -f json -o /dev/null 2>/dev/null; then
    echo -e "${GREEN}  ✓ No security issues found${NC}"
else
    echo -e "${YELLOW}  ⚠ Bandit found potential security issues${NC}"
    echo -e "${YELLOW}    Run 'bandit -r .' for details${NC}"
fi

# 7. PyDocStyle - Docstring checks
echo -e "\n${BLUE}[7/8] PyDocStyle - Documentation Standards${NC}"
if pydocstyle . 2>/dev/null; then
    echo -e "${GREEN}  ✓ Docstrings follow standards${NC}"
else
    echo -e "${YELLOW}  ⚠ Some docstrings need improvement${NC}"
    echo -e "${YELLOW}    Run 'pydocstyle .' for details${NC}"
fi

# 8. MyPy - Type checking
echo -e "\n${BLUE}[8/8] MyPy - Static Type Checking${NC}"
echo -e "${YELLOW}  → Running type checks...${NC}"

# Run mypy and capture the exit code
if mypy . --ignore-missing-imports 2>/dev/null; then
    echo -e "${GREEN}  ✓ Type checking passed${NC}"
else
    echo -e "${YELLOW}  ⚠ Type checking found issues${NC}"
    echo -e "${YELLOW}    Run 'mypy .' for details${NC}"
fi

# Additional fixes for common issues
echo -e "\n${BLUE}Additional Fixes${NC}"

# Fix trailing whitespace
echo -e "${YELLOW}  → Removing trailing whitespace...${NC}"
find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" ! -path "./.git/*" -exec sed -i '' 's/[[:space:]]*$//' {} \; 2>/dev/null || true
echo -e "${GREEN}  ✓ Trailing whitespace removed${NC}"

# Fix file endings (ensure newline at end of file)
echo -e "${YELLOW}  → Ensuring newlines at end of files...${NC}"
find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" ! -path "./.git/*" | while read -r file; do
    if [ -n "$(tail -c 1 "$file")" ]; then
        echo "" >> "$file"
    fi
done
echo -e "${GREEN}  ✓ File endings fixed${NC}"

# Summary
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}║                           Summary                              ║${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

if [ "$FIXES_MADE" = true ]; then
    echo -e "${GREEN}✓ Automatic fixes were applied${NC}"
    echo -e "${YELLOW}→ Please review the changes before committing${NC}"
fi

if [ "$ERRORS_FOUND" = true ]; then
    echo -e "${RED}✗ Some issues require manual intervention${NC}"
    echo -e "${YELLOW}→ Run 'pre-commit run --all-files' to see detailed errors${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${GREEN}→ Your code is ready to commit${NC}"
fi

# Optional: Run pre-commit as final validation
echo -e "\n${BLUE}Final Validation${NC}"
echo -e "${YELLOW}  → Running pre-commit hooks...${NC}"

if pre-commit run --all-files; then
    echo -e "${GREEN}  ✓ All pre-commit hooks passed!${NC}"
    echo -e "\n${GREEN}🎉 Your code is clean and ready to commit!${NC}"
    exit 0
else
    echo -e "${RED}  ✗ Pre-commit hooks still found issues${NC}"
    echo -e "${YELLOW}  → Please fix remaining issues manually${NC}"
    exit 1
fi
