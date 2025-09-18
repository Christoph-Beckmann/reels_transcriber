#!/bin/bash

# quickfix.sh - Quick fixes for common linting issues
# Addresses the most common pre-commit hook failures automatically

set -e

echo "🔧 Quick Fix - Addressing common linting issues..."

PROJECT_ROOT="$(dirname "$(dirname "$0")")"
cd "$PROJECT_ROOT"

echo "→ Fixing type hints (Dict/List/Tuple -> dict/list/tuple)..."
find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" | while read -r file; do
    # Fix imports - remove Dict, List, Tuple from typing imports
    sed -i '' 's/from typing import \(.*\), Dict\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
    sed -i '' 's/from typing import \(.*\)Dict, \(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
    sed -i '' 's/from typing import \(.*\), List\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
    sed -i '' 's/from typing import \(.*\)List, \(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
    sed -i '' 's/from typing import \(.*\), Tuple\(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true
    sed -i '' 's/from typing import \(.*\)Tuple, \(.*\)/from typing import \1\2/g' "$file" 2>/dev/null || true

    # Clean up trailing commas in imports
    sed -i '' 's/from typing import \(.*\),$/from typing import \1/g' "$file" 2>/dev/null || true

    # Fix type annotations
    sed -i '' 's/Dict\[/dict[/g' "$file" 2>/dev/null || true
    sed -i '' 's/List\[/list[/g' "$file" 2>/dev/null || true
    sed -i '' 's/Tuple\[/tuple[/g' "$file" 2>/dev/null || true
    sed -i '' 's/: Dict\b/: dict/g' "$file" 2>/dev/null || true
    sed -i '' 's/: List\b/: list/g' "$file" 2>/dev/null || true
    sed -i '' 's/: Tuple\b/: tuple/g' "$file" 2>/dev/null || true
done

echo "→ Fixing common ruff issues..."
# Auto-fix with ruff
ruff check --fix --unsafe-fixes . 2>/dev/null || true

echo "→ Applying Black formatting..."
black . --quiet 2>/dev/null || true

echo "→ Fixing import sorting with isort..."
isort . --quiet 2>/dev/null || true

echo "→ Removing trailing whitespace..."
find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" -exec sed -i '' 's/[[:space:]]*$//' {} \;

echo "→ Ensuring newlines at end of files..."
find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./venv/*" | while read -r file; do
    if [ -n "$(tail -c 1 "$file")" ]; then
        echo "" >> "$file"
    fi
done

echo "✅ Quick fixes applied!"
echo ""
echo "Now run: git add -A && git commit -m 'Your message'"
echo "Or run: ./scripts/format.sh for a full check"
