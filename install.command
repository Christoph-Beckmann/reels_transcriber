#!/bin/bash
# macOS Double-clickable installer
# This file will actually run when double-clicked on Mac!

# Change to script directory
cd "$(dirname "$0")"

# Run the actual installer
./install.sh

# Keep terminal open to see results
echo ""
echo "Press Enter to close this window..."
read