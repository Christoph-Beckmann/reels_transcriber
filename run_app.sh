#!/bin/bash

# Instagram Reels Transcriber - Launcher Script
clear

echo "========================================"
echo "   Instagram Reels Transcriber"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Setting up..."
    echo ""

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo "ERROR: uv is not installed"
        echo ""
        echo "Please install uv first:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        exit 1
    fi

    # Create virtual environment and install dependencies
    echo "Creating virtual environment..."
    uv venv
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    uv pip sync requirements.txt
    echo ""
    echo "Setup complete!"
    echo ""
fi

echo "Starting application..."
echo ""

# Use the virtual environment's Python
.venv/bin/python main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Application encountered an error"
    echo "========================================"
    echo ""
    echo "Press Enter to exit..."
    read
fi
