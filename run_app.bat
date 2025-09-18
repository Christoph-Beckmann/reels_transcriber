@echo off
title Instagram Reels Transcriber
echo.
echo ========================================
echo    Instagram Reels Transcriber
echo ========================================
echo.

:: Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Setting up...
    echo.

    :: Check if uv is installed
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv is not installed
        echo.
        echo Please install uv first:
        echo powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        echo.
        pause
        exit /b 1
    )

    :: Create virtual environment and install dependencies
    echo Creating virtual environment...
    uv venv
    echo.
    echo Installing dependencies (this may take a few minutes)...
    uv pip sync requirements.txt
    echo.
    echo Setup complete!
    echo.
)

:: Activate virtual environment and run the app
echo Starting application...
echo.

:: Use the virtual environment's Python
.venv\Scripts\python.exe main.py

:: Check if the app crashed
if errorlevel 1 (
    echo.
    echo ========================================
    echo Application encountered an error
    echo ========================================
    echo.
    pause
)
