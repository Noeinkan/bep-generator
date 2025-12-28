@echo off
REM Start BEP ML Service with Ollama Backend
REM This script starts the FastAPI service that uses Ollama for AI generation

echo ================================================================
echo    BEP ML Service - Ollama Backend
echo ================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first or create venv manually
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not running!
    echo.
    echo Please start Ollama first:
    echo   - Windows: Search "Ollama" in Start Menu and launch it
    echo   - Or run: ollama serve
    echo.
    echo After Ollama is running, press any key to continue...
    pause >nul
)

echo [OK] Ollama is running
echo.

echo [2/3] Checking required packages...
venv\Scripts\python.exe -c "import fastapi; import requests" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Missing packages. Installing...
    venv\Scripts\pip.exe install fastapi uvicorn requests
)
echo [OK] All packages installed
echo.

echo [3/3] Starting ML Service on port 5003...
echo.
echo ================================================================
echo API will be available at: http://localhost:5003
echo API Documentation: http://localhost:5003/docs
echo ================================================================
echo.

REM Start the service
venv\Scripts\python.exe api_ollama.py

pause
