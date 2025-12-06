@echo off
echo ============================================================
echo Starting BEP AI Text Generator Service
echo ============================================================
echo.

REM Check if model exists
if not exist "models\bep_model.pth" (
    echo ERROR: Model not trained yet!
    echo Please run: cd .. ^&^& setup-ai.bat
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting FastAPI service on http://localhost:8000
echo Press Ctrl+C to stop the service
echo.
echo ============================================================
echo.

REM Start the API service
python api.py
