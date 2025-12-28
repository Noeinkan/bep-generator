@echo off
echo ============================================================
echo BEP Generator - Starting All Services
echo ============================================================
echo.
echo This will start:
echo   [1] React Frontend (port 3000)
echo   [2] Node.js Backend (port 3001)
echo   [3] Python ML Service (port 8000)
echo.
echo Press Ctrl+C to stop all services
echo ============================================================
echo.

REM Check if ML service virtual environment exists
if not exist "ml-service\venv\Scripts\python.exe" (
    echo WARNING: ML Service virtual environment not found!
    echo The ML service may not start correctly.
    echo.
    pause
)

REM Start all services using npm
npm start

pause
