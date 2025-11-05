@echo off
echo ============================================================
echo BEP AI System - Quick Test
echo ============================================================
echo.

REM Check if model exists
if not exist "ml-service\models\bep_model.pth" (
    echo ERROR: Model not trained yet!
    echo Please run: setup-ai.bat
    echo.
    pause
    exit /b 1
)

REM Check if ML service is running
echo Testing ML service connection...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ERROR: ML service is not running!
    echo Please start it with: cd ml-service ^&^& start_service.bat
    echo.
    pause
    exit /b 1
)

echo [OK] ML service is running
echo.

REM Test health endpoint
echo Testing health endpoint...
curl -s http://localhost:8000/health
echo.
echo.

REM Test generation with sample prompt
echo Testing text generation...
echo.
curl -X POST http://localhost:8000/suggest ^
  -H "Content-Type: application/json" ^
  -d "{\"field_type\":\"executiveSummary\",\"partial_text\":\"The BEP establishes\",\"max_length\":200}"
echo.
echo.

echo ============================================================
echo Test Complete!
echo ============================================================
echo.
echo If you see generated text above, the AI system is working correctly.
echo.
echo Next steps:
echo   1. Make sure BEP Generator is running: npm start
echo   2. Open a text field in the BEP editor
echo   3. Click the sparkle icon to generate AI suggestions
echo.
pause
