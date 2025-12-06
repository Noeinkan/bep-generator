@echo off
echo ============================================================
echo BEP Generator AI Setup
echo ============================================================
echo.
echo This script will set up the AI text generation system.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/4] Python found!
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "ml-service\venv" (
    echo [2/4] Creating Python virtual environment...
    cd ml-service
    python -m venv venv
    cd ..
    echo Virtual environment created!
) else (
    echo [2/4] Virtual environment already exists.
)
echo.

REM Install Python dependencies
echo [3/4] Installing Python dependencies...
echo This may take a few minutes...
call ml-service\venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r ml-service\requirements.txt
echo Dependencies installed!
echo.

REM Train the model if it doesn't exist
if not exist "ml-service\models\bep_model.pth" (
    echo [4/4] Training AI model...
    echo This will take 10-20 minutes depending on your CPU.
    echo You can interrupt this and train later with: python ml-service\scripts\train_model.py
    echo.
    choice /C YN /M "Do you want to train the model now"
    if errorlevel 2 (
        echo.
        echo Model training skipped.
        echo You must train the model before using AI features!
        echo Run: cd ml-service ^&^& python scripts\train_model.py
        goto :done
    )
    echo.
    echo Training model with 100 epochs...
    call ml-service\venv\Scripts\activate.bat
    python ml-service\scripts\train_model.py --epochs 100
    echo.
    echo Model trained successfully!
) else (
    echo [4/4] Model already trained.
)
echo.

:done
echo ============================================================
echo AI Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Start the ML service: cd ml-service ^&^& start_service.bat
echo   2. Start BEP Generator: npm start
echo   3. Look for the sparkle icon in text editors to use AI suggestions
echo.
echo For more information, see AI_INTEGRATION_GUIDE.md
echo.
pause
