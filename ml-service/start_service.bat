@echo off
echo ============================================================
echo BEP AI Text Generator Service
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    echo.
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Checking for trained model...
if not exist "models\bep_model.pth" (
    echo.
    echo WARNING: Trained model not found!
    echo Please train the model first by running:
    echo   python scripts\train_model.py
    echo.
    echo Training with default settings now...
    python scripts\train_model.py --epochs 50
    echo.
)

echo.
echo Starting API service...
echo API will be available at: http://localhost:8000
echo API documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python api.py
