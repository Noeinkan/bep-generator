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
    echo.
    echo ============================================================
    echo Starting TensorBoard Dashboard
    echo ============================================================
    echo TensorBoard will open automatically in your browser
    echo Dashboard URL: http://localhost:6006
    echo.

    REM Start TensorBoard in background
    start "TensorBoard" cmd /c "venv\Scripts\tensorboard.exe --logdir=runs --port=6006"

    REM Wait for TensorBoard to start
    timeout /t 3 /nobreak >nul

    REM Open TensorBoard in browser
    start http://localhost:6006

    echo.
    echo ============================================================
    echo Starting Model Training with Live Monitoring
    echo ============================================================
    echo Training: 50 epochs with progress bars
    echo Monitor live in TensorBoard dashboard
    echo.

    python -u scripts\train_model.py --epochs 50
    echo.
    echo Training completed!
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
