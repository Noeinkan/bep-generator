@echo off
echo ============================================================
echo BEP AI Training with TensorBoard Dashboard
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting TensorBoard dashboard server...
echo.
echo Dashboard URL: http://localhost:6006
echo.

REM Start TensorBoard in background
start "TensorBoard Dashboard" cmd /c "venv\Scripts\tensorboard.exe --logdir=runs --port=6006"

REM Wait for TensorBoard to start
timeout /t 3 /nobreak >nul

REM Open browser
start http://localhost:6006

REM Get epochs from argument (default 100)
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=100

echo.
echo ============================================================
echo Training Configuration
echo ============================================================
echo Epochs: %EPOCHS%
echo Dashboard: http://localhost:6006
echo.
echo The browser will open automatically
echo Training will start in 2 seconds...
echo ============================================================
echo.

timeout /t 2 /nobreak >nul

REM Start training with live output
python -u scripts\train_model.py --epochs %EPOCHS%

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo TensorBoard is still running at: http://localhost:6006
echo You can view the training results and graphs
echo.
echo Press any key to stop TensorBoard and exit...
echo ============================================================
pause >nul

REM Kill TensorBoard
taskkill /F /FI "WINDOWTITLE eq TensorBoard Dashboard*" >nul 2>&1

echo.
echo TensorBoard stopped.
echo.
