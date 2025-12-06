@echo off
echo ============================================================
echo BEP AI Training with Real-time Dashboard
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting training dashboard server...
echo.
echo The dashboard will open at: http://localhost:5000
echo.
echo Starting training in 3 seconds...
echo Open the dashboard URL in your browser NOW!
echo.

REM Start dashboard in background
start /B python training_dashboard.py

REM Wait a bit for dashboard to start
timeout /t 3 /nobreak >nul

REM Start training with arguments (default 100 epochs)
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=100

echo.
echo ============================================================
echo Training will run for %EPOCHS% epochs
echo Watch progress at: http://localhost:5000
echo ============================================================
echo.

python scripts\train_model.py --epochs %EPOCHS%

echo.
echo ============================================================
echo Training complete!
echo Dashboard is still running at http://localhost:5000
echo Press any key to stop the dashboard...
echo ============================================================
pause >nul

REM Kill the dashboard server
taskkill /F /IM python.exe /FI "WINDOWTITLE eq training_dashboard*" >nul 2>&1

echo.
echo Dashboard stopped.
echo.
