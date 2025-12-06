@echo off
echo ============================================================
echo GPU Monitoring Dashboard
echo ============================================================
echo.
echo Monitoring NVIDIA GeForce RTX 5060 Laptop GPU
echo Press Ctrl+C to stop monitoring
echo.
echo ============================================================
echo.

:loop
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits
timeout /t 2 /nobreak >nul
goto loop
