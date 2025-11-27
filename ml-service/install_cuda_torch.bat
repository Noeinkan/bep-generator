@echo off
REM Install CUDA-enabled PyTorch using D: drive for temp files

REM Set temp directories to D: drive
set TEMP=D:\temp
set TMP=D:\temp
set TMPDIR=D:\temp

REM Create temp directory if it doesn't exist
if not exist D:\temp mkdir D:\temp

REM Navigate to ml-service directory
cd /d D:\03_Coding\W3_bep_generator\ml-service

REM Install CUDA-enabled PyTorch
echo Installing CUDA-enabled PyTorch (this will download ~2.8 GB)...
venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installation complete!
echo.

REM Verify CUDA is available
echo Verifying CUDA installation...
venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

pause
