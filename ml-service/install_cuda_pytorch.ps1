# PowerShell script to install CUDA-enabled PyTorch using D: drive for all temp files

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing CUDA-enabled PyTorch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create necessary directories on D: drive
$tempDir = "D:\pip_temp"
$cacheDir = "D:\pip_cache"

if (!(Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    Write-Host "Created temp directory: $tempDir" -ForegroundColor Green
}

if (!(Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    Write-Host "Created cache directory: $cacheDir" -ForegroundColor Green
}

# Set environment variables for this session
$env:TEMP = $tempDir
$env:TMP = $tempDir
$env:TMPDIR = $tempDir
$env:PIP_CACHE_DIR = $cacheDir

Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  TEMP = $env:TEMP"
Write-Host "  TMP = $env:TMP"
Write-Host "  PIP_CACHE_DIR = $env:PIP_CACHE_DIR"
Write-Host ""

# Navigate to ml-service directory
Set-Location "D:\03_Coding\W3_bep_generator\ml-service"

# Install CUDA-enabled PyTorch
Write-Host "Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Cyan
Write-Host "This will download approximately 2.8 GB of data" -ForegroundColor Yellow
Write-Host "Please be patient, this may take several minutes..." -ForegroundColor Yellow
Write-Host ""

& ".\venv\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Installation successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""

    # Verify CUDA availability
    Write-Host "Verifying CUDA installation..." -ForegroundColor Cyan
    & ".\venv\Scripts\python.exe" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

    Write-Host ""
    Write-Host "You can now train models using GPU acceleration!" -ForegroundColor Green
    Write-Host "Expected speedup: 5-10x faster than CPU training" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Installation failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error code: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
