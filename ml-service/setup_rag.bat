@echo off
REM Setup script for BEP RAG System
REM This script sets up the Retrieval-Augmented Generation system

echo ============================================================
echo BEP RAG System Setup
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Please ensure Python 3.8+ is installed
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing Python dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 1: Extract text from DOCX files
echo ============================================================
echo.

REM Extract text from DOCX files
if exist "data\training_documents\docx\" (
    echo Extracting text from DOCX files...
    python scripts\extract_docx.py

    if errorlevel 1 (
        echo Warning: Text extraction had errors
        echo You can continue, but RAG quality may be affected
    )
) else (
    echo Warning: DOCX directory not found: data\training_documents\docx\
    echo Please add your BEP DOCX files to this directory
    echo.
    mkdir "data\training_documents\docx\" 2>nul
    echo Created directory: data\training_documents\docx\
    echo Please add DOCX files and run this script again
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 2: Configure Anthropic API Key
echo ============================================================
echo.

REM Check if .env file exists
if exist ".env" (
    echo .env file found
    findstr /C:"ANTHROPIC_API_KEY" .env >nul
    if errorlevel 1 (
        echo Warning: .env exists but ANTHROPIC_API_KEY not found
    ) else (
        echo API key configuration found in .env
    )
) else (
    echo No .env file found. Creating from template...
    if exist ".env.example" (
        copy .env.example .env
        echo.
        echo ====================================
        echo IMPORTANT: Configure your API key!
        echo ====================================
        echo.
        echo A .env file has been created for you.
        echo.
        echo Please:
        echo 1. Open: ml-service\.env
        echo 2. Replace "your-api-key-here" with your actual API key
        echo 3. Get your API key from: https://console.anthropic.com/
        echo.
        echo The .env file is protected by .gitignore and will NOT be uploaded to GitHub.
        echo.
        echo Press any key to open the .env file in Notepad...
        pause >nul
        notepad .env
        echo.
        echo After saving your API key, press any key to continue...
        pause >nul
    ) else (
        echo Error: .env.example not found
        echo.
        echo Please create .env file manually with:
        echo    ANTHROPIC_API_KEY=sk-ant-your-key-here
        echo.
        set /p continue="Continue without API key (will use LSTM fallback only)? (Y/N): "
        if /i not "%continue%"=="Y" (
            echo Setup cancelled
            pause
            exit /b 0
        )
    )
)

echo.
echo ============================================================
echo Step 3: Create Vector Database
echo ============================================================
echo.

REM Create vector database
echo Creating FAISS vector database from extracted documents...
echo This will download sentence-transformers model on first run...
echo.

python -c "from rag_engine import BEPRAGEngine; engine = BEPRAGEngine(); engine.initialize(); engine.load_or_create_vectorstore(force_rebuild=True); print('\nVector database created successfully!')"

if errorlevel 1 (
    echo.
    echo Warning: Vector database creation had errors
    echo The system may still work with LSTM fallback
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo The RAG system is now configured.
echo.
echo To start the ML service:
echo   cd ml-service
echo   venv\Scripts\activate
echo   python api.py
echo.
echo Or use the shortcut:
echo   start_service.bat
echo.
echo The API will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Remember to set your Anthropic API key for full RAG functionality:
echo   setx ANTHROPIC_API_KEY "your-api-key-here"
echo.
pause
