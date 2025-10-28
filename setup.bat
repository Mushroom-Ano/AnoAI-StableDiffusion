@echo off
echo ============================================================
echo Stable Diffusion with DirectML - Setup Script
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Python detected
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
) else (
    echo [2/4] Virtual environment already exists
)
echo.

:: Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

:: Install dependencies
echo [4/4] Installing dependencies...
echo This may take several minutes on first run...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Setup completed successfully!
echo ============================================================
echo.
echo Next steps:
echo 1. Make sure your AMD GPU drivers are up to date
echo 2. Run: start.bat (or manually: python app.py)
echo 3. The web interface will open at http://127.0.0.1:7860
echo.
echo Note: First run will download the AI model (~4-5GB)
echo This is a one-time download and will be cached.
echo ============================================================
pause
