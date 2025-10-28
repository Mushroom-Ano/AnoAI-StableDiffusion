@echo off
echo ============================================================
echo Stable Diffusion with DirectML - Starting Application
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

:: Check if dependencies are installed
python -c "import torch_directml" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Dependencies not installed!
    echo Please run setup.bat first
    pause
    exit /b 1
)

:: Run the application
echo Starting Stable Diffusion application...
echo.
echo The web interface will open automatically at:
echo http://127.0.0.1:7860
echo.
echo Note: First run will download the AI model (~4-5GB)
echo Press Ctrl+C to stop the application
echo.
echo ============================================================
echo.

python app.py

pause
