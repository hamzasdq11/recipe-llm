@echo off
REM Run Recipe LLM locally on Windows

echo ========================================
echo    Recipe LLM - Local Setup ^& Run
echo ========================================

cd /d "%~dp0\.."

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    exit /b 1
)

echo Python version:
python --version

if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if not exist "data\recipes_small.json" (
    echo.
    echo Error: Recipe data not found at data\recipes_small.json
    exit /b 1
)

echo.
echo Recipe data found.

if not defined RECIPE_MODE set RECIPE_MODE=minimal
if not defined RECIPE_USE_MOCK set RECIPE_USE_MOCK=true
if not defined RECIPE_DATA_DIR set RECIPE_DATA_DIR=data

echo.
echo ========================================
echo    Starting Recipe LLM Server
echo ========================================
echo.
echo Mode: %RECIPE_MODE%
echo Mock Model: %RECIPE_USE_MOCK%
echo.
echo API will be available at: http://0.0.0.0:5000
echo Health check: http://0.0.0.0:5000/health
echo Web UI: http://0.0.0.0:5000/
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
