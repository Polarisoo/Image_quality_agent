@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Please install Python 3.10+ or Anaconda first.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating local virtual environment .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency installation failed. Please check your network or install requirements manually.
  pause
  exit /b 1
)

echo.
echo [INFO] Starting Task 3 image-quality Agent ...
echo [INFO] Open http://127.0.0.1:7860 in your browser.
python task3_web_app.py
pause
