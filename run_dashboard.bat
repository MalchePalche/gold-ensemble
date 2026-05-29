@echo off
title Gold Ensemble V4 — Dashboard
cd /d "%~dp0"

echo ========================================================
echo   Gold Ensemble V4 Dashboard
echo ========================================================
echo.

REM ── verify Python is available ─────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: python not found in PATH.
    echo Make sure Python 3.9+ is installed and on PATH.
    pause
    exit /b 1
)

REM ── ensure Flask is installed ──────────────────────────────────────────
python -c "import flask" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Flask not found. Installing...
    python -m pip install flask --quiet
)

REM ── start Flask server in a separate window ───────────────────────────
echo Starting Flask server...
start "Gold Ensemble — Server" python dashboard\web_dashboard.py

REM ── wait for server to bind, then open browser ────────────────────────
timeout /t 4 /nobreak >nul
echo Opening http://localhost:5000 ...
start "" http://localhost:5000

echo.
echo Dashboard running at http://localhost:5000
echo Close the "Gold Ensemble — Server" window to stop.
echo.
