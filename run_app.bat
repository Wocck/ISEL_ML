@echo off
setlocal

echo [INFO] Checking Python environment...

REM ---- Detect UV ----
where uv >nul 2>&1
if %errorlevel%==0 (
    set UV_AVAILABLE=1
) else (
    set UV_AVAILABLE=0
)

if %UV_AVAILABLE%==1 (
    echo [INFO] uv detected. Using uv.
) else (
    echo [INFO] uv not found. Falling back to python + pip.
)

REM ---- Deactivate any currently active venv ----
if defined VIRTUAL_ENV (
    echo [INFO] Deactivating previously active environment...
    call deactivate >nul 2>&1
)

REM ---- Create .venv if missing ----
if not exist ".venv\" (
    echo [INFO] Creating virtual environment...
    if %UV_AVAILABLE%==1 (
        uv venv
    ) else (
        python -m venv .venv
    )
)

REM ---- Activate the environment ----
call ".venv\Scripts\activate"

REM ---- Install dependencies ----
if %UV_AVAILABLE%==1 (
    echo [INFO] Installing dependencies via uv...
    uv sync --active
) else (
    echo [INFO] Installing dependencies via pip...
    if not exist requirements.txt (
        echo [ERROR] requirements.txt not found!
        echo Run:
        echo   uv pip compile pyproject.toml -o requirements.txt
        pause
        exit /b 1
    )
    pip install -r requirements.txt
)

REM ---- Run WebApp ----
echo [INFO] Starting web server...
echo URL: http://127.0.0.1:8000
echo Press CTRL+C to stop.

python -m uvicorn webapp.app:app --reload

echo.
echo [OK] WebApp stopped.
pause
