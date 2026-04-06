@echo off
setlocal

:: Check if unpacker dependency exists
if exist "%~dp0lazorr410-unpacker\python\paz_parse.py" goto :start

echo [*] lazorr410-unpacker not found, attempting to fetch...

:: Check if git is installed
where git >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Git is not installed or not in PATH.
    echo.
    echo The lazorr410-unpacker dependency is missing and cannot be fetched automatically.
    echo Please install Git from https://git-scm.com and re-run this script,
    echo or manually clone it:
    echo.
    echo   git clone https://github.com/lazorr410/crimson-desert-unpacker.git "%~dp0lazorr410-unpacker"
    echo.
    pause
    exit /b 1
)

:: Clone the unpacker
git clone https://github.com/lazorr410/crimson-desert-unpacker.git "%~dp0lazorr410-unpacker"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to clone lazorr410-unpacker. Check your internet connection.
    pause
    exit /b 1
)

echo [*] Dependency fetched successfully.
echo.

:start
:: Install Python packages if needed (check for PySide6 as marker)
python -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo [*] Installing Python dependencies...
    pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies. Check your Python/pip setup.
        pause
        exit /b 1
    )
    echo.
)

python "%~dp0src\pac_browser.py" %*
