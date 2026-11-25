@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m venv .venv
  goto check
)
where python >nul 2>nul
if %errorlevel%==0 (
  python -m venv .venv
  goto check
)
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
  goto check
)
echo Python not found. Install Python 3.12 from python.org, then try again.
pause
exit /b 1
:check
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the Python environment.
  pause
  exit /b 1
)
:run
".venv\Scripts\python.exe" run_demo.py %*
if errorlevel 1 pause
