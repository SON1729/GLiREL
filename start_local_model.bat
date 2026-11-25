@echo off
cd /d "%~dp0"
call start_windows.bat --local-model --port 8503 %*
