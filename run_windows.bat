@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe.
  echo Run install_windows.bat from this folder first.
  exit /b 1
)
.venv\Scripts\python.exe app.py %*
