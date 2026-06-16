@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  echo Reusing existing virtual environment at .venv.
) else (
  call :find_python
  if errorlevel 1 exit /b 1

  echo Creating virtual environment at .venv.
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
  if errorlevel 1 exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m playwright install chromium
if errorlevel 1 exit /b 1
echo DBillTOGIT dependencies are installed.
exit /b 0

:find_python
set "PYTHON_EXE="
set "PYTHON_ARGS="

py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=py"
  set "PYTHON_ARGS=-3"
  exit /b 0
)

python --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=python"
  set "PYTHON_ARGS="
  exit /b 0
)

echo Python 3 was not found. Trying to install Python 3.12 with winget...
where winget >nul 2>&1
if errorlevel 1 (
  echo winget is not available on this system.
) else (
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
)

py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=py"
  set "PYTHON_ARGS=-3"
  exit /b 0
)

python --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_EXE=python"
  set "PYTHON_ARGS="
  exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
  set "PYTHON_ARGS="
  exit /b 0
)

echo Automatic Python install failed.
echo Install Python manually from https://www.python.org/downloads/windows/
echo Then close this terminal, open a new Command Prompt, and run install_windows.bat again.
exit /b 1
