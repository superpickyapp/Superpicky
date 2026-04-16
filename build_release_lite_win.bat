@echo off
setlocal EnableExtensions

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Missing build python: %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0build_release_win.py" --build-type lite %*
exit /b %ERRORLEVEL%
