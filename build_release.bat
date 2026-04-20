@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

if /I "%~1"=="--help" goto :show_help
if /I "%~1"=="-h" goto :show_help

set "VERSION_ARG="
set "COPY_DIR_ARG="

if not "%~1"=="" set "VERSION_ARG=--version %~1"
if not "%~2"=="" set "COPY_DIR_ARG=--copy-dir %~2"

"%PYTHON_EXE%" "%SCRIPT_DIR%build_release_win.py" --build-type cpu %VERSION_ARG% %COPY_DIR_ARG%
exit /b %ERRORLEVEL%

:show_help
echo SuperPicky Windows compatibility wrapper
echo.
echo Usage:
echo   %~nx0 [version] [copy_dir]
echo.
echo This wrapper forwards to build_release_win.py --build-type cpu.
exit /b 0