@echo off
setlocal EnableExtensions

set "VERSION_INPUT=%~1"
if "%VERSION_INPUT%"=="" (
    set "VERSION_ARG=Win64_CPU"
) else (
    set "VERSION_ARG=%VERSION_INPUT%Win64_CPU"
)

call "%~dp0.venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

set "OUT_DIST_DIR=dist_cpu"
call "%~dp0build_release.bat" "%VERSION_ARG%" "output"
set "RET=%ERRORLEVEL%"

call "%~dp0.venv\Scripts\deactivate.bat" >nul 2>&1
exit /b %RET%
