@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "APP_NAME=SuperPicky"
set "SPEC_FILE=SuperPicky_win64_onnx.spec"
set "ROOT_DIR=%~dp0"
set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "DEFAULT_OUT_DIST_DIR=dist_cpu_onnx"
set "BUILD_SUFFIX=_Win64_CPU_ONNX"
cd /d "%ROOT_DIR%"

set "VERSION_ARG="
set "ZIP_COPY_DIR="

if "!OUT_DIST_DIR!"=="" set "OUT_DIST_DIR=%DEFAULT_OUT_DIST_DIR%"

set "BUILD_ZIP=1"

call :parse_args %*
if errorlevel 1 exit /b 1
if defined SHOW_HELP goto :show_help

goto :start

:show_help
echo SuperPicky Windows CPU ONNX build script
echo.
echo Usage:
echo   %~nx0 [version] [zip_copy_dir]
echo.
echo   version       Optional base version, for example 4.0.6
echo                 The script always appends %BUILD_SUFFIX%
echo                 When omitted, the base version is read from ui/about_dialog.py
echo   zip_copy_dir  Optional copy target directory
echo                 When omitted, no extra copy/output folder is created
echo.
exit /b 0

:parse_args
:parse_args_loop
if "%~1"=="" exit /b 0

if /i "%~1"=="--help" (
    set "SHOW_HELP=1"
    exit /b 0
)
if /i "%~1"=="-h" (
    set "SHOW_HELP=1"
    exit /b 0
)
if "%VERSION_ARG%"=="" (
    set "VERSION_ARG=%~1"
) else if "!ZIP_COPY_DIR!"=="" (
    set "ZIP_COPY_DIR=%~1"
) else (
    echo [WARNING] Ignored extra argument: %~1
)
shift
goto :parse_args_loop

:start
echo.
echo [========================================]
echo Step 0: Clean old build files
echo [========================================]

set "INNO_DIR=%ROOT_DIR%\inno"

if exist "%ROOT_DIR%\build_!OUT_DIST_DIR!" rd /s /q "%ROOT_DIR%\build_!OUT_DIST_DIR!" >nul 2>&1
if exist "%ROOT_DIR%\!OUT_DIST_DIR!" rd /s /q "%ROOT_DIR%\!OUT_DIST_DIR!" >nul 2>&1

if defined ZIP_COPY_DIR (
    if exist "%ZIP_COPY_DIR%" rd /s /q "%ZIP_COPY_DIR%" >nul 2>&1
)

echo [SUCCESS] Cleaned old build files

echo.
echo [========================================]
echo Step 1: Environment check
echo [========================================]

if not exist "%SPEC_FILE%" (
    echo [ERROR] Missing spec file: %SPEC_FILE%
    exit /b 1
)

echo [SUCCESS] Spec file found: %SPEC_FILE%

if "!PYTHON_EXE!"=="" set "PYTHON_EXE=python"
if "!PYTHON_EXE!"=="python" (
    where python >nul 2>nul && for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%i"
)
if "!PYTHON_EXE!"=="" set "PYTHON_EXE=python"
call :check_python "!PYTHON_EXE!" "default"
if errorlevel 1 exit /b 1

echo.
echo [========================================]
echo Step 1: Resolve version
echo [========================================]

set "VERSION_BASE=4.0.5_sp3"
if not "!VERSION_ARG!"=="" (
    set "VERSION_BASE=!VERSION_ARG!"
    echo [SUCCESS] Use base version from args: !VERSION_BASE!
) else (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "$c=Get-Content -Path 'ui/about_dialog.py' -Raw -Encoding UTF8; if($c -match 'v([0-9A-Za-z._-]+)'){ $matches[1] }"`) do (
        set "VERSION_BASE=%%i"
    )
    if "!VERSION_BASE!"=="" set "VERSION_BASE=0.0.0"
    echo [SUCCESS] Detected base version: v!VERSION_BASE!
)

call :compose_version "!VERSION_BASE!"
echo [SUCCESS] Final package version: !VERSION!

echo.
echo [========================================]
echo Step 1.5: Inject build metadata
echo [========================================]

set "COMMIT_HASH=unknown"
for /f "tokens=*" %%i in ('"%PYTHON_EXE%" -c "exec('try:\n from core.build_info_local import COMMIT_HASH\nexcept ImportError:\n from core.build_info import COMMIT_HASH\nprint(COMMIT_HASH or chr(0))')" 2^>nul') do set "COMMIT_HASH=%%i"
if "%COMMIT_HASH%"=="" for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set "COMMIT_HASH=%%i"
if "%COMMIT_HASH%"=="" set "COMMIT_HASH=unknown"
echo [INFO] Commit hash: %COMMIT_HASH%

set "BUILD_INFO_FILE=core\build_info.py"
set "BUILD_INFO_BACKUP=core\build_info.py.backup"
if exist "%BUILD_INFO_FILE%" copy /y "%BUILD_INFO_FILE%" "%BUILD_INFO_BACKUP%" >nul

powershell -NoProfile -Command "(Get-Content -Path '%BUILD_INFO_FILE%' -Raw -Encoding UTF8) -replace 'COMMIT_HASH\s*=\s*.*', 'COMMIT_HASH = \"%COMMIT_HASH%\"' | Set-Content -Path '%BUILD_INFO_FILE%' -Encoding UTF8"
if errorlevel 1 (
    echo [ERROR] Failed to inject build info
    call :restore_build_info >nul
    exit /b 1
)

echo [SUCCESS] Build info injected

call :build_single
set "RET=%ERRORLEVEL%"
call :restore_build_info >nul
exit /b %RET%

:compose_version
set "VERSION=%~1"
echo(!VERSION!| findstr /i /e /c:"%BUILD_SUFFIX%" >nul
if not errorlevel 1 exit /b 0
set "VERSION=!VERSION!%BUILD_SUFFIX%"
exit /b 0

:check_python
set "CHECK_PY=%~1"
set "CHECK_LABEL=%~2"

echo [INFO] Checking Python (%CHECK_LABEL%): %CHECK_PY%
"%CHECK_PY%" -c "import sys; print(sys.executable)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not available: %CHECK_PY%
    exit /b 1
)
for /f "tokens=*" %%i in ('"%CHECK_PY%" -c "import sys; print(sys.executable)" 2^>nul') do set "_PY_RESOLVED=%%i"
echo [SUCCESS] Python (%CHECK_LABEL%): !_PY_RESOLVED!

echo [INFO] Checking PyInstaller (%CHECK_LABEL%)...
"%CHECK_PY%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller missing in %CHECK_LABEL% environment
    exit /b 1
)
echo [SUCCESS] PyInstaller is available (%CHECK_LABEL%)
exit /b 0

:build_with_python
set "B_PY=%~1"
set "B_WORK=%~2"
set "B_DIST=%~3"
set "B_LABEL=%~4"

echo.
echo [========================================]
echo Build: %B_LABEL%
echo [========================================]

if exist "%B_WORK%" rd /s /q "%B_WORK%"
if exist "%B_DIST%" rd /s /q "%B_DIST%"

"%B_PY%" -m PyInstaller "%SPEC_FILE%" --clean --noconfirm --workpath "%B_WORK%" --distpath "%B_DIST%"
set "PYI_RC=%ERRORLEVEL%"
echo [INFO] PyInstaller process rc (%B_LABEL%): %PYI_RC%
if not "%PYI_RC%"=="0" (
    echo [WARNING] PyInstaller returned non-zero [%B_LABEL%]: %PYI_RC%
)

if not exist "%B_DIST%\%APP_NAME%\SuperPicky.exe" (
    echo [ERROR] Missing output exe: %B_DIST%\%APP_NAME%\SuperPicky.exe
    exit /b 1
)

echo [SUCCESS] Build completed (%B_LABEL%)
exit /b 0

:copy_dir
set "C_SRC=%~1"
set "C_DST=%~2"

if not exist "%C_SRC%" (
    echo [ERROR] Copy source not found: %C_SRC%
    exit /b 1
)

if not exist "%C_DST%" mkdir "%C_DST%"
if errorlevel 1 (
    echo [ERROR] Failed to create target dir: %C_DST%
    exit /b 1
)

robocopy "%C_SRC%" "%C_DST%" /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP >nul
set "COPY_RC=%ERRORLEVEL%"
if !COPY_RC! GEQ 8 (
    echo [ERROR] Failed to copy to %C_DST% ^(robocopy exit code !COPY_RC!^)
    exit /b 1
)

echo [SUCCESS] Copied directory: %C_SRC% -^> %C_DST%
exit /b 0

:zip_dir
set "Z_SRC=%~1"
set "Z_OUT=%~2"
set "ZIP_RC=1"

if not exist "%Z_SRC%" (
    echo [ERROR] Zip source not found: %Z_SRC%
    exit /b 1
)

if exist "%Z_OUT%" del /q "%Z_OUT%" >nul 2>&1

where 7z >nul 2>&1
if not errorlevel 1 (
    for /l %%i in (1,1,5) do (
        if exist "%Z_OUT%" del /q "%Z_OUT%" >nul 2>&1
        7z a -tzip "%Z_OUT%" "%Z_SRC%" -r >nul
        if not errorlevel 1 (
            set "ZIP_RC=0"
            goto :zip_done
        )
        echo [WARNING] 7z zip attempt %%i failed, retrying...
        timeout /t 2 /nobreak >nul
    )
) else (
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $src='%Z_SRC%'; $dst='%Z_OUT%'; for($i=1; $i -le 10; $i++){ try { if(Test-Path $dst){ Remove-Item $dst -Force -ErrorAction Stop }; Compress-Archive -Path $src -DestinationPath $dst -Force -ErrorAction Stop; exit 0 } catch { if(Test-Path $dst){ Remove-Item $dst -Force -ErrorAction SilentlyContinue }; if($i -eq 10){ Write-Error $_; exit 1 }; Write-Host ('[WARNING] Compress-Archive attempt ' + $i + ' failed, retrying...'); Start-Sleep -Seconds 2 } }"
    if not errorlevel 1 (
        set "ZIP_RC=0"
    )
)

:zip_done
if not "%ZIP_RC%"=="0" (
    echo [ERROR] Failed to create zip: %Z_OUT%
    exit /b 1
)

echo [SUCCESS] Created zip: %Z_OUT%
exit /b 0

:build_single
set "WORK_DIR=%ROOT_DIR%\build_!OUT_DIST_DIR!"
set "DIST_DIR=%ROOT_DIR%\!OUT_DIST_DIR!"

call :build_with_python "%PYTHON_EXE%" "!WORK_DIR!" "!DIST_DIR!" "release"
set "BUILD_RC=%ERRORLEVEL%"
echo [INFO] build_with_python rc: !BUILD_RC!
if !BUILD_RC! NEQ 0 exit /b !BUILD_RC!

if "%BUILD_ZIP%"=="1" (
    set "ZIP_NAME=!APP_NAME!_v!VERSION!.zip"

    if exist "!DIST_DIR!\!APP_NAME!\SuperPicky.iss" del /q "!DIST_DIR!\!APP_NAME!\SuperPicky.iss" >nul 2>&1
    if exist "!DIST_DIR!\!APP_NAME!\ChineseSimplified.isl" del /q "!DIST_DIR!\!APP_NAME!\ChineseSimplified.isl" >nul 2>&1

    call :zip_dir "!DIST_DIR!\!APP_NAME!" "!DIST_DIR!\!ZIP_NAME!"
    if errorlevel 1 exit /b 1

    if exist "%INNO_DIR%\SuperPicky.iss" (
        copy /y "%INNO_DIR%\SuperPicky.iss" "!DIST_DIR!\!APP_NAME!\SuperPicky.iss" >nul
        powershell -NoProfile -Command "(Get-Content -Path '!DIST_DIR!\!APP_NAME!\SuperPicky.iss' -Raw -Encoding UTF8) -replace 'VersionInfoVersion=.*', 'VersionInfoVersion=!VERSION!' | Set-Content -Path '!DIST_DIR!\!APP_NAME!\SuperPicky.iss' -Encoding UTF8"
    )
    if exist "%INNO_DIR%\ChineseSimplified.isl" (
        copy /y "%INNO_DIR%\ChineseSimplified.isl" "!DIST_DIR!\!APP_NAME!\ChineseSimplified.isl" >nul
    )

    if defined ZIP_COPY_DIR (
        set "TARGET_SUBDIR=%APP_NAME%_!VERSION!"
        set "TARGET_DIR=!ZIP_COPY_DIR!\!TARGET_SUBDIR!"
        if not exist "!ZIP_COPY_DIR!" mkdir "!ZIP_COPY_DIR!"
        if errorlevel 1 (
            echo [ERROR] Failed to create copy root dir: !ZIP_COPY_DIR!
            exit /b 1
        )
        if exist "!TARGET_DIR!" rd /s /q "!TARGET_DIR!"
        if exist "!TARGET_DIR!" (
            echo [ERROR] Failed to clean old target dir: !TARGET_DIR!
            exit /b 1
        )
        call :copy_dir "%DIST_DIR%\%APP_NAME%" "!TARGET_DIR!"
        if errorlevel 1 exit /b 1

        if exist "%INNO_DIR%\SuperPicky.iss" (
            copy /y "%INNO_DIR%\SuperPicky.iss" "!TARGET_DIR!\SuperPicky.iss" >nul
            if errorlevel 1 (
                echo [ERROR] Failed to copy SuperPicky.iss to target directory
                exit /b 1
            )
            echo [SUCCESS] Copied SuperPicky.iss to !TARGET_DIR!

            powershell -NoProfile -Command "(Get-Content -Path '!TARGET_DIR!\SuperPicky.iss' -Raw -Encoding UTF8) -replace 'VersionInfoVersion=.*', 'VersionInfoVersion=!VERSION!' | Set-Content -Path '!TARGET_DIR!\SuperPicky.iss' -Encoding UTF8"
            if errorlevel 1 (
                echo [ERROR] Failed to update version in SuperPicky.iss in target directory
                exit /b 1
            )
            echo [SUCCESS] Updated version in SuperPicky.iss to !VERSION! in target directory
        )

        if exist "%INNO_DIR%\ChineseSimplified.isl" (
            copy /y "%INNO_DIR%\ChineseSimplified.isl" "!TARGET_DIR!\ChineseSimplified.isl" >nul
            if errorlevel 1 (
                echo [ERROR] Failed to copy ChineseSimplified.isl to target directory
                exit /b 1
            )
            echo [SUCCESS] Copied ChineseSimplified.isl to !TARGET_DIR!
        )

        if exist "!TARGET_DIR!\SuperPicky.iss" del /q "!TARGET_DIR!\SuperPicky.iss" >nul 2>&1
        if exist "!TARGET_DIR!\ChineseSimplified.isl" del /q "!TARGET_DIR!\ChineseSimplified.isl" >nul 2>&1

        call :zip_dir "!TARGET_DIR!" "!ZIP_COPY_DIR!\!TARGET_SUBDIR!.zip"
        if errorlevel 1 exit /b 1

        if exist "%INNO_DIR%\SuperPicky.iss" (
            copy /y "%INNO_DIR%\SuperPicky.iss" "!TARGET_DIR!\SuperPicky.iss" >nul
            powershell -NoProfile -Command "(Get-Content -Path '!TARGET_DIR!\SuperPicky.iss' -Raw -Encoding UTF8) -replace 'VersionInfoVersion=.*', 'VersionInfoVersion=!VERSION!' | Set-Content -Path '!TARGET_DIR!\SuperPicky.iss' -Encoding UTF8"
        )
        if exist "%INNO_DIR%\ChineseSimplified.isl" (
            copy /y "%INNO_DIR%\ChineseSimplified.isl" "!TARGET_DIR!\ChineseSimplified.isl" >nul
        )

        echo [SUCCESS] Copied !TARGET_SUBDIR! + created !ZIP_COPY_DIR!\!TARGET_SUBDIR!.zip
    )
) else (
    set "ZIP_NAME="
    echo [INFO] ZIP creation skipped ^(--no-zip^)
)

echo.
echo [========================================]
echo Step 4: Copy Inno Setup files
echo [========================================]

set "OUTPUT_EXE_DIR=%DIST_DIR%\%APP_NAME%"

if exist "%INNO_DIR%\SuperPicky.iss" (
    copy /y "%INNO_DIR%\SuperPicky.iss" "%OUTPUT_EXE_DIR%\SuperPicky.iss" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy SuperPicky.iss
        exit /b 1
    )
    echo [SUCCESS] Copied SuperPicky.iss to %OUTPUT_EXE_DIR%

    powershell -NoProfile -Command "(Get-Content -Path '%OUTPUT_EXE_DIR%\SuperPicky.iss' -Raw -Encoding UTF8) -replace 'VersionInfoVersion=.*', 'VersionInfoVersion=%VERSION%' | Set-Content -Path '%OUTPUT_EXE_DIR%\SuperPicky.iss' -Encoding UTF8"
    if errorlevel 1 (
        echo [ERROR] Failed to update version in SuperPicky.iss
        exit /b 1
    )
    echo [SUCCESS] Updated version in SuperPicky.iss to %VERSION%
) else (
    echo [WARNING] SuperPicky.iss not found in %INNO_DIR%
)

if exist "%INNO_DIR%\ChineseSimplified.isl" (
    copy /y "%INNO_DIR%\ChineseSimplified.isl" "%OUTPUT_EXE_DIR%\ChineseSimplified.isl" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to copy ChineseSimplified.isl
        exit /b 1
    )
    echo [SUCCESS] Copied ChineseSimplified.isl to %OUTPUT_EXE_DIR%
) else (
    echo [WARNING] ChineseSimplified.isl not found in %INNO_DIR%
)

echo.
echo [========================================]
echo Build finished
echo [========================================]
echo EXE: %DIST_DIR%\%APP_NAME%\SuperPicky.exe
if defined ZIP_NAME (
echo ZIP: %DIST_DIR%\%ZIP_NAME%
if defined ZIP_COPY_DIR echo Copy: %ZIP_COPY_DIR%\%APP_NAME%_%VERSION% + .zip
) else (
echo ZIP: ^(skipped^)
)
exit /b 0

:restore_build_info
if exist "%BUILD_INFO_BACKUP%" (
    move /y "%BUILD_INFO_BACKUP%" "%BUILD_INFO_FILE%" >nul
)
exit /b 0
