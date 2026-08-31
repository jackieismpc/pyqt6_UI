@echo off
setlocal
cd /d "%~dp0"

rem Keep package/model resolution and telemetry strictly local.
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "WANDB_DISABLED=true"
set "WANDB_MODE=disabled"
set "ULTRALYTICS_OFFLINE=1"
set "UV_OFFLINE=1"
set "UV_PYTHON_DOWNLOADS=never"
set "PIP_NO_INDEX=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "SAM2_BUILD_CUDA=0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PREFLIGHT_MODE=app"
set "CHECK_ONLY=0"
for %%A in (%*) do (
    if /I "%%~A"=="--web" set "PREFLIGHT_MODE=web"
    if /I "%%~A"=="--check" set "CHECK_ONLY=1"
)
if exist "%PYTHON_EXE%" goto preflight

where uv >nul 2>nul
if errorlevel 1 goto system_python
uv sync --offline --locked --no-python-downloads
if errorlevel 1 (
    echo Offline uv sync failed: the local cache is incomplete; no network was attempted.
    exit /b 1
)
if not exist "%PYTHON_EXE%" (
    echo uv sync completed, but .venv\Scripts\python.exe was not found.
    exit /b 1
)
goto preflight

:system_python
where python >nul 2>nul
if errorlevel 1 (
    echo No .venv, uv, or python found. Prepare an offline environment as described in README.
    exit /b 1
)
set "PYTHON_EXE=python"

:preflight
"%PYTHON_EXE%" "%~dp0scripts\offline_preflight.py" --mode %PREFLIGHT_MODE%
if errorlevel 1 exit /b %errorlevel%

if "%CHECK_ONLY%"=="1" exit /b 0

"%PYTHON_EXE%" "%~dp0main.py" %*
exit /b %errorlevel%
