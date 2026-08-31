param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

# Keep package/model resolution and telemetry strictly local.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:WANDB_DISABLED = "true"
$env:WANDB_MODE = "disabled"
$env:ULTRALYTICS_OFFLINE = "1"
$env:UV_OFFLINE = "1"
$env:UV_PYTHON_DOWNLOADS = "never"
$env:PIP_NO_INDEX = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:SAM2_BUILD_CUDA = "0"

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    $Uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $Uv) {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $PythonCommand) {
            throw "No .venv, uv, or python found. Prepare an offline environment as described in README."
        }
        $PythonExe = $PythonCommand.Source
    }
    else {
        & $Uv.Source sync --offline --locked --no-python-downloads
        if ($LASTEXITCODE -ne 0) {
            throw "Offline uv sync failed: the local cache is incomplete; no network was attempted."
        }
        $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    }
}

$PreflightMode = "app"
if ($Arguments -contains "--web") {
    $PreflightMode = "web"
}

& $PythonExe (Join-Path $ProjectRoot "scripts\offline_preflight.py") --mode $PreflightMode
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Arguments -contains "--check") {
    exit 0
}

& $PythonExe (Join-Path $ProjectRoot "main.py") @Arguments
exit $LASTEXITCODE
