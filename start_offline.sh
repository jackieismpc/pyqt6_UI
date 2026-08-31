#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_ROOT"

# Keep package/model resolution and telemetry strictly local.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export ULTRALYTICS_OFFLINE=1
export UV_OFFLINE=1
export UV_PYTHON_DOWNLOADS=never
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
export SAM2_BUILD_CUDA=0

PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv sync --offline --locked --no-python-downloads
        PYTHON_EXE="$PROJECT_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXE="$(command -v python)"
    else
        echo "No .venv, uv, or python found. Prepare an offline environment as described in README." >&2
        exit 1
    fi
fi

if [[ ! -x "$PYTHON_EXE" ]] && ! command -v "$PYTHON_EXE" >/dev/null 2>&1; then
    echo "Python interpreter is not available: $PYTHON_EXE" >&2
    exit 1
fi

PREFLIGHT_MODE=app
if [[ "${1:-}" == "--web" ]]; then
    PREFLIGHT_MODE=web
fi

"$PYTHON_EXE" "$PROJECT_ROOT/scripts/offline_preflight.py" --mode "$PREFLIGHT_MODE"
if [[ "${1:-}" == "--check" ]]; then
    exit 0
fi

exec "$PYTHON_EXE" "$PROJECT_ROOT/main.py" "$@"
